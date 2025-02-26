import uproot  # need version uproot4
import numpy as np
import json
import logging
import numbers
import multiprocessing as mp
import re
from time import perf_counter
from tqdm import tqdm
from copy import deepcopy
from fnmatch import fnmatch
from itertools import zip_longest
from numexpr import evaluate as ne_evaluate
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial, lru_cache
from .core import HistogramBase, Histogram, Histogram2D
from .jitfunc import apply_phsp_correction, is_none_zero, parallel_nonzero_count

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _apply_phsp(weights, sumW2, phsp, phsp_err):
    # sumW2 *= phsp**2
    # sumW2 += (weights * phsp_err) ** 2
    # weights *= phsp
    apply_phsp_correction(weights, sumW2, phsp, phsp_err)


def histogram_eval(event, mask, *observables):
    return [ne_evaluate(obs, event)[mask] for obs in observables]


def fill_histogram(hist, mask, event, weights, sumW2=None):
    """
    Default approach to fill single histogram
    """
    if hist.weights:
        hist_w = ne_evaluate(hist.weights, event)[mask]
    else:
        hist_w = weights[mask]
    fdata = histogram_eval(event, mask, *hist.observable)
    hist.from_array(*fdata, hist_w, sumW2[mask] if sumW2 is not None else None)
    return hist


def weight_from_hist(events, obs, hist):
    """
    get the correspoding weight in weighting histogram for array of events.

    Args:
        events: (dict(str:numpy.array))
            Dict of numpy arrays. the keys are labeled as observables.

        obs: (str or tuple)
            Name of the observables.
            If tuple is provided, the dimension should match the hist

        hist : core.Histogram or core.Histogram2D.
            Can be ROOT or collinearw.Histogram.
            numpy array will also work. Basically it will find the weight given
            by the histogram for each event.
    """
    if isinstance(hist, Histogram):
        if not isinstance(obs, str):
            raise TypeError("1D Histogram obs need to be str.")
        # m_events = ne_evaluate(obs, events)
        # _digitized = np.digitize(m_events, hist.bins)
        # content = hist.get_bin_content(_digitized)
        # error = hist.get_bin_error(_digitized)
        # return content, error
        return hist.find_content_and_error(ne_evaluate(obs, events))
    elif isinstance(hist, Histogram2D):
        m_events_x = ne_evaluate(obs[0], events)
        m_events_y = ne_evaluate(obs[1], events)
        # _digitized_x = np.digitize(m_events_x, hist.bins[0])
        # _digitized_y = np.digitize(m_events_y, hist.bins[1])
        # index_list = list(zip(_digitized_x, _digitized_y))
        # content = hist.get_bin_content_list(index_list)
        # error = hist.get_bin_error_list(index_list)
        # return content, error
        return hist.find_content_and_error(m_events_x, m_events_y)
    else:
        raise TypeError("weight_hist dimension is not implemented")


def weight_from_region(events, rname, process, weight_hist):
    _w_r = process.get_region(rname)
    if isinstance(weight_hist, tuple):
        _obs = weight_hist[0]
        _w_hist = _w_r.get_histogram(weight_hist[1])
        return weight_from_hist(events, _obs, _w_hist)
    elif isinstance(weight_hist, dict):
        for key in weight_hist:
            if not fnmatch(rname, key):
                continue
            m_weight_hist = weight_hist[key]
            if not isinstance(m_weight_hist, list):
                # sinlge weighting hist case
                _obs = m_weight_hist[0]
                _w_hist = _w_r.get_histogram(m_weight_hist[1])
                return weight_from_hist(events, _obs, _w_hist)
            cml_w = None  # cumulative
            cml_err = None
            for w_hist_set in m_weight_hist:
                _obs = w_hist_set[0]
                _w_hist = _w_r.get_histogram(w_hist_set[1])
                if cml_w is None:
                    cml_w, cml_err = weight_from_hist(events, _obs, _w_hist)
                else:
                    _weights, _err = weight_from_hist(events, _obs, _w_hist)
                    # (cml_w*_err)**2 + (cml_err*_weights)**2
                    cml_err += (cml_err * _weights) ** 2
                    cml_err += (cml_w * _err) ** 2
                    cml_w *= _weights
            return cml_w, cml_err
        logger.warning("Failed to find weights from region, return (1.0, 0.0)")
        return 1.0, 0.0  # return 1 and 0 error
    else:
        raise TypeError(f"Invalid type weight_hist:{type(weight_hist)}")


@lru_cache(maxsize=None)
def get_cutbook_sum_weights(file_name, dataset_ids, run_numbers):
    """
    Retrieve the sum of weights from the cutbook for the specified dataset IDs and run numbers.
    TODO : need to hanlde cases with systematics

    Args:
        file_name (str): The name of the ROOT file to open.
        dataset_ids (iterable): An iterable of dataset IDs.
        run_numbers (iterable): An iterable of run numbers.

    Returns:
        dict: A dictionary where keys are (dataset_id, run_number) tuples and values are the sum of weights.
    """
    with uproot.open(file_name) as root_file:
        sum_weights = {}
        for dsid, run in zip(dataset_ids, run_numbers):
            cutbook_name = f"CutBookkeeper_{dsid}_{run}_NOSYS"
            sum_weights[(dsid, run)] = root_file[cutbook_name].values()[1]
        return sum_weights


def get_sum_weights(dsid, run_number=None, tfile_name=None, sum_weights_dict=None):
    """
    Retrieve the sum of weights for the given DSID and run number from the specified TFile.

    This function retrieves the sum of weights for a batch of events, allowing for optional
    use of a pre-computed sum of weights dictionary. It can handle unique DSID and run number
    combinations, and optionally uses a caching mechanism for performance optimization.
    TODO : need to handle cases with systematics

    Args:
        dsid (ndarray): Array of dataset IDs (DSID) for the events.
        run_number (ndarray, optional): Array of run numbers for the events.
            If None, only DSID is used to retrieve weights.
        tfile_name (str, optional): The name of the TFile containing weight information.
            If None, the function will use a pre-computed sum of weights dictionary.
        sum_weights_dict (dict, optional): A dictionary of pre-computed sum of weights
            with keys as (dsid, run_number) tuples. Defaults to None.

    Returns:
        ndarray: An array of the sum of weights for each event, corresponding to the input DSID
        and run number.
    """

    if tfile_name is None and sum_weights_dict is None:
        raise ValueError("Either tfile_name or sum_weights_dict must be provided.")

    if run_number is not None:
        unique_pairs = np.unique((dsid, run_number), axis=1)
        if sum_weights_dict is None:
            sum_weights_dict = get_cutbook_sum_weights(
                tfile_name, tuple(unique_pairs[0]), tuple(unique_pairs[1])
            )
        idx = np.searchsorted(unique_pairs[0], dsid)
        idx_run = np.searchsorted(unique_pairs[1], run_number)
        return np.take(
            list(sum_weights_dict.values()),
            np.ravel_multi_index(
                (idx, idx_run), (len(unique_pairs[0]), len(unique_pairs[1]))
            ),
        )

    if sum_weights_dict is None:
        raise ValueError("sum_weights_dict is required when run_number is None")

    unique_dsid, inverse_mask = np.unique(dsid, return_inverse=True)
    sum_weights = np.array([sum_weights_dict[d] for d in unique_dsid])
    return sum_weights[inverse_mask]


def sum_weights_from_files(tfile_names):
    sum_weights = {}
    for file in tfile_names:
        with uproot.open(file) as root_file:
            for obj_name in root_file.keys():
                match = re.match(r"CutBookkeeper_(\d+)_(\d+)_NOSYS", obj_name)
                if match:
                    dsid, run = map(int, match.groups())
                    sum_weights.setdefault(dsid, 0.0)
                    sum_weights[dsid] += root_file[obj_name].values()[1]
    return sum_weights


class MatchHelper:
    @staticmethod
    def region_name(name):
        def _region_name(region):
            return fnmatch(region.name, name)

        return _region_name

    @staticmethod
    def region_type(type):
        def _region_type(region):
            return type == region.type

        return _region_type


class HistMaker:
    """
    HistMaker class:
        handles histograms creation and preparation, with given Process and Region.

    """

    def __init__(
        self,
        histogram_backend="numpy",
        *,
        nthread=None,
        use_mmap=False,
        disable_child_thread=True,
        xsec_sumw=None,
    ):
        self._corrections = None
        self._entry_start = 0
        self._entry_stop = -1
        self._tree_ibatch = 0
        self._is_init = False
        self._is_close = False
        self._region_weight_tracker = {}
        self.histogram_backend = histogram_backend
        self.disable_pbar = False
        self.use_mmap = use_mmap
        self.phasespace_corr_obs = ["nJet30"]
        self.phasespace_apply_nominal = True
        self.phsp_fallback = True
        self.default_weight = None
        self.enforce_default_weight = False
        self.branch_list = None
        self.RAISE_TREENAME_ERROR = True
        self.step_size = "50MB"
        self.err_prop = True
        self.branch_rename = None  # dict for swapping branch name in the ntuple
        self.hist_fill_type = None
        self.hist_fill_executor = None
        self.xsec_sumw = xsec_sumw  # cross section and sum of event weights
        self.fill_file_status = None
        self.skip_dummy_processes = None
        self.use_cutbook_sum_weights = False
        self.acc_cutbook_sum_weights = False
        self._cutbook_sum_weights_dict = None
        self.dsid_branch = None
        self.run_number_branch = None

        self.f_decompression_executor = None
        self.f_interpretation_executor = None
        if mp.current_process().name != "MainProcess" and disable_child_thread:
            self.nthread = 1
        else:
            self.nthread = nthread or int(np.ceil(0.5 * mp.cpu_count()))

    def __del__(self):
        self.finalise()

    def initialize(self):
        """
        reserve for other initization setting.
        """
        if self._is_init:
            return None
        self._is_init = True

    def finalise(self):
        """
        Default finalise call to cleanup if needed
        """
        if self._is_close:
            return None
        if self.hist_fill_executor:
            self.hist_fill_executor.shutdown()
            self.hist_fill_executor = None
        if self.f_decompression_executor:
            self.f_decompression_executor.shutdown()
            self.f_decompression_executor = None
        if self.f_interpretation_executor:
            self.f_interpretation_executor.shutdown()
            self.f_interpretation_executor = None
        self._is_close = True

    def meta_data_from_config(self, config):
        """
        load meta data from configMgr object

        useful for commonly used feature. if you want to pass objects in config
        to histmaker, you can add to this function. this can help reduce the chance
        that if you have new feature to histmaker and you forget to propagate it into
        other methods.
        """
        self._corrections = config.corrections
        self.default_weight = config.default_weight
        self.branch_list = config.reserved_branches
        self.phasespace_corr_obs = config.phasespace_corr_obs
        self.phasespace_apply_nominal = config.phasespace_apply_nominal
        self.enforce_default_weight = config.enforce_default_weight
        self.RAISE_TREENAME_ERROR = config.RAISE_TREENAME_ERROR
        self.branch_rename = config.branch_rename or {}
        self.disable_pbar = config.disable_pbar
        self.xsec_sumw = config.xsec_sumw
        self.skip_dummy_processes = config.skip_dummy_processes
        self.use_cutbook_sum_weights = config.use_cutbook_sum_weights
        self.acc_cutbook_sum_weights = config.acc_cutbook_sum_weights
        self.dsid_branch = config.dsid_branch
        self.run_number_branch = config.run_number_branch

    def copy(self):
        return deepcopy(self)

    def open_file(self, file_name, use_mmap=True, *args, **kwargs):
        self.initialize()
        # note: executor attached to file will be shutdown automatically
        if self.f_decompression_executor is None:
            if self.nthread > 1:
                self.f_decompression_executor = ThreadPoolExecutor(self.nthread)
            else:
                self.f_decompression_executor = uproot.TrivialExecutor()
        if self.f_interpretation_executor is None:
            self.f_interpretation_executor = uproot.TrivialExecutor()
        kwargs.setdefault("decompression_executor", self.f_decompression_executor)
        kwargs.setdefault("interpretation_executor", self.f_interpretation_executor)
        opts = {}
        # kwargs.setdefault("array_cache", None)
        # kwargs.setdefault("object_cache", None)
        # opts.update({"num_fallback_workers": self.nthread})
        opts.update({"num_workers": self.nthread})
        if "xrootd_handler" not in kwargs:
            opts.update({"xrootd_handler": uproot.MultithreadedXRootDSource})
        if use_mmap or self.use_mmap:
            kwargs.update(opts)
        else:
            kwargs.setdefault("file_handler", uproot.MultithreadedFileSource)
            # kwargs.setdefault("num_workers", 2)
            # kwargs.setdefault("executor", self.f_decompression_executor)
            kwargs.update(opts)
        return uproot.open(file_name, *args, **kwargs)

    def raw_open(self, *args, **kwargs):
        return uproot.open(*args, **kwargs)

    def _make_weight_func(self, weight_func_file=None):
        """
        get the weight with a given observable. The weight function is generated elsewhere.

        Depreciated
        """
        if weight_func_file is None:
            return None
        else:
            if ".json" in weight_func_file:
                with open(weight_func_file) as dfile:
                    data = json.load(dfile)
                    return ("json", data["bins"], data["bin_content"])
            return None

    def _make_weight_gen(self, weight_function):
        """
        Depreciated
        """
        if weight_function[0] == "json":

            def weight_gen(observable_array):
                observable_digitized = np.digitize(observable_array, weight_function[1])
                weight = [
                    weight_function[2][w_bin - 1][0] for w_bin in observable_digitized
                ]
                return np.array(weight)

            return weight_gen
        return None

    def skip_dummy(self, p):
        is_dummy = getattr(p.systematic, "source", None) == "dummy"
        if not is_dummy:
            return False
        if self.skip_dummy_processes and p.name in self.skip_dummy_processes:
            logger.info(f"skip dummy for {(p.name, p.systematic)}")
            return True
        return False

    def process_weight_gen(
        self, configMgr, observable_name, weight_func_file, scale_factor
    ):
        """
        only gose into TH1 for now.
        """
        weight_func = self._make_weight_func(weight_func_file)
        if weight_func is None:
            logger.warning("cannot run process with weight.")
        else:
            weight_gen = self._make_weight_gen(weight_func)
            if weight_gen:
                self.process(
                    configMgr, ext_rweight=(weight_gen, observable_name, scale_factor)
                )
            else:
                logger.warning("cannot get weight generator.")

    def process_ttree_resolve(self, tfile, p):
        """
        Resolving TTree lookup for a given Process object. Systematic name and
        weight are also being unpack here.

        Args:
            tfile: uproot.ReadOnlyFile
                TFile opened by the uproot interface.

            p: core.Process
                instance of core.Process class

        Return:
            ttree, systematic full name, systematic weight
        """

        # if the systematic source is dummy, treat it as nominal, and the
        # systematic is just a place holder. It's usefull for phasespace
        # correction when systematic is involved.
        is_dummy = getattr(p.systematic, "source", None) == "dummy"
        if p.systematic is None or is_dummy:
            sys_weight = None
            swap_weight = None
            if is_dummy:
                sys_name = p.systematic.full_name
                logger.warning(
                    ", ".join(
                        [
                            f"'dummy' source is specified in {sys_name}",
                            f"{p.name} will be filled with nominal tree.",
                        ]
                    )
                )
            else:
                sys_name = None

            # if cross section and sum weight setting is specified,
            # spcecial filename and treename check will be performed.
            # otherwise old routine would be used.
            if self.xsec_sumw:
                if not self.xsec_sumw.nominal(tfile.file_path):
                    return None, sys_name, sys_weight, swap_weight
                # Warning: xsec_sumw cannot tell the difference between
                # different processes since they are all tree_*
                # the xsec_sumw.check_process just does a basic name comparison.
                self.xsec_sumw.match(tfile.file_path)
                if not self.xsec_sumw.check_process(p.name):
                    return None, sys_name, sys_weight, swap_weight
                if self.xsec_sumw.use_process_nominal_tree:
                    treename = p.treename
                else:
                    treename = "tree_NoSys"
            else:
                treename = f"{p.name}_NoSys"

            if treename in tfile:
                return tfile[treename], sys_name, sys_weight, swap_weight

            # fall back option to use process defined treename
            if p.treename in tfile:
                return tfile[p.treename], sys_name, sys_weight, swap_weight

            if self.RAISE_TREENAME_ERROR:
                raise RuntimeError(f"unable to get nominal tree from {p.name}")

            logger.critical(
                ", ".join(
                    [
                        f"{self.RAISE_TREENAME_ERROR=}",
                        f"tree lookup failed for process {p.name}",
                        f"skipping base tree {p.treename} in {tfile.file_path}",
                    ]
                )
            )
            return None, sys_name, sys_weight, swap_weight

        sys_name, sys_weight = p.systematic.full_name, p.systematic.weight
        swap_weight = p.systematic.swap_weight

        if self.xsec_sumw:
            self.xsec_sumw.match(tfile.file_path)
            if not self.xsec_sumw.check_process(p.name):
                return None, sys_name, sys_weight, swap_weight
            tree_prefix = "tree"
        else:
            tree_prefix = p.treename

        if "NoSys" in tree_prefix:
            treename = f"{tree_prefix.replace('NoSys', p.systematic.treename)}"
        else:
            treename = f"{tree_prefix}_{p.systematic.treename}"

        if treename not in tfile:
            if self.RAISE_TREENAME_ERROR:
                raise RuntimeError(f"cannot find {treename}")
            logger.warning(
                ", ".join(
                    [
                        f"{self.RAISE_TREENAME_ERROR=}",
                        f"tree lookup failed for process {p.name}",
                        f"skiping base tree {p.treename} in {tfile.file_path}",
                        f"systematic {sys_name}",
                        f"systematic weight {sys_weight}",
                    ]
                )
            )
            return None, sys_name, sys_weight, swap_weight

        # check for weight base systematics
        ttree = tfile[treename]
        if sys_weight:
            if self.branch_rename and sys_weight in self.branch_rename.values():
                for old_b, new_b in self.branch_rename.items():
                    if sys_weight == new_b and old_b not in ttree:
                        return None, sys_name, sys_weight, swap_weight
            elif sys_weight not in ttree:
                logger.debug(f"{sys_weight} is not in {treename} of {tfile.file_path}")
                return None, sys_name, sys_weight, swap_weight
            if self.xsec_sumw:
                # remove LHE3Weight in multiple files, maybe there's better way?
                # LHE3Weight, leptonWeight, etc are found in all files??
                # duplicated_list = {"LHE3Weight", "pileupWeight", "triggerWeight"}
                duplicated_list = self.xsec_sumw.duplicated_list
                do_check = np.any([x in sys_weight for x in duplicated_list])
                if do_check:
                    skip_campaign = self.xsec_sumw.duplicated_skip_campaign or {}
                    accept_list = self.xsec_sumw.duplicated_accept or set()
                    # if syst is one of the accept list, check campaign
                    if self.xsec_sumw['syst'] in accept_list:
                        # this duplicated_skip_campaign.get(self.xsec_sumw['campaign'], None)
                        # return a syst, e.g. 'pileupWeight'
                        skip_weight = skip_campaign.get(
                            self.xsec_sumw['campaign'], None
                        )
                        if skip_weight and skip_weight in sys_weight:
                            do_check = False
                if do_check:
                    _check_set = None
                    if self.xsec_sumw.duplicated_sets:
                        # this is a map, e.g. {'triggerWeight' : 'weight_2'}
                        # where the key is used to match the item in
                        # self.xsec_sumw.duplicated_list, then apply the matching
                        # of systematic set. If None of them were found, matching
                        # will be done with nominal.
                        for _syst_set in self.xsec_sumw.duplicated_sets:
                            if _syst_set in sys_weight:
                                _check_set = self.xsec_sumw.duplicated_sets[_syst_set]
                                break
                    # note if _check_set is None, then this is same as
                    # self.xsec_sumw.nominal(tfile.file_path)
                    if not self.xsec_sumw.check_syst_set(tfile.file_path, _check_set):
                        return None, sys_name, sys_weight, swap_weight

                # check for ptag to remove for a given weight base sysetmatic
                # This only does a generic 'in' check, so 'bTagWeight' can filter
                # any 'bTagWeight_*' systematics. Similar for p-tag. e.g. 'p4512'
                if self.xsec_sumw.remove_wsyst_ptag:
                    _tag_proc = self.xsec_sumw.remove_wsyst_ptag.get(
                        self.xsec_sumw['process'], None
                    )
                    if _tag_proc is not None:
                        campaign = _tag_proc.get(self.xsec_sumw['campaign'], None)
                        if campaign is not None:
                            for _wsyst, _ptag in campaign.items():
                                if (
                                    _wsyst in sys_weight
                                    and _ptag in self.xsec_sumw['ptag']
                                ):
                                    return None, sys_name, sys_weight, swap_weight

        return ttree, sys_name, sys_weight, swap_weight

    def parse_process_weights(self, p, syst_w, swap_w=None):
        """
        Args:
            syst_w : str
                systematic weight string

            swap_w : str
                weight string being swap with syst_w
        """
        if p.weights:
            if isinstance(p.weights, list):
                process_weights = "*".join(p.weights)
            else:
                process_weights = p.weights
        else:
            process_weights = None

        # NOTE: if swap_w is specified, the systematic weight will not be set
        # here. Swaping is expecting instead of appending.
        swap_w_dict = {}
        if syst_w:
            if swap_w is None:
                if process_weights:
                    # just append to the existing weight string
                    process_weights += f"*{syst_w}"
                else:
                    process_weights = syst_w
            else:
                swap_w_dict[swap_w] = syst_w

        logger.debug(f"Process level weights: {process_weights}")

        return process_weights, swap_w_dict

    def parse_region_weights(self, r, process_w=None, swap_w_dict=None):
        """
        Args:
            process_w : str
                process level weight string.

            swap_w_dict : {str : str}
                dictionary for swapping weight strings.
        """

        if r.name in self._region_weight_tracker:
            return self._region_weight_tracker[r.name]

        # obtaining process and region level weights
        # if none of them were found, try to use the
        # histmaker default weight. i.e self.default_weight
        r_weights = None
        if process_w is None and r.weights is None:
            if self.default_weight:
                r_weights = self.default_weight
        else:
            if r.weights:
                if isinstance(r.weights, list):
                    r_weights = "*".join(r.weights)
                else:
                    r_weights = r.weights
                if process_w:
                    r_weights += f"*{process_w}"
            elif process_w:
                r_weights = process_w

            # check if user enforce to use default weight
            if self.enforce_default_weight and self.default_weight:
                if r_weights:
                    r_weights += f"*{self.default_weight}"
                else:
                    r_weights = self.default_weight

        if swap_w_dict and r_weights:
            for old_w, new_w in swap_w_dict.items():
                r_weights = r_weights.replace(old_w, new_w)

        self._region_weight_tracker[r.name] = r_weights

        return r_weights

    def phase_space_correction(
        self,
        event,
        process,
        correction_type,
        correction_variable=None,
        systematic=None,
    ):
        if not self._corrections:
            return None, None

        if correction_variable is None:
            correction_variable = self.phasespace_corr_obs
            if not isinstance(correction_variable, list):
                correction_variable = [correction_variable]

        # if enable, force to use nominal
        if self.phasespace_apply_nominal:
            systematic = None

        weight = None
        error = None
        for corr_obs in correction_variable:
            lookup = (correction_type, process, corr_obs, systematic)
            h = self._corrections[lookup]
            if h is None:
                logger.debug(f"Cannot find {lookup} for phase space correction")
                continue
            else:
                logger.debug(f"Found {lookup} for phase space correction")
                logger.debug(f"Found correction type {type(h)}")
            if isinstance(h, (Histogram, Histogram2D)):
                m_weight, m_error = weight_from_hist(event, corr_obs, h)
            elif isinstance(h, numbers.Number):
                m_weight = h
                m_error = 0.0
            else:
                logger.warning(f"Depreciated method using {type(h)}")
                nJetArray = event[corr_obs]
                m_weight = [h.GetBinContent(h.FindBin(e)) for e in nJetArray]
                m_error = [h.GetBinError(h.FindBin(e)) for e in nJetArray]
                m_weight, m_error = np.array(m_weight), np.array(m_error)
            if weight is None:
                weight = m_weight
                error = m_error
            else:
                # (A*err_B)**2 + (err_A*B)**2
                error += (error * m_weight) ** 2
                error += (weight * m_error) ** 2
                weight *= m_weight

        return weight, error

    def _histogram_loop(self, histograms, mask, event, weights, sumW2):
        """
        default loop for list of histograms
        User external histogram loop should use the same expect input arguments
        """
        for hist in histograms:
            # check if there's specified weights in the histogram
            # overwrite the weights defined in process/region/default
            if hist.weights:
                hist_w = ne_evaluate(hist.weights, event)[mask]
            else:
                hist_w = weights[mask]
            fdata = histogram_eval(event, mask, *hist.observable)
            hist.from_array(*fdata, hist_w, sumW2[mask] if self.err_prop else None)

    def _hist_loop_threadpool(self, histograms, mask, event, weights, sumW2):
        """
        multithread version of the _histogam_loop
        """
        n = sum(h.hist_type == "1d" for h in histograms)
        if n > 2:
            n = min(n, 12)  # hard coded maximum thread
        else:
            # fall back to regular histogram loop if n is <= 2
            self._histogram_loop(histograms, mask, event, weights, sumW2)
            return
        if self.hist_fill_executor is None:
            self.hist_fill_executor = ThreadPoolExecutor(n)
        for hist in histograms:
            if hist.hist_type != "1d":
                fill_histogram(hist, mask, event, weights, sumW2)
                continue
            self.hist_fill_executor.submit(
                fill_histogram,
                hist,
                mask,
                event,
                weights,
                sumW2 if self.err_prop else None,
            )

    def _hist_loop_processpool(self, histograms, mask, event, weights, sumW2):
        """
        multitprocess version of the _histogam_loop
        """
        n = sum(h.hist_type == "1d" for h in histograms)
        if n > 2:
            n = min(n, 8)  # hard coded maximum thread
        else:
            # fall back to regular histogram loop if n is <= 2
            self._histogram_loop(histograms, mask, event, weights, sumW2)
            return
        futures = []
        ft_index = []
        if self.hist_fill_executor is None:
            self.hist_fill_executor = ProcessPoolExecutor(n)
        for id, hist in enumerate(histograms):
            if hist.hist_type != "1d":
                fill_histogram(hist, mask, event, weights, sumW2)
                continue
            ft = self.hist_fill_executor.submit(
                fill_histogram,
                hist,
                mask,
                event,
                weights,
                sumW2 if self.err_prop else None,
            )
            futures.append(ft)
            ft_index.append(id)
        for id, ft in zip(ft_index, futures):
            hist = histograms[id]
            filled_hist = ft.result()
            assert hist.name == filled_hist.name
            hist.add(filled_hist)

    def plevel_process(self, p, *args, **kwargs):
        if p.combine_tree:  # combine_tree need to be iterable
            c_p = p.copy()
            c_p.clear_content()
            for treename in p.combine_tree:
                if isinstance(treename, tuple):
                    c_p.treename = treename[0]
                    c_p.selection = treename[1]
                elif isinstance(treename, dict):
                    for key, value in treename.items():
                        setattr(c_p, key, value)
                else:
                    c_p.treename = treename
                    c_p.selection = None
                self._plevel_process(c_p, *args, **kwargs)
            p.add(c_p)
        return self._plevel_process(p, *args, **kwargs)

    def _plevel_process(
        self,
        p,
        file_name,
        *,
        branch_list=None,
        weight_generators=[],
        weight_generators_filters=[],
        ext_rweight=None,
        ext_pweight=None,
        step_size=None,
        copy=False,
        histogram_method=None,
    ):
        """
        filling process level.

        Args:
            file_name : str
                path to the ROOT file.

            p : core.Process
                An instance of core.Process object.

            branch_list : list(str)
                list of branch names.

            weight_generators : [functools.partial] or [ lambda ]
                list of weight generators that take event (dict(str:np.array))
                as direct input and return weight array.

            weight_generators_filters : [ tuple(str) ]
                filtering and matching the correspoding weight generator in
                weight_generators. The first element is the method in MatchHelper,
                and the rest are arguments. e.g ('region_name', 'electron'),
                this will use MatchHelper.region_name('electron')

            ext_rweight :
                a function that accept event and mask,
                or a dict of {"process", "obs", "weight_hist"}

            ext_pweight :
                external weight for process. [(str, str, HistogramBase)]
                e.g. [(process_name, region_name, HistogramBase object)]
        """

        # replace the global decompression_executor and interpretation_executor
        # with 2 threads executor if it is not the main process
        # this avoid the hang if there's ProcessPoolExecutor in the external layer
        # if gettting into MemoryError: Unable to allocate on SLAC, try limit on setup size
        # step_size = "100MB"
        '''
        if executor is None:
            if mp.current_process().name != "MainProcess":
                executor = ThreadPoolExecutor(1)
            else:
                executor = ThreadPoolExecutor(5)
        else:
            pass
        '''
        # step_size = 10000

        phsp_lookup = self.phase_space_correction
        phsp_fallback = self.phsp_fallback and not self.phasespace_apply_nominal
        parse_process_weights = self.parse_process_weights
        parse_region_weights = self.parse_region_weights
        # open_file = self.open_file
        open_file = uproot.open
        t_start = perf_counter()

        with open_file(file_name) as tfile:
            ttree, sys_name, sys_weight, swap_w = self.process_ttree_resolve(tfile, p)

            # check ttree and number of entry. return early if it's zero
            if ttree is None or ttree.num_entries == 0:
                return p.copy() if copy else p

            # clear existing region weights tracker
            self._region_weight_tracker = {}

            # check if external histogram looping is provided
            if histogram_method:
                histogram_loop = histogram_method
            elif self.hist_fill_type == "thread":
                histogram_loop = self._hist_loop_threadpool
            elif self.hist_fill_type == "process":
                histogram_loop = self._hist_loop_processpool
            else:
                histogram_loop = self._histogram_loop

            logger.debug(f"retrieved {ttree.name}")
            logger.debug(f"opened {file_name}")
            logger.debug(f"ext_rweight: {ext_rweight}")
            has_corr = "with correction" if self._corrections else ""

            p_weights, swap_w_dict = parse_process_weights(p, sys_weight, swap_w)

            # try to get branches from the Process instance, and the branches of
            # regions within it.
            p_branch = p.ntuple_branches.copy()
            for r in p.regions:
                p_branch |= r.ntuple_branches

            branch_filter = branch_list or p_branch or self.branch_list
            # if branch renaming is requested, we need to make sure the original
            # names are used for branch filtering.
            if self.branch_rename:
                for old_bname, new_bname in self.branch_rename.items():
                    if new_bname not in branch_filter:
                        continue
                    branch_filter.discard(new_bname)
                    branch_filter.add(old_bname)

            if self.xsec_sumw:
                branch_filter.add(self.xsec_sumw.dsid)
                # self.xsec_sumw.match(file_name) # matched in process_ttree_resolve
                evt_w = self.xsec_sumw.get_auto()
            else:
                evt_w = None

            if self.use_cutbook_sum_weights:
                if self.dsid_branch:
                    branch_filter.add(self.dsid_branch)
                if self.run_number_branch:
                    branch_filter.add(self.run_number_branch)

            with tqdm(
                desc=f"Processing {p.name}|{sys_name or 'nominal'} events {has_corr}",
                total=ttree.num_entries,
                leave=False,
                unit="events",
                dynamic_ncols=True,
                disable=self.disable_pbar,
            ) as pbar_events:
                for event, report in ttree.iterate(
                    step_size=step_size or self.step_size,
                    filter_name=branch_filter,
                    report=True,
                    library="np",
                ):
                    self._entry_start = report.tree_entry_start
                    self._entry_stop = report.tree_entry_stop
                    self._tree_ibatch += 1
                    nevent = report.tree_entry_stop - report.tree_entry_start
                    pbar_events.set_description(f"Processing {nevent} events")

                    # renaming the branches
                    if self.branch_rename:
                        for old_bname, new_bname in self.branch_rename.items():
                            if old_bname not in event:
                                continue
                            event[new_bname] = event.pop(old_bname)

                    # all_mask is a mask with only process level selection
                    # if no process level seletion, accept all events.
                    # the mask is array of True/False.
                    if p.selection_numexpr:
                        all_mask = ne_evaluate(p.selection_numexpr, event)
                        if not is_none_zero(all_mask):
                            logger.debug("No event after process selection")
                            pbar_events.update(nevent)
                            continue
                    else:
                        all_mask = np.full(nevent, True)

                    '''
                    # assuming all the dsid are the same within single file
                    if self.xsec_sumw:
                        dsid = np.unique(event[self.xsec_sumw.dsid])
                        assert len(dsid) == 1
                        dsid = str(dsid[0])
                        evt_w = self.xsec_sumw.get_xsec_sumw(dsid, "weight_1")
                    else:
                        evt_w = None
                    '''

                    pbar_regions = tqdm(
                        p.regions,
                        leave=False,
                        unit="regions",
                        disable=self.disable_pbar,
                    )
                    for r in pbar_regions:
                        pbar_regions.set_description(
                            f"{p.name}, Region: {r.name}({len(r.histograms)})"
                        )

                        # setting process level and region level selection
                        # this is basically cuts in ROOT TTree but in numpy format
                        p_r_selection = []
                        if p.selection_numexpr:
                            p_r_selection.append(f"({p.selection_numexpr})")
                        if r.selection_numexpr:
                            p_r_selection.append(f"({r.selection_numexpr})")

                        # combinding process and region level selections
                        if p_r_selection:
                            selection_str = "&".join(p_r_selection)
                            selection_str = selection_str.replace("()", "")
                            selection_str = selection_str.strip().strip("&")
                        else:
                            selection_str = ""

                        # if no selection string is found, assume accepting all values
                        if selection_str:
                            mask = ne_evaluate(selection_str, event)
                        else:
                            logger.debug(
                                f"empty seleciton on region {r.name}. Assume no selection."
                            )
                            mask = all_mask

                        # just storing the region and process level
                        # selection to the region object
                        if r._full_selection is None:
                            r._full_selection = selection_str
                            logger.debug(
                                f"full selection (region+process) for {r.name}: {r._full_selection}"
                            )

                        # go to next iteration if no events passed both
                        # process and region level selection
                        non_zero_count = parallel_nonzero_count(mask)
                        if non_zero_count == 0:
                            logger.debug("no event after region selection.")
                            continue

                        # setting the weights to the size of nevent
                        # and default the value to one
                        weights = np.ones(nevent)
                        sumW2 = np.zeros(nevent)

                        if self.use_cutbook_sum_weights:
                            evt_dsid = ne_evaluate(self.dsid_branch, event)
                            if self.acc_cutbook_sum_weights:
                                weights /= get_sum_weights(
                                    evt_dsid,
                                    sum_weights_dict=self._cutbook_sum_weights_dict,
                                )
                            else:
                                evt_num = ne_evaluate(self.run_number_branch, event)
                                weights /= get_sum_weights(evt_dsid, evt_num, file_name)

                        # combine region and process weights
                        r_weights = parse_region_weights(r, p_weights, swap_w_dict)
                        if r_weights:
                            weights *= ne_evaluate(r_weights, event)

                        # get the pure event count without any region selection.
                        # bascially this is the number of events after process selection
                        r.total_event_count += np.sum(weights[all_mask])

                        if p.lumi:  # scale with process level lumi if there's any
                            weights *= p.lumi

                        if evt_w is not None:
                            weights *= evt_w

                        # compute w2 for each event
                        sumW2 += weights**2

                        # Apply a phase-space correction factor to a given process
                        phsp, phsp_err = phsp_lookup(
                            event, p.name, r.corr_type, systematic=sys_name
                        )
                        if phsp is None:
                            # if the above is not found, try lookup based on treename
                            # temperoraly use for iterative corrction
                            # might need a better approach
                            phsp, phsp_err = phsp_lookup(
                                event, p.treename, r.corr_type, systematic=sys_name
                            )
                        if phsp is None and phsp_fallback:
                            phsp, phsp_err = phsp_lookup(event, p.name, r.corr_type)
                        if phsp is not None:
                            _apply_phsp(weights, sumW2, phsp, phsp_err)

                        # handle external weight
                        if ext_rweight:
                            # this excpet ext_rweight contains external Process object
                            # this will apply weight on every process and (filter) regions
                            if isinstance(ext_rweight, dict):
                                _rname = ext_rweight.get("region", r.name)
                                ext_w, ext_err = weight_from_region(
                                    event, _rname, **ext_rweight
                                )
                            # this expect ext_rweight is a method that directly
                            # call event
                            else:
                                ext_w = ext_rweight(event)
                                ext_err = np.ones(ext_w.shape)
                            sumW2 *= ext_w**2
                            sumW2 += (weights * ext_err) ** 2
                            weights *= ext_w

                        if ext_pweight:
                            for _pweight in ext_pweight:
                                if _pweight[0] != p.name or _pweight[1] != r.name:
                                    continue
                                if isinstance(_pweight[3], HistogramBase):
                                    _corr, _err = weight_from_hist(
                                        event,
                                        _pweight[3].observable,
                                        _pweight[3],
                                    )
                                    sumW2 *= _corr**2
                                    sumW2 += (weights * _err) ** 2
                                    weights *= _corr
                                else:
                                    weights *= _pweight[3]

                        w_gen_set = zip_longest(
                            weight_generators, weight_generators_filters
                        )
                        for w_gen, w_filter in w_gen_set:
                            # print(f"{w_filter=}")
                            if w_gen is None:
                                continue
                            if w_filter:
                                filter_name = w_filter[0]
                                filter_args = w_filter[1:]
                                wf = getattr(MatchHelper, filter_name)(*filter_args)
                                if "region" in filter_name:
                                    # print(f"wf = {wf(r)}")
                                    if not wf(r):
                                        continue
                                if "process" in filter_name:
                                    if not wf(p):
                                        continue
                            _gen_w, _gen_w_err = w_gen(event)
                            sumW2 *= _gen_w**2
                            sumW2 += (weights * _gen_w_err) ** 2
                            weights *= _gen_w

                        # check if all event weights are zero
                        # this could cause by multiply 0 (SF or correction).
                        if not is_none_zero(weights[mask]):
                            logger.debug("all of the weights are zero!")
                            continue

                        r.event_count += non_zero_count
                        r.effective_event_count += np.sum(weights[mask])
                        if self.err_prop:
                            r.sumW2 += np.sum(sumW2[mask])
                        else:
                            r.sumW2 += np.sum((weights**2)[mask])

                        histogram_loop(r.histograms, mask, event, weights, sumW2)

                    if self.disable_pbar:
                        fstatus = ", ".join(
                            [
                                f"{report.tree_entry_stop / ttree.num_entries*100.0:.2f}%",
                                f"{ttree.num_entries} evts",
                                f"{self.fill_file_status} files",
                                f"dt={perf_counter()-t_start:.2f}s/file",
                            ]
                        )
                        logger.info(f"{p.name} processed {fstatus}")
                        t_start = perf_counter()
                    else:
                        pbar_events.update(nevent)

                    # del event
        return p.copy() if copy else p

    def process(self, config, pFilter=None, *args, **kwargs):
        """
        Process instance of ConfigMgr and fill events from TTree

        Args:
            config : instance of ConfigMgr.

            ext_rweight :
                external region weight with format of
                (ext_config, [list of observable names]:, region_name_map)
                assuming ext_config use the same list of regions and observables.
        """
        psets_pbar = tqdm(
            config.process_sets,
            unit="process set",
            dynamic_ncols=True,
            disable=self.disable_pbar,
        )
        # make a generator for processes within process sets
        processes = (p for pset in psets_pbar for p in pset)
        # setup filter
        if pFilter and not isinstance(pFilter, list):
            pFilter = [pFilter]
        else:
            pFilter = set()
        for p in processes:
            if p.name in pFilter:
                continue
            if self.skip_dummy(p):  # skip dummy if specified in skip_dummy_processes
                continue

            if self.acc_cutbook_sum_weights:
                self._cutbook_sum_weights_dict = sum_weights_from_files(p.filename)

            psets_pbar.set_description(f"On process set: {p.name}")
            file_pbar = tqdm(
                p.filename,
                leave=False,
                unit="file",
                dynamic_ncols=True,
                disable=self.disable_pbar,
            )
            file_pbar.set_description(f"processing {p.name} files")
            for i, fname in enumerate(file_pbar):
                self.fill_file_status = f"{i+1}/{file_pbar.total or 'NA'}"
                if kwargs.get("copy", None):
                    kwargs["copy"] = False
                self.plevel_process(p, fname, *args, **kwargs)
        psets_pbar.close()


# ==============================================================================
def _filter_missing_ttree(process, *args, branch_rename=None, **kwargs):
    """
    check TTree in TFile for a single process
    """
    t_start = perf_counter()
    nfiles = len(process.filename)
    logger.info(f"filtering empty ttree for {process.name} in {nfiles} files")
    histmaker = HistMaker(*args, **kwargs)
    histmaker.RAISE_TREENAME_ERROR = True
    histmaker.branch_rename = branch_rename
    # open_root_file = histmaker.open_file
    open_root_file = uproot.open
    check_ttree = histmaker.process_ttree_resolve
    filtered_files = []
    append = filtered_files.append
    for f in process.filename:
        with open_root_file(f) as opened_f:
            try:
                ttree, *_ = check_ttree(opened_f, process)
            except RuntimeError:
                continue
            if ttree is None or ttree.num_entries == 0:
                continue
        append(f)
    p_name = (process.name, process.systematic)
    logger.info(
        f"{p_name} has {len(filtered_files)} good files [{perf_counter()-t_start}s]"
    )
    return filtered_files


def filter_missing_ttree(config, use_mp=True):
    """
    check TTree in TFile for an instance of ConfigMgr
    """
    missing_list = []
    flat_proc_list = (x for pset in config.process_sets for x in pset)
    if config.good_file_list:
        with open(config.good_file_list) as f:
            good_file_list = json.load(f)
        for process in flat_proc_list:
            key = str((process.name, process.systematic))
            if key in good_file_list:
                process.filename = good_file_list[key]
            else:
                missing_list.append(process)
        if missing_list:
            flat_proc_list = missing_list
        else:
            return
    filter_kwargs = {"branch_rename": config.branch_rename}
    if config.xsec_sumw:
        filter_kwargs["xsec_sumw"] = config.xsec_sumw
    _filter = partial(_filter_missing_ttree, **filter_kwargs)
    if use_mp:
        if missing_list:
            proc_list = (x for x in missing_list)
        else:
            proc_list = (x for pset in config.process_sets for x in pset)
        n_workers = int(np.ceil(0.5 * mp.cpu_count()))
        with ProcessPoolExecutor(n_workers) as pool:
            future_map = pool.map(_filter, flat_proc_list)
            for filtered_files in future_map:
                next(proc_list).filename = filtered_files
    else:
        for process in flat_proc_list:
            process.filename = _filter(process)


# ==============================================================================
def fill_dummy_processes(config, process_list, output):
    no_dummy = True
    for name in process_list:
        pset = config.get(name)
        for proc in pset.systematics:
            if proc.systematic.source != "dummy":
                continue
            no_dummy = False
            logger.info(f"filling dummy from nominal for {proc.name, proc.systematic}")
            for region in proc.regions:
                nom_r = pset.nominal.get(region.name)
                for hist in region.histograms:
                    nom_h = nom_r.get(hist.name)
                    hist.bin_content[:] = nom_h.bin_content[:]
    if no_dummy:
        logger.info("No dummy process was found.")
    else:
        config.save(output)
