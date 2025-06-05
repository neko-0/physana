import uproot  # need version uproot4
import numpy as np
import logging
import numbers
import multiprocessing as mp
from time import perf_counter
from tqdm import tqdm
from copy import deepcopy
from fnmatch import fnmatch
from numexpr import evaluate as ne_evaluate
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from collections import defaultdict
from typing import Optional, Callable, List, Dict, Any

from .algorithm import BaseAlgorithm
from ..histo import Histogram, Histogram2D
from ..histo.jitfunc import USE_JIT
from ..histo.jitfunc import apply_phsp_correction
from ..histo.jitfunc import is_none_zero as jit_is_none_zero
from ..histo.jitfunc import parallel_nonzero_count as jit_parallel_nonzero_count
from ..tools.xsec import PMGXsec
from ..tools.sum_weights import SumWeightTool

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _apply_phsp(weights, sumW2, phsp, phsp_err):
    if USE_JIT:
        apply_phsp_correction(weights, sumW2, phsp, phsp_err)
    else:
        sumW2 *= phsp**2
        sumW2 += (weights * phsp_err) ** 2
        weights *= phsp


def is_none_zero(x):
    if USE_JIT:
        return jit_is_none_zero(x)
    else:
        return np.any(x)


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
            Can be ROOT or physana.Histogram.
            numpy array will also work. Basically it will find the weight given
            by the histogram for each event.
    """
    if isinstance(hist, Histogram):
        if not isinstance(obs, str):
            raise TypeError("1D Histogram obs need to be str.")
        return hist.find_content_and_error(ne_evaluate(obs, events))
    elif isinstance(hist, Histogram2D):
        m_events_x = ne_evaluate(obs[0], events)
        m_events_y = ne_evaluate(obs[1], events)
        return hist.find_content_and_error(m_events_x, m_events_y)
    else:
        raise TypeError("weight_hist dimension is not implemented")


def weight_from_region(events, rname, process, weight_hist):
    _w_r = process.get(rname)
    if isinstance(weight_hist, tuple):
        _obs = weight_hist[0]
        _w_hist = _w_r.get(weight_hist[1])
        return weight_from_hist(events, _obs, _w_hist)
    elif isinstance(weight_hist, dict):
        for key in weight_hist:
            if not fnmatch(rname, key):
                continue
            m_weight_hist = weight_hist[key]
            if not isinstance(m_weight_hist, list):
                # sinlge weighting hist case
                _obs = m_weight_hist[0]
                _w_hist = _w_r.get(m_weight_hist[1])
                return weight_from_hist(events, _obs, _w_hist)
            cml_w = None  # cumulative
            cml_err = None
            for w_hist_set in m_weight_hist:
                _obs = w_hist_set[0]
                _w_hist = _w_r.get(w_hist_set[1])
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


class HistMaker(BaseAlgorithm):
    """
    HistMaker class:
        handles histograms creation and preparation, with given Process and Region.

    """

    def __init__(
        self,
        *,
        nthread: Optional[int] = None,
        use_mmap: bool = True,
        disable_child_thread: bool = True,
        xsec_sumw: Optional[PMGXsec] = None,
    ):
        # processing status variables
        self._entry_start: Optional[int] = None
        self._entry_stop: Optional[int] = None
        self._tree_ibatch: int = 0
        self._is_init: bool = False
        self._is_close: bool = False
        self.step_size: str = "5MB"
        self.disable_pbar: bool = False
        self.fill_file_status: Optional[Dict[str, Any]] = None
        self.use_mmap: bool = use_mmap
        self.report: bool = False

        # tracking weights defined in regions
        self._region_weight_tracker: Dict[str, Dict[str, Any]] = {}

        # default weight defined at the HistMaker
        self.default_weight: Optional[float] = None
        self.enforce_default_weight: bool = False

        # old flag for ttree lookup error handling
        self.RAISE_TREENAME_ERROR: bool = True

        # flag for sumW2 error propagation
        self.err_prop: bool = True

        # branches reserved globally
        self.reserved_branches: Optional[List[str]] = None

        # for branch name renaming in the ttree
        self.branch_rename: Optional[Dict[str, str]] = None

        self.xsec_sumw: Optional[PMGXsec] = (
            xsec_sumw  # cross section and sum of event weights
        )
        self.skip_dummy_processes: Optional[List[str]] = None

        # variables for multi-thread/process histogram filling.
        self.hist_fill_type: Optional[str] = None
        self.hist_fill_executor: Optional[ThreadPoolExecutor] = None

        # phase space correction
        self.corrections: Optional[Dict[str, Any]] = None
        self.phasespace_corr_obs: List[str] = ["nJet30"]
        self.phasespace_apply_nominal: bool = True
        self.phsp_fallback: bool = True

        # file to cutbook sum weights
        self.sum_weights_file: Optional[str] = None
        self.sum_weights_tool: Optional[SumWeightTool] = None

        # PMG xsec tool
        self.xsec_file: Optional[str] = None
        self.xsec_tool: Callable[[str], float] = lambda x: 1.0

        # systematics name handling
        self.enable_systematics: bool = True
        self.systematics_tag: str = "_SYS_"
        self._current_syst_tag: Optional[str] = None

        # selection tracking variables
        self._selection_tracker: Dict[str, Dict[str, Any]] = defaultdict(dict)

        # file processing variables and threads
        self.use_threads: bool = True
        self.file_nthreads: Optional[int] = None
        self.f_decompression_executor: Optional[ThreadPoolExecutor] = None
        self.f_interpretation_executor: Optional[ThreadPoolExecutor] = None
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

    def prepare(self, config):
        # config will check if prepared
        config.prepare()

    def meta_data_from_config(self, config):
        """
        load meta data from configMgr object

        useful for commonly used feature. if you want to pass objects in config
        to histmaker, you can add to this function. this can help reduce the chance
        that if you have new feature to histmaker and you forget to propagate it into
        other methods.
        """
        self.corrections = config.corrections
        self.default_weight = config.default_weight
        self.reserved_branches = config.reserved_branches
        self.phasespace_corr_obs = config.phasespace_corr_obs
        self.phasespace_apply_nominal = config.phasespace_apply_nominal
        self.enforce_default_weight = config.enforce_default_weight
        self.RAISE_TREENAME_ERROR = config.RAISE_TREENAME_ERROR
        self.branch_rename = config.branch_rename or {}
        self.disable_pbar = config.disable_pbar
        self.xsec_sumw = config.xsec_sumw
        self.skip_dummy_processes = config.skip_dummy_processes
        self.sum_weights_file = config.sum_weights_file
        self.xsec_file = config.xsec_file

    def copy(self):
        return deepcopy(self)

    def open_file(self, file_name, use_mmap=False, *args, **kwargs):
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
            kwargs.setdefault("handler", uproot.MemmapSource)
            kwargs.update(opts)
        else:
            kwargs.setdefault("handler", uproot.MultithreadedFileSource)
            # kwargs.setdefault("num_workers", 2)
            # kwargs.setdefault("executor", self.f_decompression_executor)
            kwargs.update(opts)
        if not self.use_threads:
            kwargs.setdefault("use_threads", self.use_threads)
        return uproot.open(file_name, *args, **kwargs)

    def raw_open(self, *args, **kwargs):
        return uproot.open(*args, **kwargs)

    def set_entry_range(self, entry_start, entry_stop):
        self._entry_start = entry_start
        self._entry_stop = entry_stop

    def skip_dummy(self, p):
        if p.systematics is None:
            return False
        is_dummy = p.systematics.is_dummy
        if not is_dummy:
            return False
        if self.skip_dummy_processes and p.name in self.skip_dummy_processes:
            logger.info(f"skip dummy for {(p.name, p.systematic)}")
            return True
        return False

    def replace_syst_tag(self, value):
        if self.enable_systematics and value:
            return value.replace(self.systematics_tag, self._current_syst_tag)
        return value

    def process_weights_parser(self, process):
        """Return the process-level weights as a string."""
        if process.is_data:
            return None

        process_weights = process.weights
        if not process_weights:
            return process_weights

        if isinstance(process_weights, list):
            process_weights = "*".join(process_weights)

        # handle systematics naming
        process_weights = self.replace_syst_tag(process_weights)

        return process_weights

    def region_weights_parser(self, region, process_weights=None):
        # skip data process weights
        if region.parent.is_data:
            return None

        if region.name in self._region_weight_tracker:
            return self._region_weight_tracker[region.name]

        region_weights = region.weights
        if isinstance(region_weights, list):
            region_weights = "*".join(region_weights)

        if self.enforce_default_weight and self.default_weight:
            if region_weights is None and process_weights is None:
                region_weights = self.default_weight
            else:
                region_weights = f"*{self.default_weight}"

        # handle systematics naming
        region_weights = self.replace_syst_tag(region_weights)

        if process_weights:
            region_weights = f"{region_weights}*{process_weights}"

        self._region_weight_tracker[region.name] = region_weights

        return region_weights

    def phase_space_correction(
        self,
        event,
        process,
        correction_type,
        correction_variable=None,
        systematic=None,
    ):
        if not self.corrections:
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
            h = self.corrections[lookup]
            if h is None:
                logger.debug(f"Cannot find {lookup} for phase space correction")
                continue
            else:
                logger.debug(f"Found {lookup} for phase space correction")
                logger.debug(f"Found correction type {type(h)}")
            if isinstance(h, (Histogram, Histogram2D)):
                if isinstance(corr_obs, str):
                    corr_obs = self.replace_syst_tag(corr_obs)
                else:
                    corr_obs = (self.replace_syst_tag(x) for x in corr_obs)
                m_weight, m_error = weight_from_hist(event, corr_obs, h)
            elif isinstance(h, numbers.Number):
                m_weight = h
                m_error = 0.0
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
                weights_str = self.replace_syst_tag(hist.weights)
                hist_w = ne_evaluate(weights_str, event)[mask]
            else:
                obs = hist.observable
                hist_w = weights[mask]

            obs = (self.replace_syst_tag(x) for x in hist.observable)

            fdata = histogram_eval(event, mask, *obs)
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

    def set_systematics_tag(self, process):
        if not self.enable_systematics:
            return

        syst = process.systematics
        if syst:
            self._current_syst_tag = f"_{syst.tag}"
        else:
            self._current_syst_tag = "_NOSYS"

    def resolve_ttree(self, tfile, process):
        treename = process.treename

        if treename in tfile:
            return tfile[treename]

        if self.RAISE_TREENAME_ERROR:
            raise RuntimeError(f"{process.name} has no tree {treename}")

        logger.critical(
            f"Tree lookup failed for process {process.name}. "
            f"Skipping base tree {treename} in {tfile.file_path}"
        )

        return None

    def prepare_selection(self, process):
        # all_mask is a mask with only process level selection
        # if no process level seletion, accept all events.
        # the mask is array of True/False.
        syst = process.systematics
        if syst:
            process_lookup = (process.name, syst.full_name)
        else:
            process_lookup = (process.name, None)

        proc_sel = self.replace_syst_tag(process.selection)

        self._selection_tracker[process_lookup] = {"selection": proc_sel}

        for region in process:
            # setting process level and region level selection
            # this is basically cuts in ROOT TTree but in numpy format
            p_r_selection = []
            if proc_sel:
                p_r_selection.append(f"({proc_sel})")
            if region.selection:
                p_r_selection.append(region.selection)

            # combinding process and region level selections
            if p_r_selection:
                selection_str = "&".join(p_r_selection)
                selection_str = selection_str.replace("()", "")
                selection_str = selection_str.strip().strip("&")
            else:
                selection_str = ""

            selection_str = self.replace_syst_tag(selection_str)

            self._selection_tracker[process_lookup][region.name] = selection_str

            # just storing the region and process level
            # selection to the region object
            if region.full_selection is None:
                region.full_selection = selection_str
                logger.debug(
                    f"Set region full selection {region.name}: {selection_str}"
                )

    def get_selection(self, process, region=None):
        syst = process.systematics
        if syst:
            process_lookup = (process.name, syst.full_name)
        else:
            process_lookup = (process.name, None)

        if region is None:
            return self._selection_tracker[process_lookup]["selection"]
        else:
            return self._selection_tracker[process_lookup][region.name]

    def select_branches(self, process):
        # Use branches from process and its regions.
        branch_filter = process.ntuple_branches
        for r in process:
            branch_filter |= r.ntuple_branches

        # if branch renaming is requested, we need to make sure the original
        # names are used for branch filtering.
        if self.branch_rename:
            branch_filter = {
                old_b if new_b in branch_filter else new_b
                for old_b, new_b in self.branch_rename.items()
            }

        if self.sum_weight_tool:
            branch_filter |= {
                self.sum_weight_tool.dsid_branch,
                self.sum_weight_tool.run_number_branch,
            } - {None, ""}

        if self.enable_systematics:
            _update_tag = self.replace_syst_tag
            branch_filter = {_update_tag(x) for x in branch_filter}

            self.branch_rename = {
                _update_tag(x): _update_tag(y) for x, y in self.branch_rename.items()
            }

        return branch_filter or self.reserved_branches

    def filling_process_from_file(self, process, *args, **kwargs):
        if process.combine_tree:
            combined_process = process.copy()
            combined_process.clear_content()
            for tree_config in process.combine_tree:
                if isinstance(tree_config, tuple):
                    combined_process.treename = tree_config[0]
                    combined_process.selection = tree_config[1]
                elif isinstance(tree_config, dict):
                    for key, value in tree_config.items():
                        setattr(combined_process, key, value)
                else:
                    combined_process.treename = tree_config
                    combined_process.selection = None
                self._filling_process_from_file(combined_process, *args, **kwargs)
            process.add(combined_process)
        return self._filling_process_from_file(process, *args, **kwargs)

    def _filling_process_from_file(
        self,
        p,
        file_name,
        *,
        reserved_branches=None,
        copy=False,
        histogram_method=None,
    ):
        phsp_lookup = self.phase_space_correction
        phsp_fallback = self.phsp_fallback and not self.phasespace_apply_nominal
        process_weights_parser = self.process_weights_parser
        region_weights_parser = self.region_weights_parser

        open_file = self.open_file
        # open_file = uproot.open

        t_start = perf_counter()

        with open_file(file_name) as tfile:
            ttree = self.resolve_ttree(tfile, p)

            # check ttree and number of entry. return early if it's zero
            if ttree is None or ttree.num_entries == 0:
                return p.copy() if copy else p

            if self._entry_start is not None and self._entry_stop is not None:
                total_entries = self._entry_stop - self._entry_start
            else:
                total_entries = ttree.num_entries

            # setting the current systematics tag
            self.set_systematics_tag(p)

            # prepare selection for all regions inside the process.
            self.prepare_selection(p)

            # clear existing region weights tracker
            self._region_weight_tracker = {}

            # setup sum weight tool state from process
            self.sum_weight_tool.load_state_from_process(p)

            # setup PMG xsec tool
            if p.is_data:
                xsec_tool = self.xsec_tool  # default return 1
            elif self.xsec_file:
                logger.info(f"Initializing PMG xsec tool with {self.xsec_file}")
                self.xsec_tool = PMGXsec(self.xsec_file)
                # hard coded the dsid for now
                xsec_tool = lambda event: self.xsec_tool(event["mcChannelNumber"])

            # check if external histogram looping is provided
            if histogram_method:
                histogram_loop = histogram_method
            elif self.hist_fill_type == "thread":
                histogram_loop = self._hist_loop_threadpool
            elif self.hist_fill_type == "process":
                histogram_loop = self._hist_loop_processpool
            else:
                histogram_loop = self._histogram_loop

            # parsing process level weights
            p_weights = process_weights_parser(p)

            # check for filtered branches
            branch_filter = reserved_branches or self.select_branches(p)

            logger.debug(f"Number of branches reserved: {len(branch_filter)}")
            logger.debug(f"Branches reserved: {branch_filter}")

            logger.debug(f"retrieved {ttree.name}")
            logger.debug(f"opened {file_name}")
            has_corr = "with correction" if self.corrections else ""

            with tqdm(
                desc=f"Processing {p.name}|{p.systematics or 'nominal'} {has_corr}",
                total=total_entries,
                leave=False,
                unit="events",
                dynamic_ncols=True,
                disable=self.disable_pbar,
            ) as pbar_events:
                for event, report in ttree.iterate(
                    step_size=self.step_size,
                    filter_name=branch_filter,
                    report=True,
                    entry_start=self._entry_start,
                    entry_stop=self._entry_stop,
                    library="np",
                ):
                    self._tree_ibatch += 1
                    nevent = report.tree_entry_stop - report.tree_entry_start
                    pbar_events.set_description(f"Processing {nevent} events")

                    # renaming the branches
                    if self.branch_rename:
                        for old_bname, new_bname in self.branch_rename.items():
                            if old_bname in event:
                                event[new_bname] = event.pop(old_bname)

                    # all_mask is a mask with only process level selection
                    # if no process level seletion, accept all events.
                    # the mask is array of True/False.
                    p_sel = self.get_selection(p)
                    if p_sel:
                        all_mask = ne_evaluate(p_sel, event)
                        if not is_none_zero(all_mask):
                            logger.debug("No event after process selection")
                            pbar_events.update(nevent)
                            continue
                    else:
                        all_mask = np.full(nevent, True)

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

                        selection_str = self.get_selection(p, r)
                        # if no selection string is found, assume accepting all values
                        if selection_str:
                            mask = ne_evaluate(selection_str, event)
                        else:
                            logger.debug(f"empty seleciton on region {r.name}.")
                            mask = all_mask

                        # go to next iteration if no events passed both
                        # process and region level selection
                        non_zero_count = jit_parallel_nonzero_count(mask)
                        if non_zero_count == 0:
                            logger.debug("no event after region selection.")
                            continue

                        # setting the weights to the size of nevent
                        # and default the value to one
                        weights = np.ones(nevent)
                        sumW2 = np.zeros(nevent)

                        # MC only weights
                        if not p.is_data:
                            # multiply MC xsec and divide by sum weights
                            weights *= xsec_tool(event) / self.sum_weight_tool(event)

                        # combine region and process weights
                        r_weights = region_weights_parser(r, p_weights)
                        if r_weights:
                            weights *= ne_evaluate(r_weights, event)

                        # compute w2 for each event
                        sumW2 += weights**2

                        # get the pure event count without any region selection.
                        # bascially this is the number of events after process selection
                        r.total_nevents += np.sum(weights[all_mask])

                        # Apply a phase-space correction factor to a given process
                        phsp, phsp_err = phsp_lookup(
                            event, p.name, r.correction_type, systematic=None
                        )
                        if phsp is None:
                            # if the above is not found, try lookup based on treename
                            # temperoraly use for iterative corrction
                            # might need a better approach
                            phsp, phsp_err = phsp_lookup(
                                event, p.treename, r.correction_type, systematic=None
                            )
                        if phsp is None and phsp_fallback:
                            phsp, phsp_err = phsp_lookup(
                                event, p.name, r.correction_type
                            )
                        if phsp is not None:
                            _apply_phsp(weights, sumW2, phsp, phsp_err)

                        # check if all event weights are zero
                        # this could cause by multiply 0 (SF or correction).
                        if not is_none_zero(weights[mask]):
                            logger.debug("all of the weights are zero!")
                            continue

                        r.filled_nevents += non_zero_count
                        r.effective_nevents += np.sum(weights[mask])
                        if self.err_prop:
                            r.sumW2 += np.sum(sumW2[mask])
                        else:
                            r.sumW2 += np.sum((weights**2)[mask])

                        histogram_loop(r.histograms, mask, event, weights, sumW2)

                    if self.disable_pbar:
                        if self._entry_start is not None:
                            current_nevent = (
                                self._entry_start
                                + report.tree_entry_stop
                                - report.tree_entry_start
                            )
                        else:
                            current_nevent = report.tree_entry_stop
                        fstatus = ", ".join(
                            [
                                f"{current_nevent / total_entries*100.0:.2f}%",
                                f"total events {total_entries}",
                                f"dt={perf_counter()-t_start:.2f}s/file",
                                self.fill_file_status or "",
                            ]
                        ).strip(", ")
                        logger.info(f"{p.name} processed {fstatus}")
                        if self.report:
                            msg = (
                                f"{r.name}, {r.total_nevents}, {r.filled_nevents}, {r.effective_nevents}"
                                for r in p.regions
                            )
                            logger.info(f"{p.name} event loop info:\n" + "\n".join(msg))
                        t_start = perf_counter()
                    else:
                        pbar_events.update(nevent)

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

            self.sum_weight_tool = SumWeightTool(self.sum_weights_file)

            psets_pbar.set_description(f"On process set: {p.name}")
            file_pbar = tqdm(
                p.input_files,
                leave=False,
                unit="file",
                dynamic_ncols=True,
                disable=self.disable_pbar,
            )

            file_pbar.set_description(f"processing {p.name} files")

            if self.file_nthreads:
                with ThreadPoolExecutor(self.file_nthreads) as executor:
                    executor.map(
                        self.filling_process_from_file,
                        [p] * len(p.input_files),
                        p.input_files,
                        *args,
                        **kwargs,
                    )
            else:
                for i, fname in enumerate(file_pbar):
                    self.fill_file_status = f"{i+1}/{file_pbar.total or 'NA'}"
                    if kwargs.get("copy", None):
                        kwargs["copy"] = False
                    self.filling_process_from_file(p, fname, *args, **kwargs)
        psets_pbar.close()


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
