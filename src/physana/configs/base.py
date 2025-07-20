import os
import re
import sys
import json
import fnmatch
import logging
import collections
import concurrent.futures
import numbers
from time import perf_counter
from copy import deepcopy, copy
from pathlib import Path

import _jsonnet
import numpy as np
from tqdm import tqdm

from ..histo import ProcessSet, Region, Histogram, Histogram2D
from ..histo.tools import from_root, from_numexpr, get_expression_variables
from ..systematics import Systematics
from ..serialization import Serialization
from ..serialization.base import async_from_pickles
from .dispatch_tools import split
from .correction import CorrectionContainer
from .xsecsumw import XSecSumEvtW

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class ConfigMgr:
    def __init__(
        self,
        name="",
        ntuple_src_path="./",
        output_path="./",
        description="",
    ):
        self.name = name
        self.filename = None
        self.ntuple_src_path = ntuple_src_path
        self.output_path = output_path
        self.descriptions = description
        self.ofilename = "output.pkl"
        self.tag = ""

        # default input ntuple files if process level is not specified.
        self.input_files = set()
        self.file_record = set()

        # status flags of a config instance.
        self.prepared = False
        self.branches_reserved = False
        self.filled = False

        # flags for histmaker
        self.USE_MP = True
        self.disable_pbar = False
        self.RAISE_TREENAME_ERROR = True
        self.branch_rename = None  # dict for renaming branch in the ntuple

        # Cross section file
        self.xsec_file = None

        # setting for sum of weight tools
        self.sum_weights_file = None

        # default weighting
        self._default_wlist = []
        self.default_weight = f"({'*'.join(self._default_wlist)})"
        self.enforce_default_weight = False

        # phase-space correction, pass through HistMaker.meta_data_from_config
        self.corrections = CorrectionContainer()
        self.phasespace_corr_obs = ["nJet30"]
        self.phasespace_apply_nominal = True

        self.meta_data = {}

        self.systematics = collections.defaultdict(list)
        self._registered_systematics = collections.defaultdict(set)

        # for keeping track of reserved branches from prepration
        self.reserved_branches = set()
        self._process_branch_dict = collections.defaultdict(set)
        self._systematic_branch_dict = collections.defaultdict(set)
        self._region_branch_dict = collections.defaultdict(set)
        self._histogram_branch_dict = collections.defaultdict(set)

        self.process_sets = []
        self.regions = []
        self.aux_regions = []
        self.histograms = []
        self.histograms2D = []

        self._abcd_region_selection = {}

        self._add_history = []

        # variables use for filtering/skimming
        self.process_region_type_filter = {}
        self.region_name_filter = set()
        self.region_type_filter = set()
        self._filtered_regions = {}

        self._xsec_sumw = None
        self.good_file_list = None

        self.skip_dummy_processes = None

        """
        Members require no deepcopy
        NOTE: those members:
            self.histograms, self.histogram2D, self.regions, self.aux_regions,
            self.reserved_branches, self._abcd_region_selection, etc
        were created for temperorary use, and users do not useullay make changes
        to those. Hence, copy is not necessary here.
        """
        self._deepcopy_exclusions = {
            "_deepcopy_exclusions",
            "corrections",
            "histograms",
            "histogram2D",
            "regiosn",
            "aux_regions",
            "reserved_branches",
            "skip_dummy_processes",
            "_abcd_region_selection",
            "_filtered_regions",
            "_add_history",
            "_xsec_sumw",
        }

    def __deepcopy__(self, memo):
        cls = self.__class__
        copy_self = cls.__new__(cls)
        memo[id(self)] = copy_self
        keys = self.__dict__.keys()
        skip_deepcopy_keys = keys & self._deepcopy_exclusions  # intersection
        deepcopy_keys = keys - skip_deepcopy_keys  # complementary
        for key in skip_deepcopy_keys:
            copy_self.__dict__[key] = self.__dict__[key]
        for key in deepcopy_keys:
            copy_self.__dict__[key] = deepcopy(self.__dict__[key], memo)
        return copy_self

    def __str__(self):
        return f"{self.name}:{self.filename}"

    @property
    def region_list(self):
        return [r.name for r in self.regions + self.aux_regions]

    @property
    def observable_list(self):
        return [h.name for h in self.histograms + self.histograms2D]

    @property
    def _processes_dict(self):
        return {p.name: p for p in self.processes}

    @property
    def _process_sets_dict(self):
        return {p_set.name: p_set for p_set in self.process_sets}

    @property
    def processes(self):
        """
        Interface for getting all nominal processes.
        """
        return [x.nominal for x in self.process_sets if x.nominal is not None]

    @property
    def default_wlist(self):
        return self._default_wlist

    @default_wlist.setter
    def default_wlist(self, wlist):
        self._default_wlist = wlist
        self.default_weight = "*".join(self._default_wlist)

    @property
    def xsec_sumw(self):
        if self._xsec_sumw:
            return XSecSumEvtW.open(self._xsec_sumw)
        else:
            return None

    @xsec_sumw.setter
    def xsec_sumw(self, xsecf):
        self._xsec_sumw = xsecf

    def __sizeof__(self):
        size = 0
        for key, item in self.__dict__.items():
            size += sys.getsizeof(item)
        for pset in self.process_sets:
            size += sys.getsizeof(pset)
        if self.regions:
            size += len(self.regions) * sys.getsizeof(self.regions[0])
        if self.aux_regions:
            size += len(self.aux_regions) * sys.getsizeof(self.aux_regions[0])
        if self.histograms:
            size += len(self.histograms) * sys.getsizeof(self.histograms[0])
        if self.histograms2D:
            size += len(self.histograms2D) * sys.getsizeof(self.histograms2D[0])
        return size

    def __getitem__(self, index):
        return self.processes[index]

    def __setitem__(self, index, value):
        self.processes[index] = value

    def __add__(self, rhs):
        c_self = self.copy()
        c_self.add(rhs)
        return c_self

    def add(self, rhs):
        if not isinstance(rhs, type(self)):
            raise TypeError(f"Invalid type {type(rhs)}")
        for name, rhs_pset in rhs.pset_items():
            self.append_process(rhs_pset, mode="merge")
        for rhs_region in rhs.regions:
            if self.has_region(rhs_region.name):
                continue
            self.append_region(rhs_region.copy())
        for rhs_aux_region in rhs.aux_regions:
            if self.has_region(rhs_aux_region.name):
                continue
            self.append_region(rhs_aux_region.copy(), aux=True)
        self._filtered_regions.update(rhs._filtered_regions)
        self.reserved_branches |= rhs.reserved_branches

    def info(self):
        basic_info = (
            self.src_path,
            self.output_path,
            self.descriptions,
            self.histogram_backend,
            self.output_processed_tag,
            self.addition_tag,
            self.input_files,
            self.ofilename,
            self.prepared,
            self.USE_MP,
        )
        return basic_info

    def config_metadata(self):
        meta = {}
        meta["process"] = self.list_processes()
        meta["path"] = (self.src_path, self.output_path)
        meta['info'] = self.info()
        return meta

    def set_input_files(self, input_files):
        if isinstance(input_files, str):
            self.input_files = {f"{self.src_path}/{input_files}"}
        elif isinstance(input_files, list):
            self.input_files = set([f"{self.src_path}/{file}" for file in input_files])
        else:
            raise ValueError(
                f"Files setting receives invalid type {type(input_files)}."
            )
        self.file_record = copy(self.input_files)

    def update_input_files_from_record(self):
        for pset in self.process_sets:
            for p in pset:
                p.input_files |= p.file_record
        self.input_files |= self.file_record

    def self_split(self, split_type, *args, **kwargs):
        if split_type == "region":
            logger.warning("splitting by region might clear all filled contents!")
        yield from split(self, split_type, *args, **kwargs)

    def copy(self, *, shallow=False):
        return copy(self) if shallow else deepcopy(self)

    @classmethod
    def copy_from(cls, config, skip=[], *, shallow=False):
        if not skip:
            return config.copy(shallow=shallow)
        new_config = cls()
        keys = config.__dict__.keys()
        shallow_keys = {}
        if shallow:
            shallow_keys = keys & set(skip)
            for s_key in shallow_keys:
                new_config.__dict__[s_key] = copy(config.__dict__[s_key])
        deep_keys = keys - shallow_keys
        for d_key in deep_keys:
            new_config.__dict__[d_key] = deepcopy(config.__dict__[d_key])
        return new_config

    def reserve_branches(self, expr=None, parser=from_root):
        if isinstance(expr, list):
            parsed = set(expr)
        elif expr:
            parsed = get_expression_variables(expr, parser=parser)
        else:
            parsed = None
        if parsed:
            self.reserved_branches |= parsed
        return parsed

    def _reserve_region_branches(self, use_mp=True, *, executor=None):
        """
        Loop through all regions and reserve weights and observable branches
        in the ROOT TTree.
        """

        t_now = perf_counter()

        if self.branches_reserved:
            return None

        if use_mp:
            logger.info("Reserving branches with multiprocessing.")

            if executor is None:
                executor = concurrent.futures.ProcessPoolExecutor()

            weight = {}
            selection = {}
            for r in self.regions + self.aux_regions:
                if r.branch_reserved:
                    continue
                if r.weights:
                    weight[r.name] = executor.submit(
                        get_expression_variables, r.weights
                    )
                if r.selection:
                    selection[r.name] = executor.submit(
                        get_expression_variables, r.selection
                    )
            # branches = weight + selection
            # num_regions = len(branches)
            num_regions = len(weight) + len(selection)
            with tqdm(
                total=num_regions, leave=False, disable=self.disable_pbar
            ) as pbar_regions:
                pbar_regions.set_description("branch reservation")
                # for b in concurrent.futures.as_completed(branches):
                for name in list(weight.keys()):
                    b_result = weight[name].result()
                    del weight[name]
                    self.reserved_branches |= b_result
                    self._region_branch_dict[name] |= b_result
                    pbar_regions.update()
                for name in list(selection.keys()):
                    b_result = selection[name].result()
                    del selection[name]
                    self.reserved_branches |= b_result
                    self._region_branch_dict[name] |= b_result
                    pbar_regions.update()
        else:
            num_regions = len(self.regions) + len(self.aux_regions)
            self_reserve_branches = self.reserve_branches
            with tqdm(
                total=num_regions, leave=False, disable=self.disable_pbar
            ) as pbar_regions:
                pbar_regions.set_description("branch reservation")
                for r in self.regions + self.aux_regions:
                    if r.branch_reserved:
                        continue
                    if r.weights:
                        b_result = self_reserve_branches(r.weights)
                        self._region_branch_dict[r.name] |= b_result
                    if r.selection:
                        b_result = self_reserve_branches(r.selection, from_numexpr)
                        self._region_branch_dict[r.name] |= b_result
                    pbar_regions.update()

        logger.info(f"branches reservation cost {perf_counter()-t_now:.2f}s")
        self.branches_reserved = True

    def add_branch_list(self, branch_list=None, use_mp=False):
        if not isinstance(branch_list, list):
            raise TypeError("branch_list should be type list.")
        for branch in branch_list:
            self.reserved_branches.add(branch)

    def add_process_branch(self, name, branches):
        if not isinstance(branches, set):
            raise TypeError("Branches need to be type set.")
        self._process_branch_dict[name] |= branches

    def add_process(self, name, *args, **kwargs):
        if name not in self.list_processes():
            my_p_set = ProcessSet.create_nominal(name, *args, **kwargs)
            p_branch = self._process_branch_dict[name]
            for p in my_p_set:
                if p.selection:
                    p_branch |= self.reserve_branches(p.selection, from_numexpr)
                if isinstance(p.weights, str):
                    p_branch |= self.reserve_branches(p.weights)
                elif isinstance(p.weights, numbers.Number):
                    p.weights = f"{p.weights}"
                else:
                    for w in p.weights:
                        p_branch |= self.reserve_branches(w)
            self.process_sets.append(my_p_set)

    """
    alias new_process to add_process.
    The reason is I need the ability to insert process after tree processing,
    and I need check if there existing process.
    """
    new_process = add_process

    def append_process(self, process, *, mode=None, copy=True):
        """
        Appending a process into the configuration manager file.

        Args:
            process (obj:Process) : a Process instance/object

        Returns:
            no return
        """
        if process.name not in self.list_processes():
            c_process = process.copy() if copy else process
            if isinstance(c_process, ProcessSet):
                self.process_sets.append(c_process)
            else:
                self.process_sets.append(ProcessSet.from_process(c_process))
        else:
            if mode == "merge":
                self.get_process_set(process.name).add(process, copy)

    def append_aux_process(self, process):
        c_process = process.copy()
        self.aux_processes.append(c_process)
        self._aux_processes_dict[c_process.name] = c_process

    def get_process(self, process_name):
        return self._processes_dict[process_name]

    def get_process_set(self, name):
        return self._process_sets_dict[name]

    def get(self, lookup):
        """
        a 'path' way lookup inner structure
        e.g. wjets//nominal//ttbarCR//met
        """
        lookup = lookup.split("//")
        obj = self.get_process_set(lookup[0])
        for i in lookup[1:]:
            obj = obj.get(i)
        return obj

    def append_region(self, region, reserve_branches=False, aux=False):
        """
        Apending region in to configMgr.
        """
        if reserve_branches:
            region.branch_reserved = True
            r_weight = self.reserve_branches(region.weights)
            r_select = self.reserve_branches(region.selection)
            region.ntuple_branches |= r_weight | r_select
        if aux:
            logger.debug(f"adding aux region: {region.name}, type {region.dtype}")
            self.aux_regions.append(region)
        else:
            logger.debug(f"adding region: {region.name}, type {region.dtype}")
            self.regions.append(region)

    def add_region(
        self,
        name,
        selection,
        weights=None,
        study_type="plot",
        corr_type='None',
        *,
        reserve_branches=False,
        aux=False,
        **kwargs,
    ):
        if self.has_region(name):
            logger.warning(f"Region {name} already exists.")
            return
        r = Region(name, weights, selection, study_type, corr_type, **kwargs)
        self.append_region(r, reserve_branches, aux=aux)

    def add_histogram1D(
        self,
        name,
        bins,
        xmin=None,
        xmax=None,
        xtitle="",
        observable=None,
        *args,
        **kwargs,
    ):
        if self.has_observable(name):
            return
        logger.info(f"adding observable {name}")
        if observable:
            hist_branch = self.reserve_branches(observable)
            observable = from_root(observable).to_numexpr()
        else:
            hist_branch = self.reserve_branches(name)
            observable = from_root(name).to_numexpr()
        kwargs.setdefault("dtype", "reco")
        kwargs.update({"observable": observable})
        if isinstance(bins, list):
            hist = Histogram.variable_bin(name, bins, xtitle=xtitle, *args, **kwargs)
        elif xmin is None or xmax is None:
            raise ValueError("xmin or xmax is None")
        else:
            hist = Histogram(name, bins, xmin, xmax, xtitle=xtitle, *args, **kwargs)
        self._histogram_branch_dict[hist.name] |= hist_branch
        self.histograms.append(hist)

    add_observable = add_histogram1D

    def append_histogram_1d(self, hist):
        if self.has_observable(hist.name):
            return
        name = hist.name
        observable = hist.observable[0]
        if observable:
            self._histogram_branch_dict[name] |= self.reserve_branches(observable)
        else:
            name = from_numexpr(name).to_root()
            self._histogram_branch_dict[name] |= self.reserve_branches(name)
        self.histograms.append(hist)

    def _parse_bin(self, bin_value: int, min_value: float, max_value: float) -> list:
        if isinstance(bin_value, list):
            return bin_value
        width = (max_value - min_value) / bin_value
        return list(np.arange(min_value, max_value + width, width))

    def add_histogram2D(
        self,
        name: str,
        xvar: str,
        yvar: str,
        xbin: int | list = 1,
        xmin: float = -1,
        xmax: float = 1,
        ybin: int | list = 1,
        ymin: float = -1,
        ymax: float = 1,
        *args,
        **kwargs,
    ):
        logger.info(f"Adding 2D observable {name}")
        kwargs.setdefault("dtype", "reco")

        # caching histogram branches
        self._histogram_branch_dict[name] |= self.reserve_branches(xvar)
        self._histogram_branch_dict[name] |= self.reserve_branches(yvar)

        # convert variable to numexpr format
        xvar = from_root(xvar).to_numexpr()
        yvar = from_root(yvar).to_numexpr()

        # hanlde variable binning
        if isinstance(xbin, list) and isinstance(ybin, list):
            bin_args = (xvar, yvar, xbin, ybin)
            histo_constructor = Histogram2D.variable_bin
        elif isinstance(xbin, list) or isinstance(ybin, list):
            ybin = self._parse_bin(ybin, ymin, ymax)
            xbin = self._parse_bin(xbin, xmin, xmax)
            bin_args = (xvar, yvar, xbin, ybin)
            histo_constructor = Histogram2D.variable_bin
        else:
            bin_args = (xvar, yvar, xbin, xmin, xmax, ybin, ymin, ymax)
            histo_constructor = Histogram2D
        _histo2d = histo_constructor(name, *bin_args, *args, **kwargs)

        # storing the histogram
        self.histograms2D.append(_histo2d)

    def append_histogram_2d(self, hist):
        if self.has_observable(hist.name):
            return
        name = hist.name
        xvar = from_numexpr(hist.xvar).to_root()
        yvar = from_numexpr(hist.yvar).to_root()
        self._histogram_branch_dict[name] |= self.reserve_branches(xvar)
        self._histogram_branch_dict[name] |= self.reserve_branches(yvar)
        self.histograms2D.append(hist)

    def add_manipulate_job(self, processes, job_type):
        self.manipulate_jobs.append([processes, job_type])

    def prepare(self, use_mp=False):
        """
        Method for preparing the ConfigMgr instance. It dispatch histograms and
        regions in the ConfigMgr level into each Process(Set) instance. Part of
        the branches reservation also happen here.
        """
        logger.info("finalizing configMgr.")

        # check to see if configMgr is already prepared
        if self.prepared:
            return True

        # reserve branches for default weights
        default_w_branch = self.reserve_branches(self._default_wlist)

        # filtering regions
        _findex = []
        for i, r in enumerate(self.regions):
            if r.name in self.region_name_filter:
                _findex.append(i)
                continue
            if r.dtype in self.region_type_filter:
                _findex.append(i)
                continue
        regions = []
        for i, r in enumerate(self.regions):
            if i in _findex:
                self._filtered_regions[r.name] = r
            else:
                regions.append(r)
        self.regions = regions

        if self.USE_MP is False or use_mp is False:
            use_mp = False
            self._reserve_region_branches(use_mp)
        else:
            workers = int(np.ceil(0.5 * os.cpu_count()))
            with concurrent.futures.ProcessPoolExecutor(workers) as exe:
                self._reserve_region_branches(use_mp, executor=exe)

        # distributing histograms into regions
        for r in self.regions + self.aux_regions:
            # both histograms and histogram2D are list,
            # so just combine them when looping.
            for hist in self.histograms + self.histograms2D:
                if not r.hist_type_filter.accept(hist):
                    logger.debug(f"{r.full_name} rejected {hist.name} due to type")
                    continue

                if not hist.filter.accept(r):
                    logger.debug(f"{hist.name} rejected {r.full_name} due to full name")
                    continue
                r.append(hist, enable_filter=True)
                histo_branches = self._histogram_branch_dict.get(hist.name, set())
                r.ntuple_branches |= histo_branches
                r.histo_branches |= histo_branches
            r.ntuple_branches |= self._region_branch_dict.get(r.name, set())

        # region weight checkfing dict
        found_region_weight = {}
        # distributing regions into processes
        psets = (p for pset in self.process_sets for p in pset)
        for p in psets:
            # check filename for each process
            # if the process does not have filename, the config level
            # filename info will be used.
            if not p.input_files:
                p.file_record = self.file_record
                p.input_files = self.input_files

            # store the Ntuple branches to process
            p.ntuple_branches |= self._process_branch_dict.get(p.name, set())
            p.ntuple_branches |= default_w_branch

            for r in self.regions + self.aux_regions:
                if r.dtype in self.process_region_type_filter.get(p.name, set()):
                    continue
                r.ntuple_branches |= self._region_branch_dict.get(r.name, set())
                p.append(r)
                # if not self.enforce_default_weight and not p.weights:
                if self.enforce_default_weight or p.weights:
                    continue
                if r.name not in found_region_weight:
                    found_region_weight[r.name] = True if r.weights else False
                if not found_region_weight[r.name]:
                    logger.warning("The 'enforce_default_weight' option is off!")
                    logger.warning(f"Cannot find any weights in {r.name} or {p.name}!")

        self.prepared = True

        return self.prepared

    def get_output_file(self):
        return Path(self.output_path).joinpath(f"{self.ofilename}.pkl").resolve()

    def save(self, filename="", backend="pickle", metadata=False, *args, **kwargs):
        start_t = perf_counter()

        m_serial = Serialization("config")
        logger.info(f"saving file. using backend {backend}")
        if backend == "pickle":
            serial_method = m_serial.to_pickle
            ext = ".pkl"
        elif backend == "shelve":
            serial_method = m_serial.to_shelve
            ext = ".shelve"
        elif backend == "klepto":
            serial_method = m_serial.to_klepto
            ext = ".kle"
        elif backend == "dir":
            serial_method = m_serial.to_dir
            ext = ".dir"
        else:
            logger.warning(f"cannot find backend {backend}, fall back to pickle.")
            serial_method = m_serial.to_pickle
            ext = ".pkl"

        if filename:
            self.ofilename = filename
        self.ofilename = self.ofilename.replace(f"{ext}", "")
        output_file = Path(self.output_path).joinpath(f"{self.ofilename}{ext}")
        output_file = output_file.resolve()

        # clear correction and parent reference
        self.corrections.clear_buffer()
        self.clear_children_parent()
        if metadata and backend == "pickle":
            m_metadata = self.config_metadata()
            output_data = [self, m_metadata]
            serial_method = m_serial.to_pickles
        else:
            output_data = self

        serial_method(output_data, output_file, *args, **kwargs)
        logger.info(f"file saved: {output_file}, time {perf_counter()-start_t}s")

        return output_file

    def save_to(self, filename):
        m_serial = Serialization("config")
        m_serial.to_pickle(self, filename)
        logger.info(f"file saved: {filename}")

    def load(self, filename):
        m_serial = Serialization("config")
        idata = m_serial.from_pickle(filename)
        self.__dict__ = deepcopy(idata.__dict__)

    @staticmethod
    def open(filename, backend="pickle"):
        logger.info(f"trying to open({filename})")
        _t_start = perf_counter()
        if not isinstance(filename, (str, Path)):
            return filename
        m_serial = Serialization("config")
        try:
            if backend == "pickle":
                m_config = m_serial.from_pickle(filename)
            elif backend == "shelve":
                m_config = ConfigMgr.merge(m_serial.from_shelve(filename))
            m_config.update_children_parent()
            m_config.filename = filename
            return m_config
        except Exception as _error:
            raise IOError(
                f"cannot open file: {filename}({type(filename)}) using {backend}"
            ) from _error
        finally:
            t_diff = perf_counter() - _t_start
            logger.info(f"open({filename}) takes {t_diff:.2f}s wall time.")

    @staticmethod
    def open_files(file_list):
        t_now = perf_counter()
        opened_files = list(async_from_pickles(file_list))
        t_diff = perf_counter() - t_now
        logger.info(f"opened {len(file_list)} files, {t_diff:.2f}s")
        return opened_files

    def add_meta_data(self, meta_data_tag, meta_data):
        if not (meta_data_tag in self.meta_data):
            self.meta_data[meta_data_tag] = []
        self.meta_data[meta_data_tag].append(meta_data)

    # --------------------------------------------------------------------------
    def filter_region(self, filter_type, action="keep"):
        for p in self.processes:
            for r in p:
                if action != "keep":
                    continue
                if r.dtype == filter_type:
                    continue
                p.regions.remove(r)

    # --------------------------------------------------------------------------
    def remove_process(self, pname, *, systematic=None):
        if not isinstance(pname, str):
            raise TypeError(f"require str, but receive {type(pname)}")
        for pset in self.process_sets:
            if pset.name != pname:
                continue
            removed = False
            if systematic is None:
                pset.nominal = None
                removed = True
            else:
                for p in pset:
                    if not p.systematic:
                        continue
                    if systematic != p.systematic.full_name:
                        continue
                    pset.systematics.remove(p)
                    removed = True
                    break
            if removed:
                logger.info(f"removed process -> {pset.name}({systematic})")

    def remove_process_set(self, pname):
        if not isinstance(pname, str):
            raise TypeError(f"require str, but receive {type(pname)}")
        for pset in self.process_sets:
            if pset.name != pname:
                continue
            self.process_sets.remove(pset)
            logger.info(f"removed process set -> {pset.name}")
            break

    def remove_nominal(self):
        for pset in self.process_sets:
            pset.nominal = None

    # --------------------------------------------------------------------------
    # TODO: work on clear process for ProcessSet
    def clear_process_content(self):
        for p in self.processes:
            p.clear_content()

    def clear_process(self):
        for p in self.processes:
            p.clear()

    def clear_process_set(self):
        self.process_sets = []

    # --------------------------------------------------------------------------
    def list_processes(self, *, pattern=None, backend=None, set=True):
        if set:
            _processes = [p.name for p in self.process_sets]
        else:
            _processes = [p.name for p in self.processes]
        if pattern is None:
            return _processes
        if backend == "re":
            reg_exp = re.compile(pattern)
            return list(filter(reg_exp.search, _processes))
        else:
            return fnmatch.fnmatch.filter(_processes, pattern)

    def items(self):
        """
        This default to nominal processes
        """
        return self._processes_dict.items()

    def pset_items(self):
        """
        similar to items, but it's for ProcessSet
        """
        return self._process_sets_dict.items()

    def list_regions(self, process, *args, **kwargs):
        return self.get_process(process).list_regions(*args, **kwargs)

    def print_config(self):
        logger.info("List of reserved branches:")
        p_count = 0
        r_count = 0
        h_count = 0
        map(print, self.reserved_branches)
        p_level = 1
        # r_level = 3
        h_level = 5
        ind = " " * 4
        for p in self.processes:
            print(f"{ind*p_level}Process: {p.name}")
            print(f"{ind*(p_level + 1)}ttreename: {p.treename}")
            print(f"{ind*(p_level + 1)}selection: {p.selection}")
            print(f"{ind*(p_level + 1)}number regions: {len(p.regions)}")
            p_count += 1
            for r in p.regions:
                print(f"{ind * (p_level + 1)}Region: {r.name}")
                print(f"{ind * (p_level + 1)}Selction: {r.selection}")
                print(f"{ind * (p_level + 1)}type: {r.dtype}")
                print(f"{ind * (p_level + 1)}number histograms: {len(r.histograms)}")
                print(f"{ind * (p_level + 1)}Histograms:")
                r_count += 1
                for hist in r.histograms:
                    print(f"{ind * h_level}{hist.observable} {type(hist)}")
                    h_count += 1
        print(f"total number of processes {p_count}")
        print(f"total number of regions {r_count}")
        print(f"total number of histograms {h_count}")

    def has_region(self, name):
        return name in self.region_list

    def has_observable(self, name):
        return name in self.observable_list

    def define_systematics_group(self, *args, **kwargs):
        """
        Define systematics group.
        """
        syst = Systematics(*args, **kwargs)
        if syst.full_name in self._registered_systematics[syst.name]:
            raise ValueError(f"Duplicated systematics {syst.name}/{syst.full_name}")
        self.systematics[syst.name].append(syst)
        self._registered_systematics[syst.name].add(syst.full_name)

    def set_systematics(self, process_name, sys_name, is_dummy=False):
        """
        Require defining systematic first before setting it.

        Systematic is defined through ConfigMgr.define_systematics_group. After defining,
        internal list of systematic will be created and stored, and this list will
        be used for looking up systematic to be set.
        """
        try:
            pset = self.get_process_set(process_name)
        except KeyError:
            logger.critical(
                f"Cannot find {process_name}. Maybe you forget add process?"
            )
            raise
        sys_list = [sys_name] if isinstance(sys_name, str) else sys_name
        for name in sys_list:
            for systematic in self.systematics[name]:
                syst = systematic.copy()
                syst.is_dummy = is_dummy
                pset.add_systematics(syst)

    def set_dummy_systematics(self, process_name, sys_name):
        """
        Add dummy copy of a systematic object to a process set. This is require
        for keeping track of all normalization factor due to different systematic
        """
        self.set_systematics(process_name, sys_name, is_dummy=True)

    def create_dummy_systematics(self, process_names):
        """
        create dummy systematic process with existing list of systematic.
        if systematic is already set, dummy will not be created.

        Args:
            process_names : list
                list of (nominal) process names for creating dummy systematics.
        """
        for syst_name in self.systematics:
            for name in process_names:
                # duplication will raise ValueError. just skip it
                try:
                    self.set_dummy_systematics(name, syst_name)
                except ValueError:
                    continue

    def list_systematic_type(self):
        systematic_type = []
        for pset in self.process_sets:
            systematic_type += pset.systematic_type()
        return set(systematic_type)

    def list_computed_systematics(self):
        computed = []
        for pset in self.process_sets:
            computed += pset.list_computed_systematics()
        return set(computed)

    def list_systematic_full_name(self, filter=True, process_level=True):
        """
        list all of the systematics with full name

        Args:
            filter : bool, default=True
                filter that make sure at least one of the process contain such
                systematic

            process_level : bool, default=True
                include systematics that are not register through the ConfigMgr.
        """
        systematics = [None]
        # check if the systematic in any of the process
        syst_set = []
        if process_level or filter:
            for pset in self.process_sets:
                syst_set += list(pset.list_systematic_full_name())
            syst_set = set(syst_set)
            if process_level:
                systematics += list(syst_set)
        for syslist in self.systematics.values():
            for m_sys in syslist:
                sys_full_name = m_sys.full_name
                if filter and sys_full_name not in syst_set:
                    continue
                if sys_full_name in systematics:
                    continue
                systematics.append(sys_full_name)
        return systematics

    def list_all_systematics(self):
        return self.list_systematic_full_name() + list(self.list_computed_systematics())

    def get_systematic(self, full_name):
        for pset in self.process_sets:
            if full_name in pset.list_systematic_full_name():
                return pset.get(full_name).systematic

    def generate_systematic_group(self, name: str, lookup: tuple) -> dict:
        """
        Generate systematic group with lookup keys

        Args:
            name (str): name of the systematic group that you are generating.

            lookup (tuple): tuple for looking up systematics. fnmatch will be used.
                e.g. ("JET*Eff*", "JET*NP", ""), the format will be the same as
                the systematic fullname.

        return:
            {systematic group name (str): list of systematics (list(str))}
        """
        output = []
        stored_systematics = set(self.list_systematic_full_name())
        stored_systematics |= set(self.list_computed_systematics())
        for systematic in stored_systematics:
            if systematic is None:
                continue
            if all([fnmatch.fnmatch(*x) for x in zip(systematic, lookup)]):
                output.append(systematic)
        return {name: output}

    # --------------------------------------------------------------------------
    def load_correction(self, file, *, format="pickle"):
        m_serial = Serialization("config")
        if format == "pickle":
            self.corrections = m_serial.from_pickle(file)

    def load_systematics(self, filename, create_dummy=None):
        """
        load and define systematics from external json(net) file.
        """
        if create_dummy is None:
            create_dummy = []
        if ".jsonnet" in filename:
            data = _jsonnet.evaluate_file(filename)
            data = json.loads(data)
        else:
            with open(filename) as f:
                data = json.load(f)
        # defining systematics first
        for tree_sys in data["tree_base"]:
            self.define_tree_systematics(**tree_sys)
        for weight_sys in data["weight_base"]:
            self.define_weight_systematics(**weight_sys)
        # keep track of used systematic
        used_syst = []
        # setting the systematics
        for process, syslist in data["set_systematics"].items():
            self.set_systematics(process, syslist)
            used_syst += syslist
        # cleanup systematics that aren't used
        for syst in list(self.systematics):
            if syst not in used_syst:
                logger.warning(f"{syst} is defined but never used, removing.")
                self.systematics.pop(syst)
        if create_dummy:
            for process, syslist in data["set_systematics"].items():
                for pset in create_dummy:
                    for syst in syslist:
                        # duplication will raise ValueError.
                        try:
                            self.set_dummy_systematics(pset, syst)
                        except ValueError:
                            continue
        # check for branch renaming
        if branch_rename := data.get("branch_rename", {}):
            self.branch_rename = self.branch_rename or {}
            self.branch_rename.update(branch_rename)

    # --------------------------------------------------------------------------
    def swap_processes(self, config, skip=None):
        if skip is None:
            skip = []
        for pset in config.process_sets:
            if pset.name in skip:
                continue
            self_pset = self.get(pset.name)
            for proc in pset:
                if proc.systematics is None:
                    self_nominal = self_pset.nominal
                    if self_nominal is None:
                        self_nominal = proc
                    else:
                        self_nominal.replace_data(proc)
                        self_nominal.input_files = proc.input_files
                    logger.info(f"replaced {proc.name} nominal content.")
                else:
                    syst_name = proc.systematics.full_name
                    self_p = self_pset.get(syst_name)
                    if self_p.systematics is None:
                        logger.info(f"Cannot find {syst_name}")
                        continue
                    else:
                        self_p.replace_data(proc)
                        self_p.input_files = proc.input_files
                        logger.info(f"replaced {proc.name} {syst_name} content.")

    # --------------------------------------------------------------------------
    def count_object(self, obj_type="process"):
        counter = 0
        if obj_type == "process":
            for pset in self.process_sets:
                counter += 1
                counter += len(pset.systematics)
        elif obj_type == "region":
            if self.prepared:
                for pset in self.process_sets:
                    for p in pset:
                        counter += len(p.regions)
            else:
                counter += len(self.process_sets)
                counter *= len(self.regions)
        elif obj_type == "histogram":
            if self.prepared:
                for pset in self.process_sets:
                    for p in pset:
                        counter += np.sum([len(r.histograms) for r in p])
            else:
                counter += len(self.process_sets)
                counter *= len(self.regions)
                counter *= len(self.histograms)
        elif obj_type == "systematic":
            counter += len(self.list_all_systematics())
        else:
            raise ValueError(f"counting does not support type {obj_type}")

        return counter

    # --------------------------------------------------------------------------
    def clear_children_parent(self):
        for pset in self.process_sets:
            for p in pset:
                p.clear_children_parent()

    def update_children_parent(self):
        for pset in self.process_sets:
            for p in pset:
                p.update_children_parent()


# ==============================================================================
