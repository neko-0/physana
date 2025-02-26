'''
class for configuration manager. Use for steering Process and Region
'''

from .core import Region
from .core import ProcessSet
from .core import Histogram, Histogram2D
from .core import Systematics
from .serialization import Serialization, async_from_pickles
from .utils import from_root, from_numexpr, _expr_var
from copy import deepcopy, copy
from pathlib import Path
from tqdm import tqdm
import ast
import os
import time
import numpy as np
import multiprocessing
import concurrent.futures
import fnmatch, re
import collections
import sys
import json
import _jsonnet
import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


def _intersec(config1, config2):
    """
    Method for getting intersection of two ConfigMgr instances.
    basically moving the non-overlap processes from config2 to config1.
    """
    config1_app = config1.append_process
    config1_ls = config1.list_processes
    config1_get = config1.get_process_set
    for pset2 in config2.process_sets:
        # check by process set name first.
        # If the name does not exist, just append the entire process set object
        if pset2.name not in config1_ls():
            config1_app(pset2)
            continue
        # loop and check through each process in the process set.
        # this include checking process with systematic.
        pset1 = config1_get(pset2.name)
        diff_p = []  # buffer list to hold processes for later appending.
        diff_p_app = diff_p.append
        for p2 in pset2:
            found_diff = True
            for p1 in pset1:
                # note process set always try to yield nominal first
                # process set will skip nominal if nominal == None
                if p1.systematic is None:
                    if p2.systematic is None:
                        # if nominal is in both configs,
                        # break and move to next one in pset2
                        found_diff = False
                        break
                else:
                    if p2.systematic is None:
                        continue
                    if p1.systematic == p2.systematic:
                        found_diff = False
                        break
            if found_diff:
                diff_p_app(p2)
        for p in diff_p:
            config1_app(p, mode="merge")


def extract_systematic_list(
    config,
    syst_list,
    nominal=False,
    skip=None,
    is_theory=False,
    keep_data="data",
):
    """
    Extracting systematics from given list into a new ConfigMgr object.
    Only the group name would be fine.
    """
    c_config = config.copy(shallow=True)
    c_config.clear_process_set()
    if skip is None:
        skip = {}
    # turn on cache of the origin config
    for pset in config.process_sets:
        if pset.name in skip:
            continue
        pset.use_cache = True
        c_pset = pset.copy(shallow=True)
        if not nominal:
            c_pset.nominal = None
        c_pset.systematics = []
        for syst in syst_list:
            pset_list = pset.get(syst)
            if pset_list is pset.nominal:
                if pset.name in skip:
                    continue
                if is_theory and pset.name not in syst:
                    continue
                if pset.name not in syst:
                    logger.warning(f"Only nominal in {syst} for {pset.name}")
                continue
            c_pset.systematics += pset_list
        c_pset.use_cache = False
        pset.use_cache = False
        c_config.append_process(c_pset, copy=False)
    return c_config


def intersect_histogram_systematic_band(config1, config2):
    config1 = ConfigMgr.open(config1)
    config2 = ConfigMgr.open(config2)
    proc_regions = ((x, y) for x in config2.processes for y in x.regions)
    # only nominal process will have the systematic band on histograms
    for process2, region2 in proc_regions:
        try:
            region1 = config1.get_process(process2.name).get(region2.name)
        except KeyError:
            logger.warning(f"Cannot find {process2.name}/{region2.name}. Skipping.")
            continue
        for histo2 in region2:
            if histo2.hist_type == "2d":
                continue
            try:
                histo1 = region1.get(histo2.name)
            except KeyError:
                logger.warning(
                    f"Cannot find {process2.name}/{region2.name}/{histo2.name} in config1"
                )
            if histo1.systematic_band is None:
                histo1._systematic_band = histo2._systematic_band
            elif histo2.systematic_band is None:
                continue
            else:
                for band2 in histo2.systematic_band.values():
                    histo1.update_systematic_band(band2)
    return config1


def file_batch_generator(files, batch_size=None):
    if batch_size is None:
        for f in files:
            yield f
    else:
        if isinstance(files, set):
            files = list(files)
        n_files = len(files)
        for i in range(0, n_files, batch_size):
            yield files[i : i + batch_size]


class CorrectionContainer:
    """
    Container for holding systematics histogram for weight lookup.
    This container will only try to lookup requested histogram from the database.
    It's better to use shelve for object persistance to avoid large dictionary
    deserialization. The database by default will be clear after each lookup, and
    this is handled by the db_persistance boolean switch.

    TODO:
        1) need to method to load everything into cache
        2) tuple key is parsed into string when listing, maybe use ast.literal_eval?
    """

    def __init__(self):
        self.files = {}
        self.enable_interanl_cache = True
        # if db_persistance is False, cleanup self._database after each __getitem__
        self.db_persistance = False
        self._correction = {}  # for caching
        self._file_backend = {}
        self._serial = Serialization()
        self._database = []
        self._loaded = False
        self._non_exist_keys = set()  # cache for non_existing keys

    def __getitem__(self, key):
        """
        Lazy get item method.
        """
        if key in self._correction:
            return self._correction[key]
        elif key in self._non_exist_keys:
            return None

        self.load_correction()
        repr_key = repr(key)
        for db in self._database:
            m_key = key if isinstance(db, dict) else repr_key
            try:
                found_corr = db[m_key]
                break
            except KeyError:
                continue
        else:
            found_corr = None
        if not self.db_persistance:
            self.clear_database()
        if self.enable_interanl_cache:
            self._correction[key] = found_corr
        if found_corr is None:
            self._non_exist_keys.add(key)

        return found_corr

    def __setitem__(self, key, value):
        self._correction[key] = value

    def __contains__(self, key):
        return key in self._correction

    def items(self):
        return self._correction.items()

    def keys(self):
        return self._correction.keys()

    def update(self, input: dict):
        for key in input:
            self.__setitem__(key, input[key])

    def list_correction(self):
        output = set()
        for db in self._database:
            output |= set(db.keys())
        return output

    def add_correction_file(self, filename, backend="shelve"):
        filename = Path(filename).resolve()
        if filename not in self.files:
            self.files[filename] = False
            self._file_backend[filename] = backend
        self._loaded = False

    def list_correction_file(self):
        return self.files.keys()

    def remove_correction_file(self, name):
        del self.files[name]

    def load_correction(self):
        if self._loaded:
            return
        for f in self.files:
            if self.files[f] != False:
                continue
            m_backend = self._file_backend[f]
            if m_backend == "shelve":
                db = self._serial.from_shelve(str(f), flag="r", writeback=False)
            elif m_backend == "pickle":
                db = self._serial.from_pickle(str(f))
            else:
                raise TypeError(f"Correction does not support {m_backend}")
            self._database.append(db)
            self.files[f] = True
        self._loaded = True

    def load_in_memory(self):
        self.load_correction()
        for name in tqdm(self.list_correction()):
            if isinstance(name, str):
                # usually handle tuple instead of str for lookup
                name = ast.literal_eval(name)
            self.__getitem__(name)

    def clear_buffer(self):
        self._correction = {}
        self.reset_files()
        self.clear_database()

    def clear_database(self):
        if not self._database:
            return
        for db in self._database:
            if not isinstance(db, dict):
                db.close()
        self._database = []
        self._loaded = False
        self.reset_files()

    def clear_files(self):
        self.files = {}
        self._file_backend = {}

    def reset_files(self):
        for f in self.files:
            self.files[f] = False

    def remove(self, name):
        del self._correction[name]

    def save(self, name):
        """
        save filename into it's dict
        """
        name = Path(name).resolve()
        self.files[name] = True

    def save_current_correction(self, output):
        self._serial.to_shelve(self._correction, output, flag="n")


class ConfigMgr:
    def __init__(
        self,
        src_path="./",
        out_path="./",
        description="",
        histogram_backend="numpy",
    ):
        self.base_dir = os.path.dirname(os.path.realpath(__file__))
        self.src_path = src_path
        self.out_path = f"{self.base_dir}/../../output/{out_path}"
        self.descriptions = description
        self.name = ""
        self.histogram_backend = histogram_backend
        self.output_processed_tag = ""
        self.addition_tag = ""
        self.filename = set()  # use set to prevent duplication
        self.file_record = set()  # record of input files
        self.ofilename = "default_output_name"
        self.prepared = False
        self.USE_MP = True
        self.branches_reserved = False
        self.disable_pbar = False
        self.filled = False
        self.RAISE_TREENAME_ERROR = True
        self.branch_rename = None  # dict for renaming branch in the ntuple
        self.use_cutbook_sum_weights = False
        self.acc_cutbook_sum_weights = False
        self.dsid_branch = "mcChannelNumber"
        self.run_number_branch = "runNumber"

        self.default_tree_sys = "NoSys"

        # default weighting
        self._default_wlist = [
            "genWeight",
            "eventWeight",
            "bTagWeight",
            "pileupWeight",
            "triggerWeight",
            "jvtWeight",
            "leptonWeight",
        ]
        self.default_weight = f"({'*'.join(self._default_wlist)})"
        self.enforce_default_weight = False

        # phase-space correction, pass through HistMaker.meta_data_from_config
        self.corrections = CorrectionContainer()
        self.phasespace_corr_obs = ["nJet30"]
        self.phasespace_apply_nominal = True

        self.meta_data = {}

        try:
            os.makedirs(self.out_path)
        except FileExistsError:
            pass
        else:
            logger.info("create output directory.")

        self.systematics = collections.defaultdict(list)

        self.reserved_branches = set()
        self._process_branch_dict = collections.defaultdict(set)
        self._systematic_branch_dict = collections.defaultdict(set)
        self._region_branch_dict = collections.defaultdict(set)
        self._histogram_branch_dict = collections.defaultdict(set)

        self.process_sets = []
        self.regions = []
        self.aux_regions = []
        self.histograms = []  # this should be private
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
        self._skip_deepcopy = {
            "_skip_deepcopy",
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
        skip_deepcopy_keys = keys & self._skip_deepcopy  # intersection
        deepcopy_keys = keys - skip_deepcopy_keys  # complementary
        for key in skip_deepcopy_keys:
            copy_self.__dict__[key] = copy(self.__dict__[key])
        for key in deepcopy_keys:
            copy_self.__dict__[key] = deepcopy(self.__dict__[key], memo)
        return copy_self

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
            self.out_path,
            self.descriptions,
            self.histogram_backend,
            self.output_processed_tag,
            self.addition_tag,
            self.filename,
            self.ofilename,
            self.prepared,
            self.USE_MP,
        )
        return basic_info

    def config_metadata(self):
        meta = {}
        meta["process"] = self.list_processes()
        meta["path"] = (self.src_path, self.out_path)
        meta['info'] = self.info()
        return meta

    def update_input_file_from_record(self):
        for p_set in self.process_sets:
            for p in p_set:
                p.filename |= p.file_record
        self.filename |= self.file_record

    @staticmethod
    def merge(config_list, *, copy=True, update_filename=False):
        """
        Merging configMgr in the configMgrList.
        Only processes (level) will be appended.
        """
        first_config = None
        first_config_add = None
        for config in tqdm(config_list, unit="config", leave=False):
            if first_config is None:
                first_config = config.copy() if copy else config
                first_config_add = first_config.add
            else:
                first_config_add(config)
                if not update_filename:
                    continue
                pending_proc = (y for x in config.process_sets for y in x)
                for p in pending_proc:
                    first_config.get(p.name).get(p.systematic).update_filename(p)
        first_config.update_children_parent()
        return first_config

    @staticmethod
    def intersection(config_list, *, copy=True):
        """
        Get the intersection of lisf of configMgr objects.
        """
        first_config = None
        for config in tqdm(config_list, unit="config", leave=False):
            if first_config is None:
                first_config = config.copy() if copy else config
            else:
                _intersec(first_config, config)
        first_config.update_children_parent()
        return first_config

    @staticmethod
    def _split_by_input_files(iconfig, shallow=False, batch_size=20, **_):
        for sub_config in ConfigMgr._split_by_processes(iconfig):
            assert len(sub_config.process_sets) == 1
            for pset in sub_config.process_sets:
                assert pset.num_processes() == 1
                for p in pset:
                    if p.filename:
                        c_sub_config = sub_config.copy(shallow=True)
                        c_sub_config.process_sets = []
                        for ifile in file_batch_generator(p.filename, batch_size):
                            c_p = p.copy(shallow=shallow)
                            c_p.filename = ifile
                            cc_sub_config = c_sub_config.copy(shallow=True)
                            cc_sub_config.process_sets = []
                            cc_sub_config.append_process(c_p, copy=False)
                            yield cc_sub_config
                    elif sub_config.filename:
                        for ifile in file_batch_generator(
                            sub_config.filename, batch_size
                        ):
                            c_sub_config = sub_config.copy(shallow=shallow)
                            c_sub_config.filename = {ifile}
                            yield c_sub_config
                    else:
                        yield sub_config

    @staticmethod
    def _split_by_regions(configMgr, region_split_size=5, copy=True):
        """
        spliting config by regions
        """
        r_size = len(configMgr.regions)
        aux_r_size = len(configMgr.aux_regions)
        n = region_split_size
        tot = int(np.ceil(r_size / n))
        tot += int(np.ceil(aux_r_size / n))  # number of region slices
        logger.info(f"{len(configMgr.regions)} regions.")
        logger.info(f"{len(configMgr.aux_regions)} aux regions.")
        logger.info(f"split size = {n}, total slices = {tot}")
        c_config = configMgr.copy(shallow=True)
        corrections = c_config.corrections
        c_config.corrections = None
        c_config.process_sets = []
        c_config.regions = []
        c_config.aux_regions = []
        c_config.disable_pbar = False
        c_config.USE_MP = False
        c_config.prepared = False
        psets = (x for pset in configMgr.process_sets for x in pset)
        for p in psets:
            split_name = p.name
            if p.systematic:
                split_name += f"-{'_'.join(p.systematic.full_name)}"
            else:
                split_name += "-nominal"
            # in the case of configMgr object already prepared, the
            # process needs to clear it's regions, otherwise it will
            # end up with double amount the events.
            c_process = p.copy(shallow=True)
            c_process.regions = []
            rsplit_count = 1
            for i in range(0, r_size, n):
                my_config = c_config.copy(shallow=True)
                my_config.process_sets = []
                my_config.corrections = corrections
                my_config.name = f"s-{split_name}-{rsplit_count}_of_{tot}"
                my_config.append_process(c_process, copy=copy)
                region_slice = configMgr.regions[i : min(n + i, r_size)]
                if copy:
                    my_config.regions = deepcopy(region_slice)
                else:
                    my_config.regions = region_slice
                    c_process = p.copy(shallow=True)
                    c_process.regions = []
                yield my_config
                rsplit_count += 1
            for i in range(0, aux_r_size, n):
                my_config = c_config.copy(shallow=True)
                my_config.process_sets = []
                my_config.corrections = corrections
                my_config.name = f"s-{split_name}-{rsplit_count}_of_{tot}"
                my_config.append_process(c_process, copy=copy)
                region_slice = configMgr.aux_regions[i : min(n + i, aux_r_size)]
                for _r in region_slice:
                    _r.branch_reserved = False
                if copy:
                    my_config.aux_regions = deepcopy(region_slice)
                else:
                    my_config.aux_regions = region_slice
                    c_process = p.copy(shallow=True)
                    c_process.regions = []
                yield my_config
                rsplit_count += 1
            logger.debug(f"splited {rsplit_count} per process for {split_name}")

    @staticmethod
    def _split_by_processes(config, copy=True):
        """
        Splitting config by process
        """
        # prepare copy of the configMgr
        c_config = config.copy(shallow=True)
        corrections = c_config.corrections
        c_config.corrections = None
        c_config.process_sets = []
        c_config.USE_MP = False
        # spliting process in process sets
        psets = (x for pset in config.process_sets for x in pset)
        for p in psets:
            split_name = p.name
            if p.systematic:
                split_name += f"-{'_'.join(p.systematic.full_name)}"
            else:
                split_name += "-nominal"
            my_config = c_config.copy(shallow=True)
            my_config.name = split_name
            my_config.process_sets = []
            my_config.corrections = corrections
            my_config.append_process(p, copy=copy)
            yield my_config
            logger.debug(f"splited process set {split_name}")

    @staticmethod
    def _split_by_systematic(
        config,
        syst_names=None,
        include_nominal=False,
        skip_process=[],
        copy=True,
        with_name=False,
        batch_size=None,
    ):
        """
        Splitting config by given single systematic
        """
        # get process names
        process_sets = [
            pset for pset in config.process_sets if pset.name not in skip_process
        ]
        # turn ON cache
        for pset in process_sets:
            pset.use_cache = True
        # check on systematic list
        if syst_names is None:
            syst_names = config.list_systematic_full_name()
        elif not isinstance(syst_names, list):
            raise ValueError(f"Require list, but receive {type(syst_names)}")
        # prepare copy of the configMgr
        c_config = config.copy(shallow=True)
        corrections = c_config.corrections
        c_config.corrections = None
        c_config.process_sets = []
        try:
            if batch_size is None:
                for syst in syst_names:
                    m_config = c_config.copy(shallow=True)
                    m_config.process_sets = []
                    m_config.corrections = corrections
                    for pset in process_sets:
                        m_proc = pset.get(syst)
                        if syst is None or m_proc.systematic is not None:
                            m_config.append_process(m_proc, copy=copy)
                        # check if nominal is requested
                        if include_nominal and syst is not None:
                            m_config.append_process(
                                pset.nominal, mode="merge", copy=copy
                            )
                    if with_name:
                        yield syst, m_config
                    else:
                        yield m_config
            else:
                n_syst = len(syst_names)
                syst_names = [
                    syst_names[i : i + batch_size] for i in range(0, n_syst, batch_size)
                ]
                for ibatch, syst in enumerate(syst_names):
                    m_config = c_config.copy(shallow=True)
                    m_config.process_sets = []
                    m_config.corrections = corrections
                    for pset in process_sets:
                        if include_nominal:
                            m_config.append_process(pset.nominal, copy=copy)
                        for i_syst in syst:
                            if i_syst is None and include_nominal:
                                # already included nominal, skip
                                continue
                            syst_proc = pset.get(i_syst)
                            if i_syst is not None and syst_proc.systematic is None:
                                # nominal is return but we ask for systematic, skip
                                continue
                            m_config.append_process(syst_proc, mode="merge", copy=copy)
                    if with_name:
                        yield f"syst_batch_{ibatch}", m_config
                    else:
                        yield m_config
        except GeneratorExit:
            logger.debug("finished splitting")
            pass
        finally:
            logger.debug("clean up cache after splitting")
            # turn OFF cache
            for pset in process_sets:
                pset.use_cache = False

    @staticmethod
    def _split_by_regions_and_files(configMgr, *args, **kwargs):
        for config in ConfigMgr._split_by_regions(configMgr, *args, **kwargs):
            for sub_config in ConfigMgr._split_by_input_files(config):
                yield sub_config

    @staticmethod
    def _split_by_processes_and_files(configMgr, *args, **kwargs):
        for config in ConfigMgr._split_by_processes(configMgr, *args, **kwargs):
            for sub_config in ConfigMgr._split_by_input_files(config):
                yield sub_config

    @staticmethod
    def _split_by_systematic_and_process(configMgr, *args, **kwargs):
        syst_split = ConfigMgr._split_by_systematic(
            configMgr, include_nominal=False, copy=False, with_name=True
        )
        for sys_name, sub_config in syst_split:
            for sub_sub_config in ConfigMgr._split_by_processes(
                sub_config, *args, **kwargs
            ):
                yield sys_name, sub_sub_config

    @staticmethod
    def _split_by_systematic_process_files(configMgr, *args, **kwargs):
        for syst_name, sub_config in ConfigMgr._split_by_systematic_and_process(
            configMgr, copy=False
        ):
            for counter, subsub_config in enumerate(
                ConfigMgr._split_by_input_files(sub_config, *args, **kwargs)
            ):
                yield syst_name, counter, subsub_config

    @staticmethod
    def split(configMgr, split_type="process", *args, **kwargs):
        """
        Splitting the processes in configMgr to produce smaller configMgr.
        If the region is set to True, it will also split regions.
        """
        if split_type == "process":
            split_method = ConfigMgr._split_by_processes
        elif split_type == "region":
            split_method = ConfigMgr._split_by_regions
        elif split_type == "systematic":
            split_method = ConfigMgr._split_by_systematic
        elif split_type == "ifile":
            split_method = ConfigMgr._split_by_input_files
        elif split_type == "region-file":
            split_method = ConfigMgr._split_by_regions_and_files
        elif split_type == "process-file":
            split_method = ConfigMgr._split_by_processes_and_files
        elif split_type == "systematic-process":
            split_method = ConfigMgr._split_by_systematic_and_process
        elif split_type == "systematic-process-files":
            split_method = ConfigMgr._split_by_systematic_process_files
        else:
            logger.warning(f"Unknown {split_type=}. using 'process' split type")
            logger.warning("Fallback to default split type")
            split_method = ConfigMgr._split_by_processes

        yield from split_method(configMgr, *args, **kwargs)

    def self_split(self, split_type, *args, **kwargs):
        if split_type == "region":
            logger.warning("splitting by region might clear all filled contents!")
        yield from ConfigMgr.split(self, split_type, *args, **kwargs)

    def set_singlefile(self, filename):
        if isinstance(filename, str):
            self.filename = {f"{self.src_path}/{filename}"}
        elif isinstance(filename, list):
            self.filename = set([f"{self.src_path}/{file}" for file in filename])
        else:
            raise ValueError(f"Files setting receives invalid type {type(filename)}.")
        self.file_record = copy(self.filename)

    def copy(self, *, shallow=False):
        if shallow:
            return copy(self)
        else:
            return deepcopy(self)

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

    def reserve_branches(self, expr=None):
        if isinstance(expr, list):
            parsed = set(expr)
        elif expr:
            parsed = _expr_var(expr)
        else:
            parsed = None
        if parsed:
            self.reserved_branches |= parsed
        return parsed

    def _reserve_region_branches(self, use_mp=True, *, executor=None):
        """
        loop through all regions and reserve weights and observable branches
        in the ROOT TTree.
        """

        t_now = time.perf_counter()

        if self.branches_reserved:
            return None

        if use_mp:
            logger.info("reserving branches with mp.")

            weight = {}
            selection = {}
            if executor:
                logger.info("using executor.")
                for r in self.regions + self.aux_regions:
                    if r.branch_reserved:
                        continue
                    if r.weights:
                        # weight.append(executor.submit(_expr_var, r.weights))
                        weight[r.name] = executor.submit(_expr_var, r.weights)
                    if r.selection:
                        # selection.append(executor.submit(_expr_var, r.selection))
                        selection[r.name] = executor.submit(_expr_var, r.selection)
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
                logger.info("using mp pool.")
                pool = multiprocessing.Pool()
                for r in self.regions + self.aux_regions:
                    if r.branch_reserved:
                        continue
                    if r.weights:
                        weight.append(pool.apply_async(_expr_var, args=(r.weights,)))
                    if r.selection:
                        selection.append(
                            pool.apply_async(_expr_var, args=(r.selection,))
                        )
                pool.close()
                pool.join()

                branches = weight + selection
                num_regions = len(branches)
                with tqdm(total=num_regions, leave=False) as pbar_regions:
                    pbar_regions.disable = self.disable_pbar
                    pbar_regions.set_description("branch reservation")
                    for b in branches:
                        self.reserved_branches |= b.get()
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
                        b_result = self_reserve_branches(r.selection)
                        self._region_branch_dict[r.name] |= b_result
                    pbar_regions.update()

        logger.info(f"branches reservation cost {time.perf_counter()-t_now:.2f}s")
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
                    p_branch |= self.reserve_branches(p.selection)
                if isinstance(p.weights, str):
                    p_branch |= self.reserve_branches(p.weights)
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
            logger.debug(f"adding aux region: {region.name}, type {region.type}")
            self.aux_regions.append(region)
        else:
            logger.debug(f"adding region: {region.name}, type {region.type}")
            self.regions.append(region)

    def add_region(
        self,
        name,
        selection,
        weight=None,
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
        r = Region(name, weight, selection, study_type, corr_type, **kwargs)
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
        kwargs.setdefault("type", "reco")
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

    def add_histogram2D(
        self,
        name,
        xvar,
        yvar,
        xbin=1,
        xmin=-1,
        xmax=1,
        ybin=1,
        ymin=-1,
        ymax=1,
        *args,
        **kwargs,
    ):
        logger.info(f"adding 2D observables {name}")
        kwargs.setdefault("type", "reco")
        self._histogram_branch_dict[name] |= self.reserve_branches(xvar)
        self._histogram_branch_dict[name] |= self.reserve_branches(yvar)
        xvar = from_root(xvar).to_numexpr()
        yvar = from_root(yvar).to_numexpr()
        if isinstance(xbin, list) and isinstance(ybin, list):
            bin_args = (name, xvar, yvar, xbin, ybin)
            _histo2d = Histogram2D.variable_bin(*bin_args, *args, **kwargs)
        else:
            bin_args = (name, xvar, yvar, xbin, xmin, xmax, ybin, ymin, ymax)
            _histo2d = Histogram2D(*bin_args, *args, **kwargs)
        self.histograms2D.append(_histo2d)

    add_observable2D = add_histogram2D

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
            if r.type in self.region_type_filter:
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
                r.add_histogram(hist, enable_filter=True)
                r.ntuple_branches |= self._histogram_branch_dict.get(hist.name, set())
            r.ntuple_branches |= self._region_branch_dict.get(r.name, set())

        # region weight checkfing dict
        found_region_weight = {}
        # distributing regions into processes
        psets = (p for pset in self.process_sets for p in pset)
        for p in psets:
            # check filename for each process
            # if the process does not have filename, the config level
            # filename info will be used.
            if not p.filename:
                p.file_record = self.file_record
                p.filename = self.filename

            # store the Ntuple branches to process
            p.ntuple_branches |= self._process_branch_dict.get(p.name, set())
            p.ntuple_branches |= default_w_branch

            for r in self.regions + self.aux_regions:
                if r.type in self.process_region_type_filter.get(p.name, set()):
                    continue
                r.ntuple_branches |= self._region_branch_dict.get(r.name, set())
                p.add_region(r)
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

    def _get_output_pickle_filename(self):
        return f"{self.ofilename}{self.output_processed_tag}{self.addition_tag}.pkl"

    def get_output_file(self):
        return Path(self.out_path).joinpath(f"{self.ofilename}.pkl").resolve()

    def save(self, filename="", backend="pickle", metadata=False, *args, **kwargs):
        start_t = time.perf_counter()

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
        output_file = Path(self.out_path).joinpath(f"{self.ofilename}{ext}")
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
        logger.info(f"file saved: {output_file}, time {time.perf_counter()-start_t}s")

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
        _t_start = time.perf_counter()
        if isinstance(filename, ConfigMgr):
            return filename
        m_serial = Serialization("config")
        try:
            if backend == "pickle":
                m_config = m_serial.from_pickle(filename)
            elif backend == "shelve":
                m_config = ConfigMgr.merge(m_serial.from_shelve(filename))
            m_config.update_children_parent()
            return m_config
        except Exception as _error:
            raise IOError(f"cannot open file: {filename} using {backend}") from _error
        finally:
            t_diff = time.perf_counter() - _t_start
            logger.info(f"open({filename}) takes {t_diff:.2f}s wall time.")

    @staticmethod
    def open_files(file_list):
        t_now = time.perf_counter()
        opened_files = list(async_from_pickles(file_list))
        t_diff = time.perf_counter() - t_now
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
                if r.type == filter_type:
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
                print(f"{ind * (p_level + 1)}type: {r.type}")
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

    # --------------------------------------------------------------------------
    def set_output_location(self, odir):
        logger.info(f"output location will set to {odir}")
        self.out_path = odir
        if not os.path.exists(self.out_path):
            os.makedirs(self.out_path)

    # --------------------------------------------------------------------------
    def define_systematics(self, name, tlist, wlist, source, **kwargs):
        """
        defining systematics group

        Args:
            name (str) : name of the systematics groups

            tlist (list(str)) : list of name for tree based systematics

            wlist (list(str)) : list of weight (branch name within TTree)
                for weight based systematics

            **kwargs : follow collinearw.core.Systematics
        """
        m_syst_branch = collections.defaultdict(set)
        if name in self.systematics:
            raise ValueError(f"Duplicated name for systematics {name}")
        for _weight in wlist:
            if isinstance(_weight, str):
                m_branch = self.reserve_branches(_weight)
                m_syst_branch[_weight] |= m_branch or set()
        for _t in tlist:
            for _w in wlist:
                sys = Systematics(name, _t, _w, source, **kwargs)
                self.systematics[name].append(sys)
                self._systematic_branch_dict[sys.full_name] |= m_syst_branch[_w]

    def define_weight_systematics(self, name, wlist, **kwargs):
        self.define_systematics(
            name, [self.default_tree_sys], wlist, "weight", **kwargs
        )

    def define_tree_systematics(self, name, tlist, **kwargs):
        self.define_systematics(name, tlist, [""], "tree", **kwargs)

    def set_systematics(self, process_name, sys_name, dummy=False):
        """
        Require defining systematic first before setting it.

        Systematic is defined through ConfigMgr.define_systematics. After defining,
        internal list of systematic will be created and stored, and this list will
        be used for looking up systematic to be set.
        """
        try:
            pset = self.get_process_set(process_name)
            nominal_branches = pset.nominal.ntuple_branches
        except KeyError:
            logger.critical(
                f"Cannot find {process_name}. Maybe you forget add process?"
            )
            raise
        sys_list = [sys_name] if isinstance(sys_name, str) else sys_name
        for name in sys_list:
            for systematic in self.systematics[name]:
                if dummy:
                    syst = systematic.copy()
                    syst.source = "dummy"
                else:
                    syst = systematic
                pset.add_systematic(syst)
                syst_branch = self._systematic_branch_dict.get(syst.full_name, set())
                pset.systematics[-1].ntuple_branches = nominal_branches | syst_branch

    def set_dummy_systematics(self, process_name, sys_name):
        """
        Add dummy copy of a systematic object to a process set. This is require
        for keeping track of all normalization factor due to different systematic
        """
        self.set_systematics(process_name, sys_name, dummy=True)

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
                if proc.systematic is None:
                    self_nominal = self_pset.nominal
                    if self_nominal is None:
                        self_nominal = proc
                    else:
                        self_nominal.regions = proc.regions
                        self_nominal.filename = proc.filename
                    logger.info(f"swapped {proc.name} nominal content.")
                else:
                    syst_name = proc.systematic.full_name
                    self_p = self_pset.get(syst_name)
                    if self_p.systematic is None:
                        logger.info(f"Cannot find {syst_name}")
                        continue
                    else:
                        self_p.regions = proc.regions
                        self_p.filename = proc.filename
                        logger.info(f"swapped {proc.name} {syst_name} content.")

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


class XSecSumEvtW:
    __slots__ = (
        "xsec_file",
        "lumi",
        "dsid",
        "xsec_name",
        "sumw_name",
        "nominal_token",
        "token_groups",
        "token_groups_rule",
        "campaign_sensitive",
        "campaign_files",
        "campaign_lumi",
        "campaign_xsec",
        "do_check_process",
        "check_map",
        "weight_base_token",
        "use_process_nominal_tree",
        "duplicated_list",
        "duplicated_sets",
        "duplicated_skip_campaign",
        "duplicated_accept",
        "remove_wsyst_ptag",
        "_xsec",
        "_is_data",
        "_cached_w",
        "_re_expr",
        "_c_re",
    )

    def __init__(self, xsec_file=None):
        self.xsec_file = xsec_file
        self.lumi = 1.0
        self.dsid = "DatasetNumber"
        self.xsec_name = "xsec"
        self.sumw_name = "AllExecutedEvents_sumOfEventWeights"
        self.nominal_token = "weight_1"
        self.token_groups = {}
        self.token_groups_rule = {
            "path": 1,
            "campaign": 2,
            "prefix": 2,
            "user": 5,
            "process": 6,
            "dsid": 7,
            "ptag": 8,
            "user-tag": 9,
            "syst": 10,
            "suffix": 11,
        }
        self.campaign_sensitive = False
        self.campaign_files = {}
        self.campaign_lumi = {}
        self.campaign_xsec = None
        self.do_check_process = False
        self.check_map = {}
        self.weight_base_token = None
        self.use_process_nominal_tree = False
        self.duplicated_list = ["LHE3Weight", "pileupWeight", "triggerWeight"]
        self.duplicated_sets = None  # e.g. {'triggerWeight' : 'weight_2'}
        self.duplicated_skip_campaign = None  # e.g. {'e' : 'pileupWeight'}
        self.duplicated_accept = None  # e.g. {'weight_2', 'kinematic_1'}
        # self.remove_wsyst_ptag has the form e.g. {'zjets_2211':{'d':{'bTagWeight' : 'p4512'}}}
        self.remove_wsyst_ptag = None
        self._xsec = None
        self._is_data = False
        self._cached_w = None
        # default regular expression
        self._re_expr = "(.*)mc16([ade])(.*)(user).(.\\w*).(.\\w*).(.\\d*).(.*).(CollinearW.SMA.v1)_(.*)_(t2_tree.root)"
        self._c_re = re.compile(self._re_expr)

    def __getitem__(self, x):
        return self.token_groups[x]

    def keys(self):
        return self.token_groups.keys()

    def items(self):
        return self.token_groups.items()

    def nominal(self, fname):
        if self.nominal_token not in fname:
            return False
        else:
            return True

    def check_syst_set(self, fname, syst_set=None):
        if syst_set is None:
            return self.nominal(fname)
        else:
            return syst_set in fname

    def set_campaign_files(self, values):
        self.campaign_files = values

    def set_campaign_lumi(self, lumi):
        self.campaign_lumi = lumi

    def check_process(self, process: str):
        if not self.do_check_process:
            return True
        current_process = "data" if self._is_data else self["process"]
        return self.check_map.get(process, process) == current_process

    def get_xsec_sumw(self, dsid=None, syst=None):
        if self._is_data:
            return 1.0
        if self.campaign_sensitive:
            return self.get_campaign_xsec_sumw()
        if self._xsec is None:
            if self.xsec_file is None:
                raise IOError("no xsec file specified.")
            with open(self.xsec_file) as f:
                self._xsec = json.load(f)
        if dsid is None:
            dsid = self["dsid"]
        if syst is None:
            syst = self["syst"]
        dataset = self._xsec[dsid][syst]
        return dataset[self.xsec_name] / dataset[self.sumw_name] * self.lumi

    def get_campaign_xsec_sumw(self):
        if self._is_data:
            return 1.0
        if self.campaign_xsec is None:
            self.campaign_xsec = {}
            for mc, fname in self.campaign_files.items():
                with open(fname) as f:
                    self.campaign_xsec[mc] = json.load(f)
        current_campaign = self["campaign"]
        dsid = self["dsid"]
        syst = self["syst"]
        lumi = self.campaign_lumi.get(current_campaign, 1.0)
        dataset = self.campaign_xsec[current_campaign][dsid][syst]
        return dataset[self.xsec_name] / dataset[self.sumw_name] * lumi

    def get_auto(self):
        if self._cached_w is None:
            self._cached_w = self.get_xsec_sumw()
        return self._cached_w

    def event_weight(self, *args, **kwargs):
        return self.get_xsec_sumw(*args, **kwargs)

    def set_expression(self, expr):
        self._re_expr = expr
        self._c_re = re.compile(expr)

    def expr(self):
        return self._re_expr

    def match(self, fname):
        tokens = self._c_re.match(fname)
        if self.token_groups:
            self.token_groups.update(
                {x: tokens.group(y) for x, y in self.token_groups_rule.items()}
            )
        else:
            self.token_groups = {
                x: tokens.group(y) for x, y in self.token_groups_rule.items()
            }
        self._is_data = "data" in self["process"]
        self._cached_w = None

    @classmethod
    def open(cls, xsecf):
        obj = cls()
        m_serial = Serialization("xsec")
        json_data = m_serial.from_json(xsecf)
        obj.set_expression(json_data.pop("expr"))
        for name, value in json_data.items():
            setattr(obj, name, value)
        return obj

    def save(self, fname):
        m_serial = Serialization("xsec")
        return m_serial.to_json(self, fname)
