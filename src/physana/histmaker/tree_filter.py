import logging
import math
import json
import multiprocessing as mp
from time import perf_counter
from functools import partial
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import uproot

from .histmaker import HistMaker

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _filter_missing_ttree(process, *args, branch_rename=None, **kwargs):
    """
    check TTree in TFile for a single process
    """
    t_start = perf_counter()
    nfiles = len(process.input_files)
    logger.info(f"filtering empty ttree for {process.name} in {nfiles} files")
    histmaker = HistMaker(*args, **kwargs)
    histmaker.RAISE_TREENAME_ERROR = True
    histmaker.branch_rename = branch_rename
    open_root_file = histmaker.open_file
    open_root_file = uproot.open
    check_ttree = histmaker.resolve_ttree

    def _check(file):
        with open_root_file(file) as opened_f:
            try:
                ttree = check_ttree(opened_f, process)
            except RuntimeError:
                return None
            if ttree is None or ttree.num_entries == 0:
                return None
        return file

    with ThreadPoolExecutor() as pool:
        future_map = pool.map(_check, process.input_files)
        filtered_files = set(future_map) - {None}

    p_name = (process.name, process.systematics)
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
            key = str((process.name, process.systematics))
            if key in good_file_list:
                process.input_files = good_file_list[key]
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
        n_workers = int(math.ceil(0.5 * mp.cpu_count()))
        with ProcessPoolExecutor(n_workers) as pool:
            future_map = pool.map(_filter, flat_proc_list)
            for filtered_files in future_map:
                next(proc_list).input_files = filtered_files
    else:
        for process in flat_proc_list:
            process.input_files = _filter(process)
