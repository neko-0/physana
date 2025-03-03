"""
This file contains some helper functions for dispatching large ConfigMgr instances into smaller ones
for parallelization processing.
"""

import logging
from math import ceil
from copy import deepcopy

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


def file_batch_generator(files, batch_size=None):
    if batch_size is None:
        yield from files
    else:
        if not isinstance(files, list):
            files = list(files)
        batch_start = 0
        while batch_start < len(files):
            yield files[batch_start : batch_start + batch_size]
            batch_start += batch_size


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


def split_by_processes(config, copy=True):
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
        if p.systematics:
            split_name += f"-{'_'.join(p.systematics.full_name)}"
        else:
            split_name += "-nominal"
        my_config = c_config.copy(shallow=True)
        my_config.name = split_name
        my_config.process_sets = []
        my_config.corrections = corrections
        my_config.append_process(p, copy=copy)
        yield my_config
        logger.debug(f"splited process set {split_name}")


def split_by_input_files(iconfig, shallow=False, batch_size=20, **_):
    for sub_config in split_by_processes(iconfig):
        assert len(sub_config.process_sets) == 1
        for pset in sub_config.process_sets:
            assert pset.num_processes() == 1
            for p in pset:
                if p.input_files:
                    c_sub_config = sub_config.copy(shallow=True)
                    c_sub_config.process_sets = []
                    for ifile in file_batch_generator(p.input_files, batch_size):
                        c_p = p.copy(shallow=shallow)
                        c_p.input_files = ifile
                        cc_sub_config = c_sub_config.copy(shallow=True)
                        cc_sub_config.process_sets = []
                        cc_sub_config.append_process(c_p, copy=False)
                        yield cc_sub_config
                elif sub_config.input_files:
                    for ifile in file_batch_generator(sub_config.input_files, batch_size):
                        c_sub_config = sub_config.copy(shallow=shallow)
                        c_sub_config.input_files = {ifile}
                        yield c_sub_config
                else:
                    yield sub_config


def split_by_regions(configMgr, region_split_size=5, copy=True):
    """
    spliting config by regions
    """
    r_size = len(configMgr.regions)
    aux_r_size = len(configMgr.aux_regions)
    n = region_split_size
    tot = int(ceil(r_size / n))
    tot += int(ceil(aux_r_size / n))  # number of region slices
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


def split_by_systematic(
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
                        m_config.append_process(pset.nominal, mode="merge", copy=copy)
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


def split_by_regions_and_files(configMgr, *args, **kwargs):
    for config in split_by_regions(configMgr, *args, **kwargs):
        for sub_config in split_by_input_files(config):
            yield sub_config


def split_by_processes_and_files(configMgr, *args, **kwargs):
    for config in split_by_processes(configMgr, *args, **kwargs):
        for sub_config in split_by_input_files(config):
            yield sub_config


def split_by_systematic_and_process(configMgr, *args, **kwargs):
    syst_split = split_by_systematic(
        configMgr, include_nominal=False, copy=False, with_name=True
    )
    for sys_name, sub_config in syst_split:
        for sub_sub_config in split_by_processes(sub_config, *args, **kwargs):
            yield sys_name, sub_sub_config


def split_by_systematic_process_files(configMgr, *args, **kwargs):
    for syst_name, sub_config in split_by_systematic_and_process(configMgr, copy=False):
        for counter, subsub_config in enumerate(
            split_by_input_files(sub_config, *args, **kwargs)
        ):
            yield syst_name, counter, subsub_config


def split(configMgr, split_type="process", *args, **kwargs):
    """
    Splitting the processes in configMgr to produce smaller configMgr.
    If the region is set to True, it will also split regions.
    """
    if split_type == "process":
        split_method = split_by_processes
    elif split_type == "region":
        split_method = split_by_regions
    elif split_type == "systematic":
        split_method = split_by_systematic
    elif split_type == "ifile":
        split_method = split_by_input_files
    elif split_type == "region-file":
        split_method = split_by_regions_and_files
    elif split_type == "process-file":
        split_method = split_by_processes_and_files
    elif split_type == "systematic-process":
        split_method = split_by_systematic_and_process
    elif split_type == "systematic-process-files":
        split_method = split_by_systematic_process_files
    else:
        logger.warning(f"Unknown {split_type=}. using 'process' split type")
        logger.warning("Fallback to default split type")
        split_method = split_by_processes

    yield from split_method(configMgr, *args, **kwargs)
