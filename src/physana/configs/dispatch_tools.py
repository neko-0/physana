"""
This file contains some helper functions for dispatching large ConfigMgr instances into smaller ones
for parallelization processing.
"""

import logging
import uproot
import typing
from math import ceil
from copy import deepcopy
from typing import Iterator, Optional, Union

if typing.TYPE_CHECKING:
    from pathlib import Path
    from ..configs import ConfigMgr

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


def file_batch_generator(
    files: Union[list[str], list["Path"]], batch_size: int = None
) -> Iterator[Union[list[str], list["Path"]]]:
    """
    A generator that yield batches of files from given list.

    Parameters
    ----------
    files : list[Union[str, pathlib.Path]]
        list of files to be batched.
    batch_size : int
        The size of each batch. If None, all files would be returned at once.

    Yields
    ------
    list[Union[str, pathlib.Path]]
        A list of files in the current batch.
    """
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
    config: "ConfigMgr",
    syst_list: list[str],
    nominal: bool = False,
    skip: Union[None, set] = None,
    is_theory: bool = False,
    keep_data: str = "data",
) -> "ConfigMgr":
    """
    Extract systematics from a given list into a new ConfigMgr object.

    Parameters
    ----------
    config : ConfigMgr
        The configuration manager object.
    syst_list : list[str]
        list of systematic names to extract.
    nominal : bool, optional
        Whether to include nominal values, by default False.
    skip : Union[None, set], optional
        Set of process names to skip, by default None.
    is_theory : bool, optional
        Flag indicating if it's a theory systematic, by default False.
    keep_data : str, optional
        Data to keep, by default "data".

    Returns
    -------
    ConfigMgr
        A new configuration manager object with extracted systematics.
    """
    c_config = config.copy(shallow=True)
    c_config.clear_process_set()
    if skip is None:
        skip = set()
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


def split_by_processes(config: "ConfigMgr", copy: bool = True) -> Iterator["ConfigMgr"]:
    """
    Splitting config by process

    Parameters
    ----------
    config : ConfigMgr
        Configuration manager to split
    copy : bool, optional
        If True, the processes are copied, by default True

    Yields
    ------
    ConfigMgr
        A new configuration manager object with a single process
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


def split_by_input_files(
    iconfig: "ConfigMgr", shallow: bool = False, batch_size: int = 20, **_
) -> Iterator["ConfigMgr"]:
    """
    Split configuration by input files.

    Parameters
    ----------
    iconfig : ConfigMgr
        The initial configuration manager with processes and input files.
    shallow : bool, optional
        Whether to perform a shallow copy of processes, by default False.
    batch_size : int, optional
        Number of files per batch, by default 20.

    Yields
    ------
    ConfigMgr
        A configuration manager object for each batch of input files.
    """
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
                    for ifile in file_batch_generator(
                        sub_config.input_files, batch_size
                    ):
                        c_sub_config = sub_config.copy(shallow=shallow)
                        c_sub_config.input_files = {ifile}
                        yield c_sub_config
                else:
                    yield sub_config


def split_by_entries(
    iconfig: "ConfigMgr", nbatch: int = 5, tree_name: str = "reco", **_
) -> Iterator[tuple["ConfigMgr", int, int]]:
    """
    Split configuration into batches of entries.

    Parameters
    ----------
    iconfig : ConfigMgr
        The initial configuration manager with processes and input files.
    nbatch : int, optional
        Number of batches, by default 5.
    tree_name : str, optional
        The name of the TTree to use, by default "reco".

    Yields
    ------
    ConfigMgr
        The configuration manager for a batch of entries.
    int
        The starting entry of the batch.
    int
        The ending entry of the batch.
    """
    for config in split_by_input_files(iconfig, batch_size=1):
        assert len(config.process_sets) == 1
        for pset in config.process_sets:
            assert pset.num_processes() == 1
            for p in pset:

                if not p.input_files:
                    yield config, None, None
                    continue

                # Count total entries from ttree
                assert len(p.input_files) == 1
                ifile = list(p.input_files)[0]
                with uproot.open(ifile) as f:
                    if tree_name not in f:
                        yield config, None, None
                        continue

                    total_entries = f[tree_name].num_entries
                    if total_entries == 0:
                        yield config, None, None
                        continue

                # Make a clean copy of the ConfigMgr
                clean_config = config.copy(shallow=True)
                clean_config.process_sets = []

                # Divide entries into batches
                batch_size = int(total_entries / nbatch)
                for batch_start in range(0, total_entries, batch_size):
                    c_config = clean_config.copy(shallow=True)
                    c_config.process_sets = []
                    c_p = p.copy(shallow=True)
                    c_p.input_files = {ifile}
                    c_config.append_process(c_p, copy=False)
                    yield c_config, batch_start, batch_start + batch_size


def split_by_regions(
    configMgr: "ConfigMgr", region_split_size: int = 5, copy: bool = True
) -> Iterator["ConfigMgr"]:
    """
    Splitting config by regions

    Parameters
    ----------
    configMgr : ConfigMgr
        Configuration manager to split
    region_split_size : int, optional
        Split size, by default 5
    copy : bool, optional
        If True, the processes are copied, by default True

    Yields
    ------
    ConfigMgr
        A new configuration manager object with a single process
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
        if p.systematics:
            split_name += f"-{'_'.join(p.systematics.full_name)}"
        else:
            split_name += "-nominal"
        # in the case of configMgr object already prepared, the
        # process needs to clear it's regions, otherwise it will
        # end up with double amount the events.
        c_process = p.copy(shallow=True)
        c_process.clear()
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
                c_process.clear()
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
                c_process.clear()
            yield my_config
            rsplit_count += 1
        logger.debug(f"splited {rsplit_count} per process for {split_name}")


def split_by_histograms(config, with_name=False):

    def split_helper():
        for process_split in split_by_processes(config, copy=False):
            for region_split in split_by_regions(process_split, 1, copy=False):
                yield region_split

    for split_config in split_helper():
        n_regions = len(split_config.regions)
        n_aux_regions = len(split_config.aux_regions)
        assert n_regions + n_aux_regions == 1
        full_name = None
        if with_name:
            proc = list((x for x in split_config.process_sets))[0]
            if proc.systematics:
                pname = f"{proc.name}/{proc.systematics.full_name}"
            else:
                pname = f"{proc.name}/NOSYS"
            if split_config.regions:
                rname = split_config.regions[0].name
            elif split_config.aux_regions:
                rname = split_config.aux_regions[0].name
            full_name = f"{pname}/{rname}"
        for hist in config.histograms:
            c_config = split_config.copy(shallow=True)
            c_config.histograms = [hist.copy(shallow=True)]
            c_config.histograms2D = []
            yield (f"{full_name}/{hist.name}", c_config) if with_name else c_config
        for hist in config.histograms2D:
            c_config = split_config.copy(shallow=True)
            c_config.histograms = []
            c_config.histograms2D = [hist.copy(shallow=True)]
            yield (f"{full_name}/{hist.name}", c_config) if with_name else c_config


def split_by_systematic(
    config: "ConfigMgr",
    syst_names: Optional[list[str]] = None,
    include_nominal: bool = False,
    skip_process: list[str] = [],
    copy: bool = True,
    with_name: bool = False,
    batch_size: Optional[int] = None,
) -> Iterator[Union["ConfigMgr", tuple[str, "ConfigMgr"]]]:
    """
    Splitting config by given single systematic

    Parameters
    ----------
    config : ConfigMgr
        configuration manager to split
    syst_names : list[str], optional
        list of systematic names to split, by default None
    include_nominal : bool, optional
        include nominal values, by default False
    skip_process : list[str], optional
        list of process names to skip, by default []
    copy : bool, optional
        whether to copy process sets, by default True
    with_name : bool, optional
        whether to yield tuple with name, by default False
    batch_size : Optional[int], optional
        batch size, by default None

    Yields
    ------
    ConfigMgr
        a new configuration manager object with a single systematic
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


def split_by_regions_and_files(
    configMgr: "ConfigMgr", *args, **kwargs
) -> Iterator["ConfigMgr"]:
    """
    Splitting the processes in configMgr to produce smaller configMgr.
    Split by regions first, then split by input files.
    """

    for config in split_by_regions(configMgr, *args, **kwargs):
        for sub_config in split_by_input_files(config):
            yield sub_config


def split_by_processes_and_files(
    configMgr: "ConfigMgr", *args, **kwargs
) -> Iterator["ConfigMgr"]:
    """
    Splitting the processes in configMgr to produce smaller configMgr.
    Split by processes first, then split by input files.
    """
    for config in split_by_processes(configMgr, *args, **kwargs):
        for sub_config in split_by_input_files(config):
            yield sub_config


def split_by_systematic_and_process(
    configMgr: "ConfigMgr", *args, **kwargs
) -> Iterator[tuple[str, "ConfigMgr"]]:
    """
    Splitting the processes in configMgr to produce smaller configMgr.
    Split by systematic first, then split by process.
    """
    syst_split = split_by_systematic(
        configMgr, include_nominal=False, copy=False, with_name=True
    )
    for sys_name, sub_config in syst_split:
        for sub_sub_config in split_by_processes(sub_config, *args, **kwargs):
            yield sys_name, sub_sub_config


def split_by_systematic_process_files(
    configMgr: "ConfigMgr", *args, **kwargs
) -> Iterator[tuple[str, int, "ConfigMgr"]]:
    """
    Splitting the processes in configMgr to produce smaller configMgr.
    Split by systematic first, then split by process and then by input files.
    """
    for syst_name, sub_config in split_by_systematic_and_process(configMgr, copy=False):
        for counter, subsub_config in enumerate(
            split_by_input_files(sub_config, *args, **kwargs)
        ):
            yield syst_name, counter, subsub_config


def split(
    configMgr: "ConfigMgr", split_type: str = "process", *args, **kwargs
) -> Iterator[
    Union["ConfigMgr", tuple[str, "ConfigMgr"], tuple[str, int, "ConfigMgr"]]
]:
    """
    Splitting the processes in configMgr to produce smaller configMgr based on the split type.

    Parameters
    ----------
    configMgr : ConfigMgr
        Configuration manager to be split.
    split_type : str, optional
        Type of split to perform: "process", "region", "histogram", "systematic", "ifile",
        "region-file", "process-file", "systematic-process", "systematic-process-files",
        or "entries". Defaults to "process".
    *args
        Additional arguments passed to the specific split function.
    **kwargs
        Additional keyword arguments passed to the specific split function.

    Yields
    ------
    Iterator[Union[ConfigMgr, tuple[str, ConfigMgr], tuple[str, int, ConfigMgr]]]
        An iterator over the split configuration manager objects.
    """
    if split_type == "process":
        split_method = split_by_processes
    elif split_type == "region":
        split_method = split_by_regions
    elif split_type == "histogram":
        split_method = split_by_histograms
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
    elif split_type == "entries":
        split_method = split_by_entries
    else:
        logger.warning(f"Unknown {split_type=}. Using 'process' split type")
        logger.warning("Fallback to default split type")
        split_method = split_by_processes

    yield from split_method(configMgr, *args, **kwargs)
