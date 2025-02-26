import collections
import importlib
import os
import warnings
import time
import multiprocessing as mp
import logging
import json
from concurrent.futures import as_completed as cf_as_completed
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from uuid import uuid4
from pathlib import Path

import numpy as np

from . import strategies
from .histMaker import HistMaker, filter_missing_ttree
from .configMgr import ConfigMgr
from .core import SystematicBase


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def histmaker_generic_interface(config, histmaker=None, **process_kwargs):
    """
    Generic interface to fill a ConfigMgr object using a HistMaker.

    Parameters
    ----------
    config : str or ConfigMgr
        The input config to be filled.
    histmaker : HistMaker, optional
        The instance of HistMaker to be used. If None, a default HistMaker
        will be created.
    **process_kwargs : dict
        Keyword arguments to be passed to histmaker.process()

    Returns
    -------
    filled_config : ConfigMgr
        The filled config object
    """
    if histmaker is None:
        histmaker = HistMaker()
    elif not isinstance(histmaker, HistMaker):
        raise RuntimeError(f"histmaker {type(histmaker)} is not instance of HistMaker")
    config = ConfigMgr.open(config)
    if config.filled:
        logger.warning("config is already filled")
        return config
    if not config.prepared:
        config.prepare()
    histmaker.initialize()
    histmaker.meta_data_from_config(config)
    histmaker.process(config, **process_kwargs)
    histmaker.finalise()
    config.filled = True
    return config


# -----------------------------------------------------------------------------
def run_HistMaker(
    config,
    *,
    split_type=None,
    n_workers=None,
    filter_ttree=False,
    external_weight=None,
    **kwargs,
):
    """
    Top-level HistMaker runner method

    Parameters
    ----------
    config : str or ConfigMgr
        The configuration input, either as a file path or a ConfigMgr instance.
    split_type : str, optional
        Type of split to perform, if any.
    n_workers : int, optional
        Number of workers to use for processing.
    filter_ttree : bool, default False
        Whether to filter missing TTree entries.
    external_weight : str, optional
        Path to an external weight file.
    **kwargs : dict
        Additional keyword arguments for processing.

    Returns
    -------
    ConfigMgr
        The processed ConfigMgr object.
    """
    # Open the configuration file or use the provided ConfigMgr instance
    config = ConfigMgr.open(config)

    # Optionally filter missing TTree entries
    if filter_ttree:
        filter_missing_ttree(config)

    # Choose the processing method based on whether a split type is provided
    if split_type is not None:
        # Run the HistMaker process with splitting
        return _run_HistMaker_split_process(
            config,
            split_type,
            n_workers,
            external_weight,
            **kwargs,
        )
    else:
        # Run the standard HistMaker process
        return _run_HistMaker(config, external_weight, **kwargs)


# -----------------------------------------------------------------------------
def run_HistMaker_syst(
    config,
    executor=None,
    output="syst_config",
    ext_pweight=None,
    as_completed=None,
    filter_ttree=False,
    type=None,
    **kwargs,
):
    """
    similar to other HistMaker process, but pre-set for handle systematics.

    Args:
        config: ConfigMgr
            instance ConfigMgr prepared with systematics)

        exeutor: default=None
            process pool that have submit() method

    """
    logger.info("Running HistMaker with systematics")
    if not os.path.exists(output):
        os.makedirs(output)

    if executor is None:
        workers = int(np.ceil(0.5 * os.cpu_count()))
        pool = ProcessPoolExecutor(workers)
    else:
        pool = executor

    if as_completed is None:
        as_completed = cf_as_completed

    if filter_ttree:
        logger.info("filtering missing ttree.")
        filter_missing_ttree(config)

    # name = config.name  # grab the origin config name
    split_configs = ConfigMgr.split(
        config,
        "systematic",
        include_nominal=False,
        copy=False,
        with_name=True,
        **kwargs,
    )
    pending_jobs = []
    perf_counter = time.perf_counter()
    with open(f"{output}/splitted_tracker.txt", "w+") as split_tracker:
        for counter, (syst_name, sub_config) in enumerate(split_configs):
            split_tracker.write(f"{syst_name}\n")
            sub_config.name = syst_name  # update the config name to syst name
            sub_config.disable_pbar = True
            if syst_name:
                syst_name = '_'.join(syst_name)
                syst_name = syst_name.replace("/", "_")
            sub_config = sub_config.save(f"{output}/input/histmaker_{syst_name}")
            if ext_pweight:
                job = (sub_config, ext_pweight, type)
            else:
                job = (sub_config, None, type)
            pending_jobs.append(pool.submit(_process, job))
            if counter % 100 == 0:
                t_diff = time.perf_counter() - perf_counter
                logger.info(f"submitted {counter} jobs [{t_diff:.2f}s]")
        logger.info(f"total number of splited objects {counter}")

    # set the first opened config to None to hint resource recycle in next GC
    config = None

    # start checking submitted jobs.
    perf_counter = time.perf_counter()
    n_pending = len(pending_jobs)
    with tqdm(total=n_pending, leave=False) as merged_pbar:
        merged_pbar.set_description("Retrieving jobs")
        with open(f"{output}/finish_tracker.txt", "w+") as finish_tracker:
            for pending_job in as_completed(pending_jobs):
                fname = pending_job.result()
                sub_finish_config = ConfigMgr.open(fname)
                syst_name = sub_finish_config.name
                finish_tracker.write(f"{syst_name}\n")
                sub_finish_config.save(f"{output}/{syst_name}")
                os.unlink(fname)
                merged_pbar.update()
                del pending_job  # as_completed does not release future object

    if not executor:
        pool.shutdown()

    return None


# -----------------------------------------------------------------------------
def _prep_syst_process(iconfig):
    c_config = ConfigMgr.open(iconfig)
    # remove if iconfig is string/filename
    if isinstance(iconfig, str):
        os.unlink(iconfig)
    if not c_config.prepared:
        c_config.prepare(use_mp=False)
    filter_missing_ttree(c_config, use_mp=False)
    return c_config.save()


def _syst_process(
    config, show_progress=False, split_type=None, n_workers=None, **kwargs
):
    iconfig, ext_pweight, type = config

    if split_type:
        c_config = run_HistMaker(
            iconfig,
            split_type=split_type,
            copy=False,
            filter_ttree=True,
            n_workers=n_workers,
            **kwargs,
        )
        c_config.corrections.clear_buffer()
        return c_config.save(f"temp_output/{uuid4()}")

    if type is None:
        hist_maker = HistMaker()
    elif type == 'bootstrap':
        hist_maker = strategies.bootstrap.BootstrapHistMaker()
    c_config = ConfigMgr.open(iconfig)
    if not c_config.prepared:
        c_config.prepare(use_mp=False)
    if show_progress:
        c_config.disable_pbar = False
    logger.info(f"Filtering missing tree for {c_config.name}")
    filter_missing_ttree(c_config, use_mp=False)
    logger.info(f"start processing {c_config.name}")
    hist_maker.initialize()
    hist_maker.meta_data_from_config(c_config)
    hist_maker.process(c_config, ext_pweight=ext_pweight)
    hist_maker.finalise()
    c_config.corrections.clear_buffer()
    # save the file on disk instead of returning the object.
    return c_config.save(f"temp_output/{uuid4()}")


def _parse_futures(futures):
    for future in futures:
        try:
            data = future.result()
        except Exception as _err:
            logger.warning(f"cannot parse {future} due to: {_err}")
            data = None
        # future.__del__()
        del future
        if data is None:
            continue
        yield data


def prepare_systematic(
    config,
    output="syst_config",
    ext_pweight=None,
    executor=None,
    as_completed=None,
    type=None,
    do_prepare=True,
    job_prepare=None,
    filebatch=False,
    **kwargs,
):
    """
    Method for splitting a ConfigMgr instance into smaller ConfigMgr instances.
    Each instance will only have one type of systematics. A tracker text file
    is return and it can be used in the run_prepared_systematic() method.

    Args:
        config : ConfigMgr
            instance of ConfigMgr with systematics.

        output : str, default='syst_config'
            output directory for splitted config files.

        do_prepare : bool, default=True
            run preparation step for the splitted config files.

    Returns:
        dict : {str:str}
            dictionary object with {systematic name : path to splitted config}
    """
    logger.info("Preparing config with systematics")
    if not os.path.exists(output):
        os.makedirs(output)

    if executor is None:
        workers = int(np.ceil(0.5 * os.cpu_count()))
        pool = ProcessPoolExecutor(workers)
    else:
        pool = executor

    if as_completed is None:
        as_completed = cf_as_completed

    if job_prepare is None and do_prepare:
        logger.info("Using default job prepare method.")
        job_prepare = _prep_syst_process
    elif job_prepare and not do_prepare:
        logger.warning(f"Using user define job prepare method, but {do_prepare=}")

    # name = config.name  # grab the origin config name
    split_configs = ConfigMgr.split(
        config,
        "systematic-process-files" if filebatch else "systematic-process",
        copy=False,
        **kwargs,
    )
    pending_jobs = []
    split_tracker = {}  # f"{output}/splitted_tracker.json"
    perf_counter = time.perf_counter()
    for counter, config_setting in enumerate(split_configs):
        if filebatch:
            syst_name, fbatch, sub_config = config_setting
            fbatch = f"_fbatch_{fbatch}"
        else:
            syst_name, sub_config = config_setting
            fbatch = ""
        _processes = sub_config.list_processes()
        assert len(_processes) == 1
        if syst_name:
            flat_syst_name = '_'.join(syst_name).replace("/", "_")
        else:
            flat_syst_name = syst_name
        flat_syst_name = f"{_processes[0]}_{flat_syst_name}{fbatch}"
        sub_config.name = flat_syst_name  # update the config name to syst name
        sub_config.disable_pbar = True
        sub_config = sub_config.save(f"{output}/input/histmaker_{flat_syst_name}")
        if do_prepare:
            pending_jobs.append(pool.submit(job_prepare, sub_config))
        split_tracker[flat_syst_name] = str(sub_config)  # It's a PosixPath
        if counter % 100 == 0:
            t_diff = time.perf_counter() - perf_counter
            logger.info(f"submitted {counter} jobs [{t_diff:.2f}s]")
    logger.info(f"total number of splited objects {counter}")

    # set the first opened config to None to hint resource recycle in next GC
    config = None

    with open(f"{output}/splitted_tracker.json", "w") as ofile:
        ofile.write(json.dumps(split_tracker, indent=4))

    if not do_prepare:
        return split_tracker

    # start checking submitted jobs.
    output_jobs = []
    n_pending = len(pending_jobs)
    perf_counter = time.perf_counter()
    with tqdm(total=n_pending, leave=False) as merged_pbar:
        merged_pbar.set_description("Retrieving jobs")
        for pending_job in as_completed(pending_jobs):
            output_jobs.append(pending_job.result())
            merged_pbar.update()
            del pending_job  # as_completed does not release future object

    if not executor:
        pool.shutdown()

    return split_tracker


def run_prepared_systematic(
    input_tracker,
    output="syst_config",
    ext_pweight=None,
    executor=None,
    as_completed=None,
    nworkers=1,
    use_mp=True,
    check_exist=True,
    type=None,
    start_findex=None,
    end_findex=None,
    **kwargs,
):
    """
    similar to other HistMaker process, but only used with prepare_systematic()

    Args:
        input_tracker : str
            name of the JSON file generated by prepare_systematic() method.

        output : str
            output directory for filled config files.

        exeutor: default=None
            process pool that have submit() method

    """
    logger.info("Running HistMaker with systematics")
    if not os.path.exists(output):
        os.makedirs(output)

    if isinstance(input_tracker, str):
        with open(f"{output}/{input_tracker}.json", "r") as ifile:
            input_tracker = json.load(ifile)
        if start_findex is not None or end_findex is not None:
            input_filtered = {}
            input_tracker_list = list(input_tracker)
            if start_findex is None:
                start_findex = 0
            if end_findex is None:
                end_findex = len(input_tracker_list)
            logger.info(f"Only submitting findex ({start_findex}, {end_findex})")
            for i in range(start_findex, end_findex):
                name = input_tracker_list[i]
                input_filtered[name] = input_tracker[name]
            input_tracker = input_filtered
            nworkers = min(nworkers, end_findex - start_findex)

    if use_mp and nworkers > 1:
        if executor is None:
            workers = nworkers or int(np.ceil(0.5 * os.cpu_count()))
            pool = ProcessPoolExecutor(workers)
        else:
            pool = executor
        if as_completed is None:
            as_completed = cf_as_completed
    else:
        pool = None

    exist_files = {}
    pending_jobs = []
    submitted = []
    perf_counter = time.perf_counter()
    for counter, (key, sub_config) in enumerate(input_tracker.items()):
        if check_exist:
            my_file = Path(f"{output}/{key}.pkl")
            if my_file.is_file():
                # logger.info(f"File exist {my_file}")
                exist_files[key] = str(my_file)
                continue
        if ext_pweight:
            job = (sub_config, ext_pweight, type)
        else:
            job = (sub_config, None, type)
        if use_mp and pool is not None:
            pending_jobs.append(pool.submit(_syst_process, job, **kwargs))
        else:
            pending_jobs.append(_syst_process(job, **kwargs))
        if counter % 100 == 0:
            t_diff = time.perf_counter() - perf_counter
            logger.info(f"submitted {counter} jobs [{t_diff:.2f}s]")
        submitted.append(sub_config)
    logger.info(f"total number of splited objects {counter}")
    logger.info(f"total number of submitted jobs {len(submitted)}")
    if exist_files:
        for _c in submitted:
            logger.info(f"submitted file -> {_c}")

    # start checking submitted jobs.
    perf_counter = time.perf_counter()
    n_pending = len(pending_jobs)
    output_jobs = []
    finish_tracker = {}
    if use_mp and pool is not None:
        pending_jobs = _parse_futures(as_completed(pending_jobs))
    with tqdm(total=n_pending, leave=False) as merged_pbar:
        merged_pbar.set_description("Retrieving jobs")
        for fname in pending_jobs:
            try:
                sub_finish_config = ConfigMgr.open(fname)
            except Exception as _err:
                raise IOError(f"Unable to open {fname}") from _err
            if isinstance(fname, str):
                os.unlink(fname)
            syst_name = sub_finish_config.name
            fname = sub_finish_config.save(f"{output}/{syst_name}")
            output_jobs.append(fname)
            finish_tracker[syst_name] = str(fname)
            merged_pbar.update()

    finish_tracker.update(exist_files)
    with open(f"{output}/finish_tracker.json", "w") as ofile:
        ofile.write(json.dumps(finish_tracker, indent=4))

    if not executor and pool:
        pool.shutdown()

    return output_jobs


# -----------------------------------------------------------------------------
def _run_HistMaker(config_mgr, ext_pweight, type=None, **process_kwargs):
    """
    Private HistMaker runner method
    """
    # check if config_mgr is a string or a ConfigMgr instance
    if isinstance(config_mgr, str):
        if ".py" in config_mgr:
            warnings.warn(
                "Using .py file as configuration input is deprecated. "
                "Please use .json or .yml file instead.",
                DeprecationWarning,
            )
            configGlobals, configLocals = {}, {}
            exec(open(config_mgr).read(), configGlobals, configLocals)
            config_mgr = None
            for k, v in configLocals.items():
                if isinstance(v, ConfigMgr):
                    config_mgr = v
                    break
        else:
            config_mgr = ConfigMgr.open(config_mgr)

    if not config_mgr.prepared:
        config_mgr.prepare()

    if type is None:
        hist_maker = HistMaker()
    elif type == 'bootstrap':
        hist_maker = strategies.bootstrap.BootstrapHistMaker()
    hist_maker.initialize()
    hist_maker.meta_data_from_config(config_mgr)
    hist_maker.process(config_mgr, ext_pweight=ext_pweight, **process_kwargs)
    hist_maker.finalise()

    config_mgr.filled = True

    return config_mgr


# -----------------------------------------------------------------------------


def _run_HistMaker_weight_gen(config_name, obs_name, weight_func_file, scale_factor):
    logger.info("Running run_HistMaker with weight gen")

    configGlobals, configLocals = {}, {}
    exec(open(config_name).read(), configGlobals, configLocals)

    my_configMgr = None
    for k, v in configLocals.items():
        if isinstance(v, ConfigMgr):
            my_configMgr = v
            break

    hist_maker = HistMaker()
    hist_maker.process_with_weight(
        my_configMgr, obs_name, weight_func_file, scale_factor
    )
    hist_maker.finalise()

    logger.info("after processing \n\n\n")

    # my_configMgr.print_config()

    return my_configMgr


# -----------------------------------------------------------------------------


def _process(configuration):
    input_config, external_weight, hist_maker_type = configuration

    if hist_maker_type == 'bootstrap':
        hist_maker = strategies.bootstrap.BootstrapHistMaker()
    else:
        hist_maker = HistMaker()

    config_mgr = ConfigMgr.open(input_config)

    if isinstance(input_config, (str, Path)):
        os.unlink(input_config)

    if not config_mgr.prepared:
        config_mgr.prepare(use_mp=False)

    hist_maker.initialize()
    hist_maker.meta_data_from_config(config_mgr)
    hist_maker.process(config_mgr, ext_pweight=external_weight)
    hist_maker.finalise()
    config_mgr.corrections.clear_buffer()

    return config_mgr.save(f"temp_output/{uuid4()}")


def _run_HistMaker_split_process(
    config,
    split_method=None,
    num_workers=None,
    external_pweight=None,
    executor=None,
    as_completed=None,
    submission_buffer_size=0,
    merge_buffer_size=0,
    multiprocessing_context="spawn",
    histmaker_type=None,
    **split_kwargs,
):
    """Run HistMaker with split configs."""
    logger.info(f"Running HistMaker with split configs using {split_method}")

    config_name = config.name
    num_workers = num_workers or int(np.ceil(0.5 * os.cpu_count()))
    merge_buffer_size = merge_buffer_size or num_workers * 2

    multiprocessing_context = mp.get_context(multiprocessing_context)
    pool = executor or ProcessPoolExecutor(
        num_workers, mp_context=multiprocessing_context
    )
    as_completed = as_completed or cf_as_completed

    unique_id = uuid4()
    start_time = time.perf_counter()

    pending_jobs = []
    finished_configs = []

    for job_index, sub_config in enumerate(
        ConfigMgr.split(config, split_method, **split_kwargs)
    ):  # split into small jobs
        sub_config.disable_pbar = True
        sub_config = sub_config.save(f"temp_input/{unique_id}_{job_index}")
        job = (sub_config, external_pweight, histmaker_type)
        pending_jobs.append(pool.submit(_process, job))
        if job_index % 100 == 0:
            logger.info(
                f"Submitted {job_index} jobs [{time.perf_counter() - start_time:.2f}s]"
            )
        if submission_buffer_size and job_index % submission_buffer_size != 0:
            continue
        if submission_buffer_size:
            for pending_job in as_completed(pending_jobs):
                file_name = pending_job.result()
                finished_configs.append(ConfigMgr.open(file_name))
                if len(finished_configs) != merge_buffer_size:
                    continue
                merged_config = ConfigMgr.merge(finished_configs, copy=False)
                finished_configs = [merged_config]
            pending_jobs = []

    logger.info(f"Total number of split objects {job_index}")

    # start merging process.
    start_time = time.perf_counter()
    num_pending_jobs = len(pending_jobs)
    with tqdm(total=num_pending_jobs, leave=False) as merge_progress_bar:
        for pending_job in as_completed(pending_jobs):
            file_name = pending_job.result()
            finished_configs.append(ConfigMgr.open(file_name))
            os.unlink(file_name)
            merge_progress_bar.update()
            del pending_job
            if len(finished_configs) != merge_buffer_size:
                continue
            merged_config = ConfigMgr.merge(finished_configs, copy=False)
            finished_configs = [merged_config]

    final_config = ConfigMgr.merge(finished_configs, copy=False)

    if not executor:
        pool.shutdown()

    logger.info(f"Finished merging {time.perf_counter()-start_time:.2f}s")
    final_config.name = config_name

    final_config.filled = True

    if split_method == "ifile":
        final_config.update_input_file_from_record()

    return final_config


# ==============================================================================
def _split_merge(jobs):
    from configMgr import ConfigMgr

    files = jobs[0]
    out_path = jobs[1]
    buffer = []
    for entry, f in enumerate(files):
        if "obj" in f:
            logger.info(f"prepared {round(100 * entry / len(files), 2)}")
            tmp_config = ConfigMgr()
            tmp_config.load(out_path + "/" + f)
            buffer.append(tmp_config)

    if buffer:
        logger.info("start merging")
        my_final_config = buffer[0]
        for entry, b in enumerate(buffer[1:]):
            logger.info(f"merged {round(100 * entry / len(buffer), 2)}")
            my_final_config += b
        my_final_config.addition_tag = "_merged_"
        return my_final_config
    else:
        return []


def merge_output(config_name, oname, split=True):
    my_configMgr = importlib.import_module(f"user_config.{config_name}").make_config()

    files = [file for file in os.listdir(my_configMgr.out_path)]
    if split:
        step = 400
        split_files = [[files[i::step], my_configMgr.out_path] for i in range(step)]
        with mp.Pool(10) as pool:
            result = pool.map(_split_merge, split_files)
        my_final_config = result[0]
        for entry, b in enumerate(result[1:]):
            logger.info(f"final merged {round(100 * entry / len(result), 2)}")
            my_final_config += b
        my_final_config.addition_tag = "_merged_"
    else:
        buffer = []
        for entry, f in enumerate(files):
            if "obj" in f:
                logger.info(f"prepared {round(100 * entry / len(files), 2)}")
                tmp_config = ConfigMgr()
                tmp_config.load(my_configMgr.out_path + "/" + f)
                buffer.append(tmp_config)

        logger.info("start merging")
        my_final_config = buffer[0]
        for entry, b in enumerate(buffer[1:]):
            logger.info(f"merged {round(100 * entry / len(buffer), 2)}")
            my_final_config += b
        my_final_config.addition_tag = "_merged_"

    # logger.info(result)
    my_final_config.save(oname)


# ===============================================================================
def refill_process(
    config,
    process_list,
    *,
    use_mp=True,
    systematic=None,
    mp_context="spawn",
    split_region=False,
    nworkers=None,
    **kwargs,
):
    histmaker = HistMaker()
    histmaker.meta_data_from_config(config)
    plevel_process = histmaker.plevel_process

    process = [config.get_process_set(p).get(systematic).copy() for p in process_list]

    # check if all processes has the given systematic
    # _syst_name = None
    # _syst_type = None
    _syst_full_name = systematic
    if _syst_full_name is not None:
        for p in process:
            if p.systematic is None:
                continue
            _syst_name = p.systematic.name
            _syst_type = p.systematic.sys_type
            assert _syst_full_name == p.systematic.full_name
            break
        else:
            _syst_name = _syst_full_name
            _syst_type = None

    # remove processes that are going to be refilled.
    for p in process:
        if _syst_full_name and p.systematic is None:
            p.systematic = SystematicBase(
                _syst_name, _syst_full_name, "dummy", _syst_type
            )
        config.remove_process(p.name, systematic=_syst_full_name)
        p.clear_content()

    branch_list = collections.defaultdict(set)
    # splitting process by region
    if split_region:
        ready_processes = []
        ready_processes_append = ready_processes.append
        for i_p in process:
            branch_list[i_p.name] |= i_p.ntuple_branches
            regions = i_p.list_regions()
            c_p = i_p.copy(shallow=True)
            c_p.clear()
            for _region in regions:
                m_region = i_p.pop_region(_region)
                branch_list[i_p.name] |= m_region.ntuple_branches
                m_p = c_p.copy()
                m_p.add_region(m_region)
                ready_processes_append(m_p)
    else:
        ready_processes = process

    result = []

    # non multiprocessing approach
    if not use_mp:
        for p in ready_processes:
            _nfiles = len(p.filename)
            kwargs.update({"branch_list": branch_list.get(p.name, None)})
            logger.info(f"refilling {p.name} from {_nfiles} files.")
            for i, f in enumerate(p.filename):
                histmaker.fill_file_status = f"{i+1}/{_nfiles} files"
                # pass in copy of p here to avoid refilling same process
                job = plevel_process(p.copy(), f, **kwargs)
                result.append(job)
        for p in result:
            config.append_process(p, mode="merge")
        # make sure corrections buffer is clear
        config.corrections.clear_buffer()
        return

    # multiprocessing approach
    if nworkers is None:
        nworkers = min(len(ready_processes), os.cpu_count())
    logger.info(f"create {nworkers} workers.")
    mp_context = mp.get_context(mp_context)
    with ProcessPoolExecutor(nworkers, mp_context=mp_context) as pool:
        for p in ready_processes:
            kwargs.update({"branch_list": branch_list.get(p.name, None)})
            for f in p.filename:
                logger.info(f"submit {p.name} with {f}")
                result.append(pool.submit(plevel_process, p, f, **kwargs))
        for p in cf_as_completed(result):
            config.append_process(p.result(), mode="merge")
    # make sure corrections buffer is clear
    config.corrections.clear_buffer()


# ==============================================================================
def _good_file_list(config):
    good_list = {}
    config = ConfigMgr.open(config)
    if not config.prepared:
        config.prepare(use_mp=False)
    filter_missing_ttree(config, use_mp=False)
    for pset in config.process_sets:
        for p in pset:
            key = (p.name, p.systematic)
            good_list[str(key)] = list(p.filename)
    return good_list


def _prepare_file_list(config):
    good_list = {}
    config = ConfigMgr.open(config)
    if not config.prepared:
        config.prepare(use_mp=False)
    for pset in config.process_sets:
        for p in pset:
            key = (p.name, p.systematic)
            good_list[str(key)] = f"{config.out_path}/{config.ofilename}.pkl"
    return good_list


def generate_good_files_list(
    input_tracker,
    output="process_GFL.json",
    use_mp=True,
    mp_handler=None,
):
    logger.info("Generating good file list with systematics.")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    if use_mp:
        if mp_handler:
            executor, as_completed = mp_handler
        else:
            workers = int(np.ceil(0.5 * os.cpu_count()))
            executor = ProcessPoolExecutor(workers)
            as_completed = cf_as_completed
    else:
        executor = None
        as_completed = None

    with open(input_tracker) as ifile:
        input_tracker = json.load(ifile)

    pending_futures = []
    for key, sub_config in input_tracker.items():
        if use_mp:
            future = executor.submit(_prepare_file_list, sub_config)
        else:
            future = _prepare_file_list(sub_config)
        pending_futures.append(future)

    prepared_list = {}
    if use_mp:
        for future in as_completed(pending_futures):
            prepared_list.update(future.result())
    else:
        for future in pending_futures:
            prepared_list.update(future)

    # real thing here
    pending_futures = []
    for key, sub_config in prepared_list.items():
        logger.info(f"submitting {key}")
        if use_mp:
            future = executor.submit(_good_file_list, sub_config)
        else:
            future = _good_file_list(sub_config)
        pending_futures.append(future)

    good_list = {}
    if use_mp:
        for future in as_completed(pending_futures):
            good_list.update(future.result())
    else:
        for future in pending_futures:
            good_list.update(future)

    with open(output, "w") as ofile:
        ofile.write(json.dumps(good_list, indent=4))

    if use_mp and not mp_handler:
        executor.shutdown()


# ==============================================================================
def _good_file_list_2(tfile_list, xsec_sumw, unique_processes, branch_rename=None):
    histmaker = HistMaker(xsec_sumw=xsec_sumw)
    histmaker.RAISE_TREENAME_ERROR = True
    check_ttree = histmaker.process_ttree_resolve
    open_root_file = histmaker.raw_open
    histmaker.branch_rename = branch_rename
    good_list = {}
    for f in tfile_list:
        with open_root_file(f) as opened_f:
            for key, p in unique_processes.items():
                if key not in good_list:
                    good_list[key] = []
                try:
                    ttree, *_ = check_ttree(opened_f, p)
                except RuntimeError:
                    continue
                if ttree is None or ttree.num_entries == 0:
                    continue
                good_list[key].append(f)
    return good_list


def _config_batch(configs):
    unique_processes = {}
    for config in configs:
        config = ConfigMgr.open(config)
        processes = (y for x in config.process_sets for y in x)
        for p in processes:
            key = str((p.name, p.systematic))
            if key not in unique_processes:
                unique_processes[key] = p
    return unique_processes


def generate_good_files_list_2(
    configs,
    tfile_list,
    xsec_sumw=None,
    output="process_GFL.json",
    batch_size=20,
    config_batch_size=20,
    nworkers=20,
    executor=None,
    as_completed=None,
    branch_rename=None,
):
    if config_batch_size:
        client = executor or ProcessPoolExecutor(nworkers)
        as_completed = as_completed or cf_as_completed
        futures = []
        for i in range(0, len(configs), config_batch_size):
            futures.append(
                client.submit(_config_batch, configs[i : i + config_batch_size])
            )
        unique_processes = {}
        for future in tqdm(as_completed(futures), total=len(futures), leave=False):
            unique_processes.update(future.result())
        if executor:
            executor.shutdown()
    else:
        unique_processes = _config_batch(tqdm(configs))

    if batch_size:
        futures = []
        nfiles = len(tfile_list)
        client = executor or ProcessPoolExecutor(nworkers)
        as_completed = as_completed or cf_as_completed
        for i in range(0, nfiles, batch_size):
            sub_files = tfile_list[i : i + batch_size]
            future = client.submit(
                _good_file_list_2, sub_files, xsec_sumw, unique_processes, branch_rename
            )
            futures.append(future)
        good_list = None
        for future in tqdm(as_completed(futures), total=len(futures)):
            if good_list is None:
                good_list = future.result()
                continue
            fdata = future.result()
            for key, value in fdata.items():
                if key not in good_list:
                    good_list[key] = value
                else:
                    good_list[key] += value
        if executor:
            executor.shutdown()
    else:
        good_list = _good_file_list_2(
            tqdm(tfile_list), xsec_sumw, unique_processes, branch_rename
        )
    with open(output, "w") as ofile:
        ofile.write(json.dumps(good_list, indent=4))
