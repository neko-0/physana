import json
import os
import shutil
from glob import glob
import pathlib
import logging
import socket
from functools import partial
from typing import Union, Dict, List, Any, Callable, Optional, Tuple
from collections import defaultdict

from tqdm import tqdm
from dask_jobqueue import HTCondorCluster
from dask.distributed import Client, Future, LocalCluster
from dask.distributed import as_completed as dask_as_completed

from .dataset_group import get_ntuple_files, get_nfiles

from ..configs.base import ConfigMgr
from ..configs.dispatch_tools import split
from ..configs.merge_tools import merge
from ..algorithm import run_algorithm, HistMaker
from ..tools import extract_cutbook_sum_weights

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

red = "\033[91m"
green = "\033[92m"
yellow = "\033[93m"
reset = "\033[0m"


class JSONHistSetup:
    def __init__(self, json_path: str):
        self._show_job_script: bool = True

        self.weights: List[str] = []
        self.selections: List[str] = []
        self.define_selections: Dict[str, str] = {}
        self.process_list: List[Dict[str, str]] = []
        self.region_list: List[Dict[str, str]] = []
        self.observable_list: List[Dict[str, str]] = []
        self.observable2D_list: List[Dict[str, str]] = []

        with open(json_path) as f:
            self.setting: dict = json.load(f)

    def initialize(self) -> None:
        self.parse_block_io()
        self.parse_block_config()
        self.parse_block_jobs()
        self.parse_block_condor()
        self.parse_block_others()

    def parse_block_io(self) -> None:
        block_ios = self.setting["InputOutput"]
        self.out_path = block_ios["output_directory"]

    def parse_block_config(self) -> None:
        block_config = self.setting["ConfigMgr"]
        self.weights = block_config.get("weights", [])
        self.selections = block_config.get("common_selections", [])
        self.define_selections = block_config.get("define_selections", {})
        self.process_list = block_config.get("processes", [])
        self.region_list = block_config.get("regions", [])
        self.observable_list = block_config.get("observables", [])
        self.observable2D_list = block_config.get("observables2D", [])

    def parse_block_jobs(self) -> None:
        block_jobs = self.setting["Jobs"]
        self.jobs = block_jobs

    def parse_block_condor(self) -> None:
        block_condor = self.setting["Condor"]
        self.condor = block_condor

    def parse_block_others(self) -> None:
        block_others = self.setting["Others"]
        self.others = block_others

    def setup_cluster(self) -> HTCondorCluster:
        n_port = 8786
        self.condor.setdefault(
            "scheduler_options", {'port': n_port, 'host': socket.gethostname()}
        )
        self.condor.setdefault("worker_extra_args", ['--worker-port 10000:10100'])
        cluster = HTCondorCluster(**self.condor)

        if self._show_job_script:
            self._show_job_script = False
            logger.info("Job script:\n" + cluster.job_script())

        return cluster

    def launch(self) -> None:
        CURRENT_ATTEMPT = 0
        MAX_ATTEMPTS = self.others["max_attempts"]

        use_single_job = self.others.get("single_job", False)
        if use_single_job:
            dispatch_tool = single_thread_job_dispatch
        else:
            dispatch_tool = job_dispatch

        failed = dispatch_tool(self)

        if any(failed.values()):
            if CURRENT_ATTEMPT < MAX_ATTEMPTS:
                CURRENT_ATTEMPT += 1
                logger.warning(f"Attempt {CURRENT_ATTEMPT} failed. Retrying...")
                failed = dispatch_tool(self)
            else:
                for name, job_failed in failed.items():
                    logger.warning(
                        f"{name} status: {f'{red} failed {reset}' if job_failed else f'{green} succeeded {reset}'}"
                    )
                logger.critical(f"All attempts {red}failed{reset}. Exiting...")
        else:
            logger.info(f"All attempts {green}succeeded{reset}. Exiting...")


def combine_json_setups(json_files: Union[str, List[str]]) -> 'JSONHistSetup':
    """
    Combine multiple JSONHistSetup objects from different JSON files.

    Parameters
    ----------
    json_files : str or list of str
        A path to JSON file that contains a list of paths to other JSON files.
        Each of those files contains part of JSONHistSetup object.
        Alternatively, a list of paths to JSON files can be provided directly.

    Returns
    -------
    JSONHistSetup
        A new JSONHistSetup object with the settings from all the input files.
    """

    if isinstance(json_files, str):
        with open(json_files) as input_file:
            file_paths = json.load(input_file)
    elif isinstance(json_files, list):
        file_paths = json_files
    else:
        raise ValueError(f"Files setting receives invalid type {type(json_files)}.")

    setups = [JSONHistSetup(path) for path in file_paths]

    combined = setups[0]
    for setup in setups[1:]:
        for group in ["InputOutput", "ConfigMgr", "Jobs", "Condor", "Others"]:
            combined.setting[group] = {
                **combined.setting.get(group, {}),
                **setup.setting.get(group, {}),  # overwrite with new settings
            }

    return combined


def fill_config(
    config_name: str, output_dir: str, entry_range: Optional[Tuple[int, int]] = None
) -> Union[str, None]:
    """
    Opens a configuration file, processes it using HistMaker, and saves the result.

    Parameters
    ----------
    config_name : str
        The path to the configuration file to be opened and processed.
    output_dir : str
        The directory where the processed configuration will be saved.

    Returns
    -------
    str or None
        The path to the saved processed configuration, or None if an error occurs.
    """
    try:
        sub_config = ConfigMgr.open(config_name)
    except OSError:
        try:
            os.unlink(config_name)
        except Exception as _err:
            logger.warning(f"Unable to remove {config_name} due to {_err}")
        return None

    assert len(sub_config.process_sets) == 1

    for pset in sub_config.process_sets:
        for p in pset:
            assert len(p.input_files) <= 1
            if len(p.input_files) == 1:
                ifile = pathlib.Path(list(p.input_files)[0]).stem.strip(".root")
            else:
                ifile = None

    if entry_range:
        start, end = entry_range
    else:
        start, end = None, None

    # Setting up output file name.
    output = f"{output_dir}/output_{pset.name}_{ifile}"
    if start is None and end is None:
        output += ".pkl"
    else:
        output += f"_{start}_{end}.pkl"

    if pathlib.Path(output).exists():
        return output

    sub_config.disable_pbar = True

    histmaker = HistMaker(use_mmap=False)
    histmaker.use_threads = False
    histmaker.step_size = "5MB"
    histmaker.nthread = 1
    histmaker.disable_pbar = True

    run_algorithm(sub_config, histmaker, entry_start=start, entry_stop=end)

    return sub_config.save(output)


def batch_runner(prepared_jobs_list: List[Callable[[], None]]) -> List[None]:
    """
    Runs a list of jobs in batch.

    Parameters
    ----------
    prepared_jobs_list : List[Callable[[], None]]
        A list of jobs to be run in batch.

    Returns
    -------
    List[None]
        A list of the results of the jobs. Each job is expected to return None.
    """
    return [job() for job in prepared_jobs_list]


def generate_config(
    json_config: JSONHistSetup, src_path: str, lumi: float, mc_campaign: str
) -> ConfigMgr:
    """Generate a ConfigMgr object based on the given configuration and input files.

    Parameters
    ----------
    json_config : JSONHistSetup
        The configuration object
    src_path : str
        The path to the input files
    lumi : float
        The luminosity to be used
    mc_campaign : str
        The MC campaign to be used

    Returns
    -------
    ConfigMgr
        The generated configuration object
    """
    ntuple_files = glob(src_path)
    process_file_map = get_ntuple_files(ntuple_files)[mc_campaign]

    weights = "*".join(json_config.weights)

    process_selection = "&&".join(json_config.selections)

    setting = ConfigMgr("", description="Zjj polarization")

    # Define processes
    # ===================================================================
    for process in json_config.process_list:
        process = process.copy()  # make a copy to avoid modifying the original
        name = process["name"]
        max_files = process.pop("max_files", None)
        input_files = process_file_map[name]
        if max_files:
            input_files = get_nfiles(name, input_files, max_files)
        process.setdefault("input_files", input_files)
        process.setdefault("selection", process_selection)
        process.setdefault("treename", "reco")
        process.setdefault("weights", lumi)
        setting.add_process(**process)

    # Define regions
    # ===================================================================
    for region in json_config.region_list:
        region = region.copy()  # make a copy to avoid modifying the original
        region.setdefault("weights", weights)
        selection_name = region["selection"]
        selection_list = [json_config.define_selections[x] for x in selection_name]
        region["selection"] = "&&".join(selection_list)
        setting.add_region(**region)

    # Define observables
    # ===================================================================
    for observable in json_config.observable_list:
        setting.add_observable(**observable)

    for observable in json_config.observable2D_list:
        setting.add_observable2D(**observable)

    setting.RAISE_TREENAME_ERROR = False

    setting.xsec_file = json_config.others["xsec_file"]

    return setting


def preparing_jobs(
    json_config: JSONHistSetup,
) -> Dict[str, List[Callable[[], Union[str, None]]]]:
    """
    Prepare jobs for processing.

    Parameters
    ----------
    json_config : JSONHistSetup
        The configuration object.

    Returns
    -------
    Dict[str, List[Callable[[], None]]]
        A dictionary of prepared jobs.
    """
    jobs = json_config.jobs
    # Check if there is already a merged output
    if json_config.others.get("check_output", True):
        for output_name in list(jobs):
            output_filename = pathlib.Path(f"{json_config.out_path}/{output_name}.pkl")
            if output_filename.exists():
                jobs.pop(output_name)

    prepared_jobs: Dict[str, List[Callable[[], Union[str, None]]]] = defaultdict(list)
    for name, job in jobs.items():
        setting = generate_config(json_config, job[0], job[1], job[2])

        sum_weights_file = f"{json_config.out_path}/{name}_SumWeights.txt"
        if pathlib.Path(sum_weights_file).exists():
            setting.sum_weights_file = sum_weights_file
        else:
            setting.sum_weights_file = extract_cutbook_sum_weights(
                setting, sum_weights_file, treename_match=None
            )

        enable_cache_split = json_config.others.get("cache_split", False)
        enable_prepare = json_config.others.get("prepare", False)

        cache_split_file = f"{json_config.out_path}/{name}_cache_split.json"
        if not pathlib.Path(cache_split_file).exists():
            enable_cache_split = False

        if enable_cache_split:
            with open(cache_split_file) as f:
                split_config = json.load(f)
        else:
            split_config = split(
                setting,
                "entries",
                nbatch=json_config.others.get("nbatch", 5),
                tree_name="reco",
            )

        logger.info(f"Start preparing {name}")

        cache_split_keeper = []
        for sub_config, start, end in tqdm(split_config, leave=False):
            if not enable_cache_split:
                # Make sure there is only one input file per process
                for pset in sub_config.process_sets:
                    for p in pset:
                        assert len(p.input_files) <= 1
                        if len(p.input_files) == 1:
                            ifile = pathlib.Path(list(p.input_files)[0]).stem.strip(
                                ".root"
                            )
                        else:
                            ifile = None
                        ifile = f"{pset.name}_{ifile}"

                config_name = f"{json_config.out_path}/input_{name}/input_{ifile}_{start}_{end}.pkl"
                if not pathlib.Path(config_name).exists():
                    if json_config.others.get("prepare_prior_save", False):
                        sub_config.prepare()
                    config_name = sub_config.save(config_name)
                cache_split_keeper.append([str(config_name), start, end])
            else:
                config_name = sub_config

            if enable_prepare:
                config_name = ConfigMgr.open(config_name)
                config_name.prepare()

            prepared_jobs[name].append(
                partial(
                    fill_config,
                    config_name,
                    f"{json_config.out_path}/output_{name}/",
                    (start, end),
                )
            )

        if not enable_cache_split:
            with open(cache_split_file, "w") as f:
                json.dump(cache_split_keeper, f)

    return prepared_jobs


def single_thread_job_dispatch(json_config: JSONHistSetup) -> Dict[str, bool]:

    prepared_jobs = preparing_jobs(json_config)

    if not prepared_jobs:
        logger.warning(f"{yellow} All jobs are merged {reset}")
        return {x: False for x in prepared_jobs.keys()}

    batch_size = json_config.others.get("file_batch_size", 3)

    failed = {x: False for x in prepared_jobs}

    results = defaultdict(list)
    for name, job_list in prepared_jobs.items():
        results_list = results[name]
        for i in range(0, len(job_list), batch_size):
            results_list += batch_runner(job_list[i : i + batch_size])

    for name, result in results.items():
        logger.info(f"{name} merging jobs: {len(result)}")
        try:
            config = merge((ConfigMgr.open(x) for x in results[name]))
            config.save(f"{json_config.out_path}/{name}.pkl")
            if json_config.others.get("cleanup_input", True):
                shutil.rmtree(f"{json_config.out_path}/input_{name}")
        except OSError:
            for x in results[name]:
                if x is None:
                    continue
                try:
                    _ = ConfigMgr.open(x)
                except OSError:
                    os.unlink(x)
            failed[name] = True

    return failed


def job_dispatch(json_config: JSONHistSetup) -> Dict[str, bool]:
    """
    Dispatch jobs for processing based on the given JSON configuration.

    Parameters
    ----------
    json_config : JSONHistSetup
        The configuration object containing job settings.

    Returns
    -------
    Dict[str, bool]
        A dictionary indicating the failure status of each job.
    """
    prepared_jobs = preparing_jobs(json_config)

    if not prepared_jobs:
        logger.warning(f"{yellow} All jobs are merged {reset}")
        return {x: False for x in prepared_jobs.keys()}

    max_jobs = json_config.others["max_num_jobs"]
    jobs_size = [len(x) for x in prepared_jobs.values()]
    if max_jobs < 1:
        max_jobs = max(jobs_size)
    else:
        max_jobs = min(max_jobs, min(jobs_size))
    logger.info(f"Scaling jobs to {max_jobs}")

    if json_config.others["local"]:
        cluster = LocalCluster(n_workers=max_jobs)
    else:
        cluster = json_config.setup_cluster()
        cluster.scale(jobs=max_jobs)

    batch_size = json_config.others.get("file_batch_size", 3)
    timeout = json_config.others.get("timeout", 60 * 5)
    failed = {x: False for x in prepared_jobs}
    futures: Dict[str, List[Future]] = defaultdict(list)
    results: Dict[str, List[Any]] = defaultdict(list)

    with cluster, Client(cluster) as pool:
        for name, job_list in prepared_jobs.items():
            futures_name_append = futures[name].append
            for i in range(0, len(job_list), batch_size):
                futures_name_append(
                    pool.submit(batch_runner, job_list[i : i + batch_size])
                )

        for name, futures_list in futures.items():
            total_batches = len(futures_list)
            retry_count = 0
            max_retries = 3
            finished_num_batch = 0
            while retry_count < max_retries:
                try:
                    result_fail = False
                    for i, future in enumerate(
                        dask_as_completed(futures_list, timeout=timeout), start=1
                    ):
                        try:
                            results[name] += future.result()
                        except Exception:
                            result_fail = True
                            continue
                        if i >= finished_num_batch:
                            percent = round(100 * i / total_batches, 2)
                            logger.info(
                                f"{name} completed batch {i}/{total_batches}, {percent}%"
                            )
                            finished_num_batch = i
                    failed[name] = result_fail
                    break
                except TimeoutError:
                    logger.warning(f"{name} timed out. Retrying...")
                    retry_count += 1
                    results[name] = []
                    if retry_count == max_retries:
                        logger.critical(f"{name} failed after {max_retries} retries")
                        failed[name] = True
                        break
                except Exception as e:
                    logger.critical(f"{name} failed with exception {e}")
                    logger.critical(f"{name} failed after {retry_count} retries")
                    failed[name] = True
                    results[name] = []
                    break

    if not any(failed.values()):
        for name, result in results.items():
            logger.info(f"{name} merging jobs: {len(result)}")
            try:
                config = merge((ConfigMgr.open(x) for x in results[name]))
                config.save(f"{json_config.out_path}/{name}.pkl")
                if json_config.others.get("cleanup_input", True):
                    shutil.rmtree(f"{json_config.out_path}/input_{name}")
            except OSError:
                for x in results[name]:
                    if x is None:
                        continue
                    try:
                        _ = ConfigMgr.open(x)
                    except OSError:
                        os.unlink(x)
                failed[name] = True

    return failed
