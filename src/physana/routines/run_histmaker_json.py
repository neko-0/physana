import json
import os
import shutil
from glob import glob
import pathlib
import logging
import socket
from functools import partial
from typing import Union, Dict, List, Any, Callable
from collections import defaultdict

from tqdm import tqdm
from dask_jobqueue import HTCondorCluster
from dask.distributed import Client, Future, LocalCluster
from dask.distributed import as_completed as dask_as_completed

from .dataset_group import get_ntuple_files

from ..configs.base import ConfigMgr
from ..configs.dispatch_tools import split
from ..configs.merge_tools import merge
from ..algorithm import run_algorithm, extract_cutbook_sum_weights, HistMaker

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

red = "\033[91m"
green = "\033[92m"
yellow = "\033[93m"
reset = "\033[0m"


class JSONHistSetup:
    def __init__(self, json_path: str):
        self._show_job_script = True
        with open(json_path) as f:
            self.setting = json.load(f)

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

        failed = job_dispatch(self)
        if any(failed.values()):
            if CURRENT_ATTEMPT < MAX_ATTEMPTS:
                CURRENT_ATTEMPT += 1
                logger.warning(f"Attempt {CURRENT_ATTEMPT} failed. Retrying...")
                failed = job_dispatch(self)
            else:
                for name, job_failed in failed.items():
                    logger.warning(
                        f"{name} status: {f'{red} failed {reset}' if job_failed else f'{green} succeeded {reset}'}"
                    )
                logger.critical(f"All attempts {red}failed{reset}. Exiting...")
        else:
            logger.info(f"All attempts {green}succeeded{reset}. Exiting...")


def fill_config(config_name: str, output_dir: str) -> Union[str, None]:
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

    output = f"{output_dir}/input_{pset.name}_{ifile}.pkl"
    if pathlib.Path(output).exists():
        return output

    histmaker = HistMaker(use_mmap=False)
    histmaker.step_size = "32MB"
    histmaker.nthread = 1
    histmaker.disable_pbar = True
    sub_config.disable_pbar = True
    run_algorithm(sub_config, histmaker)

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
        process.setdefault("input_files", process_file_map[name])
        process.setdefault("selection", process_selection)
        process.setdefault("treename", "reco")
        setting.add_process(**process)

    # Define regions
    # ===================================================================
    for region in json_config.region_list:
        region = region.copy()  # make a copy to avoid modifying the original
        region.setdefault("weight", weights)
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

    setting.xsec_file = "/cvmfs/atlas.cern.ch/repo/sw/database/GroupData/dev/PMGTools/PMGxsecDB_mc16.txt"

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
    # Check merged output
    for output_name in list(jobs):
        output_filename = pathlib.Path(f"{json_config.out_path}/{output_name}.pkl")
        if output_filename.exists():
            jobs.pop(output_name)

    prepared_jobs: Dict[str, List[Callable[[], Union[str, None]]]] = defaultdict(list)
    for name, job in jobs.items():
        setting = generate_config(json_config, job[0], job[1], job[2])

        sum_weight_file = f"{json_config.out_path}/{name}_SumWeights.txt"
        if pathlib.Path(sum_weight_file).exists():
            setting.sum_weights_file = sum_weight_file
        else:
            setting.sum_weights_file = extract_cutbook_sum_weights(
                setting, sum_weight_file
            )

        split_config = split(setting, "ifile", batch_size=1)

        logger.info(f"Starting preparing {name}")

        for i, sub_config in tqdm(enumerate(split_config), leave=False):
            config_name = f"{json_config.out_path}/input_{name}/{i}.pkl"
            if not pathlib.Path(config_name).exists():
                config_name = sub_config.save(config_name)
            prepared_jobs[name].append(
                partial(fill_config, config_name, f"{json_config.out_path}/tmp_{name}/")
            )

    return prepared_jobs


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
        cluster = LocalCluster()
        cluster.scale(max_jobs)
    else:
        cluster = json_config.setup_cluster()
        cluster.scale(jobs=max_jobs)

    batch_size = 3
    failed = {x: False for x in prepared_jobs}
    futures: Dict[str, List[Future]] = defaultdict(list)
    results: Dict[str, List[Any]] = defaultdict(list)

    with Client(cluster) as pool:
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
                    for i, future in enumerate(
                        dask_as_completed(futures_list, timeout=60 * 5), start=1
                    ):
                        results[name] += future.result()
                        if i >= finished_num_batch:
                            logger.info(f"{name} completed batch {i}/{total_batches}")
                            finished_num_batch = i
                    failed[name] = False
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

    cluster.close()

    if not any(failed.values()):
        for name, result in results.items():
            logger.info(f"{name} merging jobs: {len(result)}")
            try:
                config = merge((ConfigMgr.open(x) for x in results[name]))
                config.save(f"{json_config.out_path}/{name}.pkl")
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
