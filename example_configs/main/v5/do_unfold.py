from collinearw import ConfigMgr
from collinearw.strategies import unfolding
import time
import os
import dask
from dask_jobqueue import SLURMCluster
from dask.distributed import Client
from dask.distributed import as_completed as dask_as_completed
import pathlib

import logging

logging.basicConfig()
logging.getLogger().setLevel(logging.CRITICAL)

do_multi_thread = True


def single_thread():
    # run2_with_fakes = ConfigMgr.open(f"./output/unfoldTest_v3_fullrun2_sys/fakes.pkl")
    run2_with_fakes = ConfigMgr.open("fakes_2211.pkl")

    make_debug_plots = False
    kwargs = {}
    if make_debug_plots:
        kwargs = {'systematic_names': None}
    unfold = unfolding.run_auto(
        run2_with_fakes,
        lumi=138.861,
        output_folder="unfolding",
        debug=make_debug_plots,
        **kwargs,
    )
    # lumi = 138.861
    # unfold = unfolding.run(run2_with_fakes, 'wjets', 'wjets', 'wjets', 'closure')
    if not make_debug_plots:
        unfold.save('unfold.pkl')


def wrap_run_auto(fname, **kwargs):
    m_config = ConfigMgr.open(fname)
    return fname, unfolding.run_auto(m_config, **kwargs)


def multi_thread():
    run2_with_fakes = ConfigMgr.open("fakes_2211.pkl")
    syslist = run2_with_fakes.list_systematic_full_name()

    workers = min(100, len(syslist))
    cores = 8
    account = "shared"
    cluster = SLURMCluster(
        queue=account,
        walltime='01:00:00',
        project="collinear_Wjets",
        # nanny=False,
        cores=1,
        processes=1,
        memory="64 GB",
        job_extra=[
            f'--account={account}',
            f'--partition={account}',
            f'--cpus-per-task={cores}',
        ],
        local_directory="dask_unfold_output",
        log_directory="dask_unfold_logs",
        n_workers=workers,
        death_timeout=36000,
    )
    client = Client(cluster)
    client.get_versions(check=True)

    make_debug_plots = False

    fileID = 0
    buffer = []
    futures = []
    with cluster, Client(cluster) as client:
        client.get_versions(check=True)
        for syst in syslist:
            m_config = ConfigMgr.split(
                run2_with_fakes,
                split_type="systematic",
                syst_names=[syst],
                include_nominal=True,
            )
            m_config = next(m_config)
            fname = m_config.save(f"temp_output/unfold{fileID}")
            kwargs = {'systematic_names': [syst]}
            unfold = client.submit(
                wrap_run_auto,
                fname,
                lumi=138.861,
                output_folder="unfolding",
                debug=make_debug_plots,
                **kwargs,
            )
            futures.append(unfold)
            fileID += 1

        counter = 0
        start_t = time.time()
        for future in dask_as_completed(futures):
            fname, result = future.result()
            buffer.append(result)
            future.release()
            os.unlink(fname)
            counter += 1
            if counter % workers == 0:
                print(f"done {counter}/{len(syslist)} {time.time()-start_t}s")
                start_t = time.time()
                merged_config = ConfigMgr.intersection(buffer, copy=False)
                buffer = [merged_config]

    unfold = ConfigMgr.intersection(buffer, copy=False)

    unfold.save("unfold.pkl")


if __name__ == "__main__":
    if not do_multi_thread:
        single_thread()
    else:
        multi_thread()
