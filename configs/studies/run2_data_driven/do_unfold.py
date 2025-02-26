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

def single_thread(ifile, ofile):
    # run2_with_fakes = ConfigMgr.open(f"./output/unfoldTest_v3_fullrun2_sys/fakes.pkl")
    run2_with_fakes = ConfigMgr.open(ifile)

    #dijets = run2_with_fakes.get_process("dijets")
    #dijets.name = "fakes"
    #dijets.process_type = "fakes"
    #run2_with_fakes.remove_process_set("dijets")
    #run2_with_fakes.append_process(dijets)

    make_debug_plots = False
    kwargs = {}
    if make_debug_plots:
        kwargs = {'systematic_names': [None]}
    unfold = unfolding.run_auto(
        run2_with_fakes, lumi=138.861, output_folder="unfolding", debug=make_debug_plots, **kwargs
    )
    # lumi = 138.861
    # unfold = unfolding.run(run2_with_fakes, 'wjets', 'wjets', 'wjets', 'closure')
    if not make_debug_plots:
        unfold.save(ofile)
    else:
        unfold.save(f'debug_{ofile}.pkl')

def wrap_run_auto(fname, **kwargs):
    m_config = ConfigMgr.open(fname)
    return fname, unfolding.run_auto(m_config, include_fakes=False, **kwargs)

def multi_thread(ifile, ofile):
    run2_with_fakes = ConfigMgr.open(ifile)
    syslist = run2_with_fakes.list_systematic_full_name()

    workers = min(150, len(syslist))
    cores = 16
    account = "shared"
    cluster = SLURMCluster(
        queue=account,
        walltime='05:00:00',
        project="collinear_Wjets",
        nanny=False,
        cores=1,
        processes=1,
        memory="128GB",
        job_extra=[f'--account={account}', f'--partition={account}', f'--cpus-per-task={cores}'],
        local_directory="dask_unfold_output",
        log_directory="dask_unfold_logs",
        n_workers=workers,
        death_timeout=36000,
    )
    client = Client(cluster)
    client.get_versions(check=True)

    make_debug_plots = False

    merge_size = 20

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
                lumi=138.861,#36.207,
                output_folder="unfolding",
                debug=make_debug_plots,
                **kwargs,
            )
            futures.append(unfold)
            print(f"submitted {syst}")
            fileID += 1

        counter = 0
        start_t = time.perf_counter()
        finished_files = []
        for future in dask_as_completed(futures):
            fname, result = future.result()
            buffer.append(result)
            future.release()
            finished_files.append(fname)
            counter += 1
            if counter % merge_size == 0:
                print(f"done {counter}/{len(syslist)} {time.perf_counter()-start_t}s")
                start_t = time.perf_counter()
                merged_config = ConfigMgr.intersection(buffer, copy=False)
                buffer = [merged_config]
                for _i, _f in enumerate(finished_files):
                    os.unlink(_f)
                    del finished_files[_i]

        for _f in finished_files:
            os.unlink(_f)


    unfold = ConfigMgr.intersection(buffer, copy=False)

    unfold.save(ofile)

if __name__ == "__main__":
    input_f = "prune_run2_2211.pkl"
    config = ConfigMgr.open(input_f)
    dijets_var = ConfigMgr.open("dijets_variation.pkl")
    dijets = config.get_process_set("dijets")
    dijets.nominal = None # set to None
    config.add(dijets_var)
    breakpoint()
    input_f = config.save("prune_run2_2211_dijets_corr_var.pkl")
    output_f = "unfold_run2_2211_dijets_corr_var.pkl"
    do_multi_thread = True
    if not do_multi_thread:
        single_thread(input_f, output_f)
    else:
        multi_thread(input_f, output_f)
