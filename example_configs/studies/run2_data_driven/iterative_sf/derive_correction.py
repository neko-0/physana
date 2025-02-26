from collinearw import ConfigMgr
from collinearw.serialization import Serialization
from collinearw.histMaker import weight_from_hist
from collinearw.strategies.abcd import unpack_correlation_correction
import collinearw.histManipulate
import concurrent.futures
import dask
import functools
from dask_jobqueue import SLURMCluster
from dask.distributed import Client, as_completed, LocalCluster, secede, get_client
import tqdm
import os
import time


tf_param = {
    "*muon*" : (("Ht30-jet1Pt"), "Ht30-jet1Pt"),
    "*electron*": (("abs(lep1Eta)", "lep1Pt"), "tf_eta_vs_lepPt_el"),
}

def wrap_run_iterative_correction(id, config, *args, **kwargs):
    data = collinearw.histManipulate.run_iterative_correction(config, *args, **kwargs)
    m_serial = Serialization()
    ofile = f"temp_output/_temp_data_{id}.pkl"
    m_serial.to_pickle(data, ofile)
    return ofile, config

def dask_version(
    signal="wjets",
    bkgd=["zjets", "ttbar", "singletop", "diboson"],
    skip=["wjets_mg", "wjets_FxFx", "wjets_powheg"],
    el_cr={"ttbar": "ttbarCR_Ele", "zjets": "ZjetsCR_Ele"},
    mu_cr={"ttbar": "ttbarCR_Mu", "zjets": "ZjetsCR_Mu"},
):
    #dask.config.set({"distributed.worker.daemon ": False})

    run2 = ConfigMgr.open("run2.pkl")
    syslist = run2.list_systematic_full_name()

    njobs = 250
    cores = 12
    account = "shared"
    cluster = SLURMCluster(
        queue=account,
        walltime='10:00:00',
        project="collinear_Wjets",
        #cores=100,
        #processes=1,
        nanny=False,
        memory="64GB",
        cores=1,
        processes=1,
        #memory="200 GB",
        job_extra=[f'--account={account}', f'--partition={account}', f'--cpus-per-task={cores}'],
        local_directory="dask_iterative_output",
        log_directory="dask_iterative_logs",
        # death_timeout=36000,
        #n_workers=1,
    )
    cluster.scale(jobs=njobs)
    print(cluster.job_script())
    # cluster = LocalCluster()

    pbar = tqdm.tqdm(total=len(syslist), dynamic_ncols=True, leave=False, position=3)

    output_bkgd = {}
    output_signal = {}
    futures = []
    fileID = 0
    file_buffer = []
    with cluster, Client(cluster) as client:
        client.get_versions(check=True)
        for syst in syslist:
            #print(f"submitting systematic {syst}")
            m_config = ConfigMgr.split(run2, split_type="systematic", syst_names=[syst])
            m_config = next(m_config)
            m_config = m_config.save(f"temp_output/_ID{fileID}.pkl")
            fileID += 1
            file_buffer.append(m_config)
            # print(f"submit {syst}")
            # client = get_client()
            future = client.submit(
                wrap_run_iterative_correction,
                fileID,
                m_config,
                signal=signal,
                skip=skip,
                bkgd=bkgd,
                iteration=3,
                tf_param=tf_param,
                electron_type=[("electron", "PID-fake-electron-et-cone", "electron")],
                muon_type=[("muon", "fake-muon", "muon")],
                el_cr=el_cr,
                mu_cr=mu_cr,
                systematic=syst,
                save=False,
                use_mp=True,
                enable_plot=True,
                show_nominal_only=True,
                correlation_correction = "../../met_correlation/calo_isolation_correction_met.pkl",
            )
            futures.append(future)
            #secede()

        counter = 0
        m_serial = Serialization()
        _start = time.time()
        for future in as_completed(futures):
            counter += 1
            fname, m_config = future.result()
            #print(f"doing {m_config}")
            _signal, _bkgd = m_serial.from_pickle(fname)
            output_signal.update(_signal)
            output_bkgd.update(_bkgd)
            pbar.update()
            #print(f"{counter=}")
            os.unlink(fname)
            os.unlink(m_config)
            # print(f"done with {fname}")
            if counter % njobs == 0:
                print(f"cost {time.time()-_start}s/{njobs}jobs")
                _start = time.time()
            future.release()

    m_serial = Serialization()
    m_serial.to_shelve(output_signal, f"run2_{signal}_signal_correction.shelf", flag="n")
    m_serial.to_shelve(output_bkgd, f"run2_{signal}_bkgd_correction.shelf", flag="n")


def concurrent_version(
    signal="wjets",
    bkgd=["zjets", "ttbar", "singletop", "diboson"],
    skip=["wjets_mg", "wjets_FxFx", "wjets_powheg"],
    el_cr={"ttbar": "ttbarCR_Ele", "zjets": "ZjetsCR_Ele"},
    mu_cr={"ttbar": "ttbarCR_Mu", "zjets": "ZjetsCR_Mu"},
):
    correlation_file = "../../dijets_study/abcd_correleation/calo_isolation_correction_met.pkl"

    run2 = ConfigMgr.open("run2.pkl")

    syslist = run2.list_systematic_full_name()

    pbar = tqdm.tqdm(total=len(syslist), dynamic_ncols=True, leave=False, position=3)

    output_bkgd = {}
    output_signal = {}
    futures = []
    workers = 5
    with concurrent.futures.ProcessPoolExecutor(workers) as client:
        start_time = time.time()
        for syst in syslist:
            print(f"submitting systematic {syst}")
            m_config = ConfigMgr.split(run2, split_type="systematic", syst_names=[syst])
            m_config = next(m_config)
            print(f"submit {syst}")
            #future = collinearw.histManipulate.run_iterative_correction(
            future = client.submit(collinearw.histManipulate.run_iterative_correction,
                m_config,
                signal=signal,
                skip=skip,
                bkgd=bkgd,
                iteration=5,
                tf_param=tf_param,
                #electron_type=[("electron", "PID-fake-electron-et-cone", "electron")],
                electron_type=[("electron", "met-pid-met-ELTrack-et-cone", "electron")],
                muon_type=[("muon", "muon-met-iso", "muon")],
                el_cr=el_cr,
                mu_cr=mu_cr,
                systematic=syst,
                save=True,
                use_mp=True,
                show_nominal_only=True,
                correlation_correction = correlation_file,
                legend_opt="int",
                corr_obs="nJet30",
                include_signal=True,
            )
            futures.append(future)

        for future in concurrent.futures.as_completed(futures):
            _signal, _bkgd = future.result()
            output_signal.update(_signal)
            output_bkgd.update(_bkgd)
            pbar.update()
            print(f"cost {time.time()-start_time}s")
            start_time = time.time()

    m_serial = Serialization()
    m_serial.to_shelve(output_signal, f"run2_{signal}_signal_correction.shelf", flag="n")
    m_serial.to_shelve(output_bkgd, f"run2_{signal}_bkgd_correction.shelf", flag="n")


if __name__ == "__main__":

    '''
    dask_version(
        signal="wjets_2211",
        skip=["wjets_mg", "wjets_FxFx", "wjets_powheg", "wjets", "zjets", "diboson"],
        bkgd=["zjets_2211", "ttbar", "singletop", "diboson_powheg"],
        el_cr={"ttbar": "ttbarCR_Ele", "zjets_2211": "ZjetsCR_Ele"},
        mu_cr={"ttbar": "ttbarCR_Mu", "zjets_2211": "ZjetsCR_Mu"},
    )
    '''
    concurrent_version(
        signal="wjets_2211",
        skip=["wjets_mg", "wjets_FxFx", "wjets_powheg", "wjets", "zjets", "diboson", "dijets"],
        bkgd=["zjets_2211", "ttbar", "singletop", "diboson_powheg", "wjets_2211_tau", "vgamma"],
        el_cr={"ttbar": "ttbarCR_Ele", "zjets_2211": "ZjetsCR_Ele"},
        mu_cr={"ttbar": "ttbarCR_Mu", "zjets_2211": "ZjetsCR_Mu"},
    )
