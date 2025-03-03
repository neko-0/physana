from collinearw import ConfigMgr
from collinearw.serialization import Serialization
import collinearw.histManipulate
import concurrent.futures
from dask_jobqueue import SLURMCluster
from dask.distributed import Client, as_completed  # , LocalCluster
import tqdm


def dask_version(
    signal="wjets",
    bkgd=["zjets", "ttbar", "singletop", "diboson"],
    skip=["wjets_mg", "wjets_FxFx", "wjets_powheg"],
    el_cr={"ttbar": "ttbarCR_Ele", "zjets": "ZjetsCR_Ele"},
    mu_cr={"ttbar": "ttbarCR_Mu", "zjets": "ZjetsCR_Mu"},
):
    # dask.config.set({"distributed.worker.daemon ": False})

    workers = 20
    cluster = SLURMCluster(
        queue='usatlas',
        walltime='03:00:00',
        project="collinear_Wjets",
        cores=16,
        # processes=1,
        # nanny=False,
        memory="250 GB",
        job_extra=['--account=usatlas', '--partition=usatlas'],
        local_directory="dask_iterative_output",
        log_directory="dask_iterative_logs",
        # death_timeout=36000,
    )
    cluster.scale(jobs=workers)
    # cluster = LocalCluster()

    run2 = ConfigMgr.open("run2.pkl")

    syslist = run2.list_systematic_full_name()

    pbar = tqdm.tqdm(total=len(syslist), dynamic_ncols=True, leave=False, position=3)

    output_bkgd = {}
    output_signal = {}
    futures = []
    with cluster, Client(cluster) as client:
        for syst in syslist:

            future = client.submit(
                collinearw.histManipulate.run_iterative_correction,
                "run2.pkl",
                signal=signal,
                skip=skip,
                bkgd=bkgd,
                iteration=5,
                electron_type=[("electron", "fake-electron-PID", "electron")],
                muon_type=[("muon", "fake-muon", "muon")],
                el_cr=el_cr,
                mu_cr=mu_cr,
                systematic=syst,
                save=False,
                use_mp=False,
            )
            futures.append(future)

        for future in as_completed(futures):
            _signal, _bkgd = future.result()
            output_signal.update(_signal)
            output_bkgd.update(_bkgd)
            pbar.update()

    m_serial = Serialization()
    m_serial.to_pickle(output_signal, f"run2_{signal}_signal_correction.pkl")
    m_serial.to_pickle(output_bkgd, f"run2_{signal}_bkgd_correction.pkl")


def concurrent_version(
    signal="wjets",
    bkgd=["zjets", "ttbar", "singletop", "diboson"],
    skip=["wjets_mg", "wjets_FxFx", "wjets_powheg"],
    el_cr={"ttbar": "ttbarCR_Ele", "zjets": "ZjetsCR_Ele"},
    mu_cr={"ttbar": "ttbarCR_Mu", "zjets": "ZjetsCR_Mu"},
):

    run2 = ConfigMgr.open("run2.pkl")

    syslist = run2.list_systematic_full_name()

    pbar = tqdm.tqdm(total=len(syslist), dynamic_ncols=True, leave=False, position=3)

    output_bkgd = {}
    output_signal = {}
    futures = []
    workers = 10
    with concurrent.futures.ProcessPoolExecutor(workers) as client:
        for syst in syslist:
            future = client.submit(
                collinearw.histManipulate.run_iterative_correction,
                "run2.pkl",
                signal=signal,
                skip=skip,
                bkgd=bkgd,
                iteration=5,
                electron_type=[("electron", "fake-electron-PID", "electron")],
                muon_type=[("muon", "fake-muon", "muon")],
                el_cr=el_cr,
                mu_cr=mu_cr,
                systematic=syst,
                save=False,
                use_mp=False,
            )
            futures.append(future)

        for future in concurrent.futures.as_completed(futures):
            _signal, _bkgd = future.result()
            output_signal.update(_signal)
            output_bkgd.update(_bkgd)
            pbar.update()

    m_serial = Serialization()
    m_serial.to_pickle(output_signal, f"run2_{signal}_signal_correction.pkl")
    m_serial.to_pickle(output_bkgd, f"run2_{signal}_bkgd_correction.pkl")


if __name__ == "__main__":

    dask_version(
        signal="wjets_2211",
        skip=["wjets_mg", "wjets_FxFx", "wjets_powheg", "wjets", "zjets"],
        bkgd=["zjets_2211", "ttbar", "singletop", "diboson"],
        el_cr={"ttbar": "ttbarCR_Ele", "zjets_2211": "ZjetsCR_Ele"},
        mu_cr={"ttbar": "ttbarCR_Mu", "zjets_2211": "ZjetsCR_Mu"},
    )
    dask_version(
        signal="wjets",
        skip=["wjets_mg", "wjets_FxFx", "wjets_powheg", "wjets_2211", "zjets_2211"],
        bkgd=["zjets", "ttbar", "singletop", "diboson"],
        el_cr={"ttbar": "ttbarCR_Ele", "zjets": "ZjetsCR_Ele"},
        mu_cr={"ttbar": "ttbarCR_Mu", "zjets": "ZjetsCR_Mu"},
    )
