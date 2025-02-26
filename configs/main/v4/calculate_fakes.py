from collinearw import ConfigMgr
from collinearw import histManipulate
import logging
from dask_jobqueue import SLURMCluster
from dask.distributed import Client
from dask.distributed import as_completed as dask_as_completed

logging.basicConfig()
logging.getLogger().setLevel(logging.CRITICAL)

doEventLevel = True
doCorr = True

workers = 40
account = "shared"
cluster = SLURMCluster(
    queue=account,
    walltime='01:00:00',
    project="collinear_Wjets",
    cores=16,
    memory="200 GB",
    job_extra=[f'--account={account}', f'--partition={account}'],
    local_directory="dask_fakes_output",
    log_directory="dask_fakes_logs",
    death_timeout=36000,
)
cluster.scale(jobs=workers)
client = Client(cluster)
client.get_versions(check=True)
print(cluster.job_script())


def main(signal="wjets_2211", skip=["wjets", "zjets"]):

    run2 = ConfigMgr.open("run2_2211.pkl")
    if doEventLevel:
        skip_processes = [
            process.name
            for process in run2.processes
            if process.process_type not in ["data", "signal", "bkg"]
        ]

        skip_processes += skip

        tf_params = {
            "*muon*": (("abs(lep1Eta)", "lep1Pt"), "eta_vs_lepPt_muon"),
            "*electron*": (("abs(lep1Eta)", "lep1Pt"), "eta_vs_lepPt_electron"),
        }

        corr_path = "./iterative/run2_wjets_2211_signal_correction.pkl"

        fake_config = histManipulate.run_abcd_fakes_estimation(
            run2,
            tf_params,
            signal=signal,
            correction=corr_path,
            skip_process=skip_processes,
            systematic=run2.list_systematic_full_name(),
            use_mp=True,
            # nonclosure="./nonclosure/nonclosure.pkl",
            executor=client,
            as_completed=dask_as_completed,
            # workers=10,
        )
        # breakpoint()
    else:
        # pass
        fake_config = histManipulate.run_ABCD_Fakes(run2, False, None, ext_tf=None)

    fake_config.save('fakes_2211.pkl')

    from collinearw import run_PlotMaker

    run2_fakes = ConfigMgr.open("fakes_2211.pkl")
    run_PlotMaker.run_stack(
        run2_fakes,
        "fakes_2211_plots",
        data="data",
        mcs=["wjets_2211", "zjets_2211", "ttbar", "singletop", "diboson", "fakes"],
        low_yrange=(0.5, 1.7),
        logy=True,
        workers=None,
        rname_filter=["*rA*"],
        # check_region=True,
        low_ytitle="Data/Pred",
    )


if __name__ == "__main__":
    main()
