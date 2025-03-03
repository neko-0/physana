from collinearw import ConfigMgr
from collinearw.serialization import Serialization
from collinearw import histManipulate
from collinearw.histMaker import weight_from_hist
from collinearw import run_PlotMaker
import functools
import os
import logging
import dask
from dask_jobqueue import SLURMCluster
from dask.distributed import Client
from dask.distributed import as_completed as dask_as_completed


logging.basicConfig()
logging.getLogger().setLevel(logging.CRITICAL)

doEventLevel = True
doCorr = True

'''
tf_params = {
    "*muon*": (("abs(lep1Eta)", "lep1Pt"), "eta_vs_lepPt_muon"),
    "*electron*": (("abs(lep1Eta)", "lep1Pt"), "eta_vs_lepPt_electron"),
}
'''
tf_params = {
    "*muon*inclusive_*" : (("Ht30-jet1Pt"), "Ht30-jet1Pt"),
    "*muon*inclusive2j_*" : (("Ht30-jet1Pt"), "Ht30-jet1Pt"),
    "*muon*collinear_*" : (("jet1Pt", "Ht30-jet1Pt+lep1Pt"), "jet1Pt_Ht30"),
    #"*muon*backtoback_*" : (("jet1Pt", "Ht30-jet1Pt+lep1Pt"), "jet1Pt_Ht30"),
    #"*muon*backtoback_*" : (("Ht30-jet1Pt"), "Ht30-jet1Pt"),
    "*muon*backtoback_*": (("abs(lep1Eta)", "lep1Pt"), "eta_vs_lepPt_muon"),
    "*electron*": (("abs(lep1Eta)", "lep1Pt"), "eta_vs_lepPt_electron"),
}

def main(signal="wjets_2211", skip=["wjets", "zjets"]):

    print("opening file.")
    run2 = ConfigMgr.open("run2_2211.pkl")
    systlist = run2.list_systematic_full_name()

    workers = min(200, len(systlist))
    cores = 16
    account = "shared"
    '''
    cluster = SLURMCluster(
        queue=account,
        walltime='03:00:00',
        project="collinear_Wjets",
        nanny=False,
        cores=1,
        processes=1,
        memory="128GB",
        job_extra=[f'--account={account}', f'--partition={account}', f'--cpus-per-task={cores}'],
        local_directory="dask_fakes_output",
        log_directory="dask_fakes_logs",
        # death_timeout=36000,
        n_workers=workers,
    )
    client = Client(cluster)
    client.get_versions(check=True)
    print(cluster.job_script())
    '''

    print("ready for fakes.")
    if doEventLevel:
        skip_processes = [
            process.name
            for process in run2.processes
            if process.process_type not in ["data", "signal", "bkg"]
        ]

        skip_processes += skip

        print(f"skipping {skip_processes}")

        #run2_fakes = ConfigMgr.open("fakes_2211_nominal.pkl")
        run_PlotMaker.run_stack(
            run2,
            "fakes_2211_nominal_before_fakes",
            data="data",
            mcs=["wjets_2211", "zjets_2211", "ttbar", "singletop", "diboson_powheg", "wjets_2211_tau", "vgamma"],
            low_yrange=(0.5, 1.7),
            logy=True,
            workers=8,
            #rname_filter=["*rA*"],
            # check_region=True,
            legend_opt="int",
            low_ytitle="Data/Pred",
        )

        basedir = os.path.dirname(os.path.realpath(__file__))
        corr_path = "./iterative_sf/run2_wjets_2211_signal_correction.shelf"

        fake_config = histManipulate.run_abcd_fakes_estimation(
            run2,
            tf_params,
            signal = signal,
            correction=corr_path,
            skip_process=skip_processes,
            systematic=systlist,
            use_mp=True,
            enforced_mp=True,
            workers=8,
            #ext_tf_config="MC_TF.pkl",
            #nonclosure="./nonclosure/nonclosure.pkl",
            #executor = client,
            #as_completed = dask_as_completed,
            correlation_correction = "../dijets_study/abcd_correleation/calo_isolation_correction_met.pkl",
            #workers=10,
            prune=False,
        )
        #breakpoint()
    else:
        #pass
        fake_config = histManipulate.run_ABCD_Fakes(run2, False, None, ext_tf=None)

    fake_config.save('fakes_2211.pkl')

    #run2_fakes = ConfigMgr.open("fakes_2211_nominal.pkl")
    run_PlotMaker.run_stack(
        fake_config,
        "fakes_plots",
        data="data",
        mcs=["wjets_2211", "zjets_2211", "ttbar", "singletop", "diboson_powheg", "wjets_2211_tau", "vgamma", "fakes"],
        low_yrange=(0.5, 1.7),
        logy=True,
        workers=None,
        rname_filter=["*rA*"],
        # check_region=True,
        legend_opt="int",
        low_ytitle="Data/Pred",
    )


if __name__ == "__main__":
    main(skip=["wjets", "zjets", "diboson", "dijets"])
