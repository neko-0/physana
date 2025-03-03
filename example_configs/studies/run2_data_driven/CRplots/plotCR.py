from collinearw import ConfigMgr, run_PlotMaker, histManipulate
from collinearw.serialization import Serialization
from collinearw.histMaker import weight_from_hist
import functools
import os
import logging
import dask
from dask_jobqueue import SLURMCluster
from dask.distributed import Client
from dask.distributed import as_completed as dask_as_completed
from compute_band import compute_band


logging.basicConfig()
logging.getLogger().setLevel(logging.CRITICAL)

doEventLevel = True
doCorr = True


def main():

    lookup_systematic_groups = {
        "experimental" : [
            ("JET-GroupedNP-up", ("*JET_GroupedNP*", "*JET*up*", "")),
            ("JET-GroupedNP-down", ("*JET_GroupedNP*", "*JET*down*", "")),
            ("JET-JER-up", ("*JET*JER*", "*JET*up*", "")),
            ("JET-JER-down", ("*JET*JER*", "*JET*down*", "")),
            ("JET-EtaIntercalibration-NonClosure-up", ("*JET*Eta*", "*JET*up*", "")),
            ("JET-EtaIntercalibration-NonClosure-down", ("*JET*Eta*", "*JET*down*", "")),
            ("Lepton-EL-Weight-up", ("*leptonWeight*EL*", "NoSys", "*leptonWeight*up*")),
            ("Lepton-EL-Weight-down", ("*leptonWeight*EL*", "NoSys", "*leptonWeight*down*")),
            ("Lepton-MU-Weight-up", ("*leptonWeight*MUON*", "NoSys", "*leptonWeight*up*")),
            ("Lepton-MU-Weight-down", ("*leptonWeight*MUON*", "NoSys", "*leptonWeight*down*")) ,
            ("JVT-up", ("*jvtWeight_JET_JvtEfficiency*", "NoSys", "*up*")),
            ("JVT-down", ("*jvtWeight_JET_JvtEfficiency*", "NoSys", "*down*")),
            ("Trig-EL-up", ("*trigWeight_EL*_*", "NoSys", "*up*")),
            ("Trig-EL-down", ("*trigWeight_EL*", "NoSys", "*down*")),
            ("Trig-MUON-up", ("*trigWeight_MUON*_*", "NoSys", "*up*")),
            ("Trig-MUON-down", ("*trigWeight_MUON*", "NoSys", "*down*")),
            ("bTagWeight-up", ("*bTagWeight*", "NoSys", "*up*")),
            ("bTagWeight-down", ("*bTagWeight*", "NoSys", "*down*")),
        ],

        "ttbar_theory": [
            ("ttbar-ren_fac_scale-up", ("*ttbar*ren_fac_scale*", "min_max", "max")),
            ("ttbar-ren_fac_scale-down", ("*ttbar*ren_fac_scale*", "min_max", "min")),
            ("ttbar-ISR_scale-up", ("*ttbar*ISR_scale*", "min_max", "max")),
            ("ttbar-ISR_scale-down", ("*ttbar*ISR_scale*", "min_max", "min")),
            ("ttbar-FSR_scale-up", ("*ttbar*FSR_scale*", "min_max", "max")) ,
            ("ttbar-FSR_scale-down", ("*ttbar*FSR_scale*", "min_max", "min")),
            ("ttbar-NNPDF30_PDF-up", ("*ttbar*NNPDF30_PDF*", "stdev", "std_up")),
            ("ttbar-NNPDF30_PDF-down", ("*ttbar*NNPDF30_PDF*", "stdev", "std_down")),
        ],

        "singletop_theory": [
            ("singletop-ren_fac_scale-up", ("*singletop_ren_fac_scale*", "min_max", "max")),
            ("singletop-ren_fac_scale-down", ("*singletop_ren_fac_scale*", "min_max", "min")),
        ],

        "zjets_theory": [
            ("zjets_2211-MUR_MUF-up", ("*zjets*", "min_max", "max")),
            ("zjets_2211-MUR_MUF-down", ("*zjets*", "min_max", "min")),
            ("zjets_2211-NNPDF30nnlo_hessian-up", ("*zjets*", "stdev", "std_up")),
            ("zjets_2211-NNPDF30nnlo_hessian-down", ("*zjets*", "stdev", "std_down")),
        ],

        "wjets_theory" : [
            ("wjets_2211-MUR_MUF-up", ("*wjets*", "min_max", "max")),
            ("wjets_2211-MUR_MUF-down", ("*wjets*", "min_max", "min")),
            ("wjets_2211-NNPDF30nnlo_hessian-up", ("*wjets*", "stdev", "std_up")),
            ("wjets_2211-NNPDF30nnlo_hessian-down", ("*wjets*", "stdev", "std_down")),
        ],

        "diboson_theory" : [
            ("diboson-MUR_MUF-up", ("*diboson*", "min_max", "max")),
            ("diboson-MUR_MUF-down", ("*diboson*", "min_max", "min")),
            ("diboson-NNPDF30nnlo_hessian-up", ("*diboson*", "stdev", "std_up")),
            ("diboson-NNPDF30nnlo_hessian-down", ("*diboson*", "stdev", "std_down")),
        ],
    }

    compute_systematic_groups = [
        ("wjets_2211_MUR_MUF_Scale", "min_max"),
        ("wjets_2211_NNPDF30nnlo_hessian", "stdev"),
        ("zjets_2211_MUR_MUF_Scale", "min_max"),
        ("zjets_2211_NNPDF30nnlo_hessian", "stdev"),
        ("diboson_MUR_MUF_Scale", "min_max"),
        ("diboson_NNPDF30nnlo_hessian", "stdev"),
        ("ttbar_ren_fac_scale", "min_max"),
        ("ttbar_ISR_scale", "min_max"),
        ("ttbar_FSR_scale", "min_max"),
        ("ttbar_NNPDF30_PDF", "stdev"),
        ("singletop_ren_fac_scale", "min_max"),
    ]

    #compute_band("run2.pkl")
    run2 = ConfigMgr.open("run2.pkl")

    run_PlotMaker.run_stack(
        run2,
        "cr_plots_corr",
        data="data",
        mcs=["wjets_2211", "zjets_2211", "ttbar", "singletop", "diboson_powheg", "wjets_2211_tau", "dijets"],
        low_yrange=(0.5, 1.7),
        logy=True,
        workers=None,
        # rname_filter=["*rA*"],
        # check_region=True,
        low_ytitle="Data/Pred",
        #lookup_systematic_groups = lookup_systematic_groups,
        #compute_systematic_groups = compute_systematic_groups,
        hide_process=False,
        legend_opt="int",
    )


if __name__ == "__main__":
    main()
