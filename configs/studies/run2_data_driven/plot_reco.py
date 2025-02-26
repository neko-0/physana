from collinearw import ConfigMgr, run_PlotMaker, histManipulate, PlotMaker
from collinearw.serialization import Serialization
from collinearw.histMaker import weight_from_hist
from collinearw.strategies.systematics.core import compute_quadrature_sum, compute_systematics, compute_process_set_systematics

import functools
import os
import logging
import dask
from dask_jobqueue import SLURMCluster
from dask.distributed import Client
from dask.distributed import as_completed as dask_as_completed
from compute_band import compute_band


logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)


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
            ("singletop-A14_Tunning", ("*singletop_A14_Tunning*", "min_max", "*")),
            ("singletop-PDF", ("*singletop_PDF*", "stdev", "*")),
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
            # ("diboson-MUR_MUF-up", ("*diboson*", "min_max", "max")),
            # ("diboson-MUR_MUF-down", ("*diboson*", "min_max", "min")),
            # ("diboson-NNPDF30nnlo_hessian-up", ("*diboson*", "stdev", "std_up")),
            # ("diboson-NNPDF30nnlo_hessian-down", ("*diboson*", "stdev", "std_down")),
            ("diboson_powheg-MUR_MUF_Scale", ("*diboson_powheg*MUR_MUF_Scale*", "min_max", "*")),
        ],

        "dijets_theory" : [
            # ("dijets_flat", ("dijets_Flat15Percent", "NoSys", "*")),
            ("dijets_FSR_scale", ("dijets_FSR_scale", "min_max", "*")),
            ("dijets_ISR_scale", ("dijets_ISR_scale", "min_max", "*")),
            ("dijets_A14_Tunning", ("dijets_A14*", "min_max", "*")),
        ],
    }

    compute_systematic_groups = [
        ("wjets_2211_MUR_MUF_Scale", "min_max"),
        ("wjets_2211_NNPDF30nnlo_hessian", "stdev"),
        ("zjets_2211_MUR_MUF_Scale", "min_max"),
        ("zjets_2211_NNPDF30nnlo_hessian", "stdev"),
        #("diboson_MUR_MUF_Scale", "min_max"),
        #("diboson_NNPDF30nnlo_hessian", "stdev"),
        ("diboson_powheg_MUR_MUF_Scale", "min_max"),
        ("ttbar_ren_fac_scale", "min_max"),
        ("ttbar_ISR_scale", "min_max"),
        ("ttbar_FSR_scale", "min_max"),
        ("ttbar_NNPDF30_PDF", "stdev"),
        ("singletop_ren_fac_scale", "min_max"),
        ("singletop_A14_Tunning", "min_max"),
        ("singletop_PDF", "stdev"),
        ("JET_GroupedNP_1", "symmetrize_up_down"),
        ("JET_GroupedNP_2", "symmetrize_up_down"),
        ("JET_GroupedNP_3", "symmetrize_up_down"),
        ("JET_JER_EffectiveNP_1", "symmetrize_up_down"),
        ("JET_JER_EffectiveNP_2", "symmetrize_up_down"),
        ("JET_JER_EffectiveNP_3", "symmetrize_up_down"),
        ("JET_JER_EffectiveNP_4", "symmetrize_up_down"),
        ("JET_JER_EffectiveNP_5", "symmetrize_up_down"),
        ("JET_JER_EffectiveNP_6", "symmetrize_up_down"),
        ("JET_JER_EffectiveNP_7restTerm", "symmetrize_up_down"),
        ("dijets_ISR_scale", "min_max"),
        ("dijets_FSR_scale", "min_max"),
        ("dijets_A14_Tunning", "min_max"),
    ]

    mc_list = [
        "wjets_2211",
        "zjets_2211",
        "ttbar",
        "singletop",
        "diboson_powheg",
        "wjets_2211_tau",
        "vgamma",
        "dijets",
    ]

    compute_reco_band = False

    if compute_reco_band:
        run2 = ConfigMgr.open("prune_run2_2211.pkl")
        mc_sum = histManipulate.sum_process_sets(
            [run2.get_process_set(p) for p in mc_list]
        )
        for syst_group in compute_systematic_groups:
            # syst_group should be tuple (parent name, algorithm name)
            # e.g. ("wjets_2211_MUR_MUF_Scale", "min_max")
            compute_process_set_systematics(mc_sum, *syst_group)

        # generate dict of systematic groups
        syst_groups = {}
        for syst_type, syst_lookup_list in lookup_systematic_groups.items():
            logger.info(f"found {syst_type} type systematic")
            syst_groups[syst_type] = {}
            for lookup_key in syst_lookup_list:
                syst_groups[syst_type].update(
                    mc_sum.generate_systematic_group(*lookup_key)
                )

        # compute quad sum for syst.
        for syst_type in syst_groups:
            for name, syst_list in syst_groups[syst_type].items():
                logger.info(f"computing {syst_type} {name}.")
                compute_quadrature_sum(
                    run2,
                    "",
                    name,
                    syst_type,
                    syst_list,
                    external_process_set=mc_sum,
                )

        run2.append_process(mc_sum)
        splitted_run2 = run2.self_split("systematic")
        run2 = next(splitted_run2)
        run2.save("reco_band.pkl")
    else:
        run2 = ConfigMgr.open("reco_band.pkl")


    external_syst_process = run2.get_process(f"sum({','.join(mc_list)})")

    #PlotMaker.PLOT_STATUS = "Work In Progress"

    run_PlotMaker.run_stack(
        run2,
        "reco_plots_corr_updates",
        data="data",
        mcs=mc_list,
        low_yrange=(0.5, 1.7),
        yrange=(10, 1e6),
        logy=True,
        workers=None,
        rname_filter=["*rA*"],
        # check_region=True,
        low_ytitle="Data/Pred",
        include_systematic_band = True,
        #lookup_systematic_groups = lookup_systematic_groups,
        #compute_systematic_groups = compute_systematic_groups,
        external_syst_process = external_syst_process,
        hide_process=False,
        legend_opt="",
        enable_legend=True,
    )


if __name__ == "__main__":
    main()
