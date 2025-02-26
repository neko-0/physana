import ROOT
from collinearw import ConfigMgr, run_PlotMaker, histManipulate, PlotMaker
from collinearw.serialization import Serialization
from collinearw.histMaker import weight_from_hist
from collinearw.strategies.systematics.core import (
    compute_quadrature_sum,
    compute_systematics,
    compute_process_set_systematics,
)

import functools
import os
import logging

# import dask
# from dask_jobqueue import SLURMCluster
# from dask.distributed import Client
# from dask.distributed import as_completed as dask_as_completed
# from compute_band import compute_band


logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)


doEventLevel = True
doCorr = True


def main():
    lookup_systematic_groups = {
        "experimental": [
            ("JET", ("*JET*", "*symmetrize*", "*")),
            ("JVT", ("*jvtWeight_JET_JvtEfficiency*", "NoSys", "*")),
            ("LeptonWeight", ("*leptonWeight*", "NoSys", "*leptonWeight*")),
            ("TriggerWeight", ("*triggerWeight*", "NoSys", "*")),
            ("MUON", ("*MUON*", "*MUON*", "*")),
            ("EG", ("*EG*", "*EG*", "*")),
            ("MET", ("*MET*", "*MET*", "*")),
            ("bTagWeight", ("*bTagWeight*", "NoSys", "*")),
            ("bTagFlavor", ("bTag_*", "*", "*")),
        ],
        "ttbar_theory": [
            ("ttbar-MUR_MUF", ("*ttbar*MUR_MUF*", "min_max", "*")),
            ("ttbar-NNPDF30nnlo_hessian", ("*ttbar*", "hessian", "*")),
            ("ttbar-Alpha_S", ("*ttbar*Alpha*", "min_max", "*")),
            ("ttbar_SF_variation", ("ttar_TF*", "*", "*")),
        ],
        "singletop_theory": [
            ("singletop-ren_fac_scale", ("*singletop_ren_fac_scale*", "min_max", "*")),
            ("singletop-A14_Tunning", ("*singletop_A14_Tunning*", "min_max", "*")),
            ("singletop-PDF", ("*singletop_PDF*", "stdev", "*")),
            ("singletop-DS", ("*singletop_Wt_DS*", "*", "*")),
        ],
        "zjets_theory": [
            ("zjets_SF_variation", ("zjets_2211_TF*", "*", "*")),
            ("zjets_2211-MUR_MUF", ("*zjets*MUR_MUF*", "min_max", "*")),
            ("zjets_2211-NNPDF30nnlo_hessian", ("*zjets*", "hessian", "*")),
            ("zjets_2211-Alpha_S", ("*zjets*Alpha*", "min_max", "*")),
        ],
        "wjets_theory": [
            ("wjets_2211-MUR_MUF", ("wjets_2211_MUR_MUF_Scale", "min_max", "*")),
            (
                "wjets_2211-NNPDF30nnlo_hessian",
                ("wjets_2211_NNPDF30nnlo_hessian", "hessian", "*"),
            ),
            ("wjets_2211-Alpha_S", ("wjets_2211_Alpha_S", "min_max", "*")),
        ],
        "diboson_theory": [
            (
                "diboson_powheg-MUR_MUF_Scale",
                ("*diboson_powheg*MUR_MUF_Scale*", "min_max", "*"),
            ),
        ],
        "dijets_theory": [
            ("dijets_FSR_scale", ("dijets_FSR_scale", "min_max", "*")),
            ("dijets_ISR_scale", ("dijets_ISR_scale", "min_max", "*")),
            ("dijets_A14_Tunning", ("dijets_A14*", "min_max", "*")),
            ("dijets_CR_Def_Var", ("dijets_CR_Def*", "*", "*")),
        ],
        "vgamma_theory": [
            ("vgamma-MUR_MUF", ("vgamma_MUR_MUF_Scale", "min_max", "*")),
            (
                "vgamma-NNPDF30nnlo_hessian",
                ("vgamma_NNPDF30nnlo_hessian", "hessian", "*"),
            ),
            ("vgamma-Alpha_S", ("vgamma_Alpha_S", "min_max", "*")),
        ],
        "wjj_theory": [
            ("wjets_EW_2211-MUR_MUF", ("wjets_EW_2211_MUR_MUF_Scale", "min_max", "*")),
            (
                "wjets_EW_2211-NNPDF30nnlo_hessian",
                ("wjets_EW_2211_NNPDF30nnlo_hessian", "hessian", "*"),
            ),
            ("wjets_EW_2211-Alpha_S", ("wjets_EW_2211_Alpha_S", "min_max", "*")),
        ],
        "wjets_tau_theory": [
            ("wjets_tau-MUR_MUF", ("wjets_tau_MUR_MUF_Scale", "min_max", "*")),
            ("wjets_tau-PDF", ("wjets_tau_*PDF*", "hessian", "*")),
        ],
    }

    compute_systematic_groups = [
        ("wjets_2211_MUR_MUF_Scale", "min_max"),
        ("wjets_2211_NNPDF30nnlo_hessian", "stdev"),
        ("wjets_2211_Alpha_S", "min_max"),
        ("zjets_2211_MUR_MUF_Scale", "min_max"),
        ("zjets_2211_NNPDF30nnlo_hessian", "stdev"),
        ("zjets_2211_Alpha_S", "min_max"),
        # ("diboson_MUR_MUF_Scale", "min_max"),
        # ("diboson_NNPDF30nnlo_hessian", "stdev"),
        ("diboson_powheg_MUR_MUF_Scale", "min_max"),
        ("ttbar_ren_fac_scale", "min_max"),
        ("ttbar_ISR_scale", "min_max"),
        ("ttbar_FSR_scale", "min_max"),
        ("ttbar_NNPDF30_PDF", "stdev"),
        ("ttbar_MUR_MUF_Scale", "min_max"),
        ("ttbar_NNPDF30nnlo_hessian", "hessian"),
        ("ttbar_Alpha_S", "min_max"),
        ("singletop_ren_fac_scale", "min_max"),
        ("singletop_A14_Tunning", "min_max"),
        ("singletop_PDF", "stdev"),
        ("JET_JER_EffectiveNP_1", "symmetrize_up_down"),
        ("JET_JER_EffectiveNP_2", "symmetrize_up_down"),
        ("JET_JER_EffectiveNP_3", "symmetrize_up_down"),
        ("JET_JER_EffectiveNP_4", "symmetrize_up_down"),
        ("JET_JER_EffectiveNP_5", "symmetrize_up_down"),
        ("JET_JER_EffectiveNP_6", "symmetrize_up_down"),
        ("JET_JER_EffectiveNP_7", "symmetrize_up_down"),
        ("JET_JER_EffectiveNP_8", "symmetrize_up_down"),
        ("JET_JER_EffectiveNP_9", "symmetrize_up_down"),
        ("JET_JER_EffectiveNP_10", "symmetrize_up_down"),
        ("JET_JER_EffectiveNP_11", "symmetrize_up_down"),
        ("JET_JER_EffectiveNP_12restTerm", "symmetrize_up_down"),
        ("JET_BJES_Response", "symmetrize_up_down"),
        ("JET_EffectiveNP_Detector1", "symmetrize_up_down"),
        ("JET_EffectiveNP_Detector2", "symmetrize_up_down"),
        ("JET_EffectiveNP_Mixed1", "symmetrize_up_down"),
        ("JET_EffectiveNP_Mixed2", "symmetrize_up_down"),
        ("JET_EffectiveNP_Mixed3", "symmetrize_up_down"),
        ("JET_EffectiveNP_Modelling1", "symmetrize_up_down"),
        ("JET_EffectiveNP_Modelling2", "symmetrize_up_down"),
        ("JET_EffectiveNP_Modelling3", "symmetrize_up_down"),
        ("JET_EffectiveNP_Modelling4", "symmetrize_up_down"),
        ("JET_EffectiveNP_R10_Mixed1", "symmetrize_up_down"),
        ("JET_EffectiveNP_R10_Modelling1", "symmetrize_up_down"),
        ("JET_EffectiveNP_Statistical1", "symmetrize_up_down"),
        ("JET_EffectiveNP_Statistical2", "symmetrize_up_down"),
        ("JET_EffectiveNP_Statistical3", "symmetrize_up_down"),
        ("JET_EffectiveNP_Statistical4", "symmetrize_up_down"),
        ("JET_EffectiveNP_Statistical6", "symmetrize_up_down"),
        ("JET_JetTagSF_Dijet_Modelling", "symmetrize_up_down"),
        ("JET_JetTagSF_Gammajet_Modelling", "symmetrize_up_down"),
        ("JET_JetTagSF_Hadronisation", "symmetrize_up_down"),
        ("JET_JetTagSF_MatrixElement", "symmetrize_up_down"),
        ("JET_JetTagSF_Radiation", "symmetrize_up_down"),
        ("JET_PunchThrough_MC16", "symmetrize_up_down"),
        ("JET_SingleParticle_HighPt", "symmetrize_up_down"),
        ("JET_Rtrk_Baseline_frozen_mass", "symmetrize_up_down"),
        ("JET_Rtrk_ExtraComp_Baseline_frozen_mass", "symmetrize_up_down"),
        ("JET_Rtrk_ExtraComp_Modelling_frozen_mass", "symmetrize_up_down"),
        ("JET_Rtrk_Modelling_frozen_mass", "symmetrize_up_down"),
        ("JET_Rtrk_TotalStat_frozen_mass", "symmetrize_up_down"),
        ("JET_Rtrk_Tracking1_frozen_mass", "symmetrize_up_down"),
        ("JET_Rtrk_Tracking2_frozen_mass", "symmetrize_up_down"),
        ("JET_Rtrk_Tracking3_frozen_mass", "symmetrize_up_down"),
        ("JET_EtaIntercalibration_Modelling", "symmetrize_up_down"),
        ("JET_EtaIntercalibration_NonClosure_2018data", "symmetrize_up_down"),
        ("JET_EtaIntercalibration_NonClosure_highE", "symmetrize_up_down"),
        ("JET_EtaIntercalibration_NonClosure_negEta", "symmetrize_up_down"),
        ("JET_EtaIntercalibration_NonClosure_posEta", "symmetrize_up_down"),
        ("JET_EtaIntercalibration_TotalStat", "symmetrize_up_down"),
        ("dijets_ISR_scale", "min_max"),
        ("dijets_FSR_scale", "min_max"),
        ("dijets_A14_Tunning", "min_max"),
        ("wjets_EW_2211_MUR_MUF_Scale", "min_max"),
        ("wjets_EW_2211_NNPDF30nnlo_hessian", "hessian"),
        ("wjets_EW_2211_Alpha_S", "min_max"),
        ("vgamma_MUR_MUF_Scale", "min_max"),
        ("vgamma_NNPDF30nnlo_hessian", "hessian"),
        ("vgamma_Alpha_S", "min_max"),
        ("wjets_tau_MUR_MUF_Scale", "min_max"),
        ("wjets_tau_PDF", "hessian"),
        # ("dijets_SF", "min_max"),
    ]

    mc_list = [
        "wjets_2211",
        "zjets_FxFx",
        "zjets_2211_tau",
        "ttbar",
        "singletop",
        "diboson_powheg",
        "wjets_tau",
        "vgamma",
        "dijets",
        "wjets_EW_2211",
    ]

    compute_reco_band = False

    if compute_reco_band:
        run2 = ConfigMgr.open("merged_run2_full_syst_June15.pkl")
        # run2 = ConfigMgr.open("merged_DS_run2_lowMet_TF_CR.pkl")

        # run2.get_process_set("ttbar").get().title = "ttbar (Sh 2.2.12)" # temp fix name
        # run2.get_process_set("zjets_2211").get().title = "Z+jets (Sh 2.2.11)" # temp fix name

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

        # grouping process =====================================================
        top_sum = histManipulate.sum_process_sets(
            [run2.get_process_set(x) for x in ["ttbar", "singletop"]]
        )
        # only if you need systematic band specific to Top process. -----------
        for syst_group in compute_systematic_groups:
            compute_process_set_systematics(top_sum, *syst_group)
        # generate dict of systematic groups
        syst_groups = {}
        for syst_type, syst_lookup_list in lookup_systematic_groups.items():
            logger.info(f"found {syst_type} type systematic")
            syst_groups[syst_type] = {}
            for lookup_key in syst_lookup_list:
                syst_groups[syst_type].update(
                    top_sum.generate_systematic_group(*lookup_key)
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
                    external_process_set=top_sum,
                )
        # ----------------------------------------------------------------------

        top_sum.name = "Top"
        for p in top_sum:
            p.title = "Top"
            p.name = "Top"
        run2.append_process(top_sum)

        # ===================================================== grouping process

        # get only the nominal after the band is computed
        splitted_run2 = run2.self_split("systematic")
        run2 = next(splitted_run2)
        run2.save("reco_band_full_syst.pkl")
        run2 = ConfigMgr.open("reco_band_full_syst.pkl")
    else:

        run2 = ConfigMgr.open(
            #"prefit_run2_nominal_only.pkl"
            "wjets_2211_fake_iter1.pkl"
            #"/gpfs/slac/atlas/fs1/d/yuzhan/configs_bk/ColWJet_2022/v6b/run2_MC_driven_Aug2_ABCD/fakes_2211_nominal_corr.pkl"
            #"run2.pkl"
            #"/gpfs/slac/atlas/fs1/d/yuzhan/configs_bk/ColWJet_2022/v6b/run2_MC_driven_Jun16_noBjetVeto/merged_run2_full_syst_June15.pkl"
        )  

#"W+jets"   : {"processes": ["wjets_2211","wjets_tau","wjets_EW_2211"], "color": ROOT.TColor.GetColor("#de77ae") }
        sum_rules = {"W+jets"   : {"processes": ["wjets_2211","wjets_tau","wjets_EW_2211"], "color": ROOT.TColor.GetColor("#de77ae") },
                     "Z+jets"   : {"processes": ["zjets_FxFx"], "color": ROOT.TColor.GetColor("#d73027") },
                     "Top"      : {"processes": ["ttbar", "singletop"], "color": ROOT.TColor.GetColor("#fdae61")},
                     "V+#gamma" : {"processes": ["vgamma"], "color": ROOT.TColor.GetColor("#4575b4")},
                     "Diboson"  : {"processes": ["diboson_powheg"], "color": ROOT.TColor.GetColor("#8073ac") },
                     "Multijet" : {"processes": ["dijets"], "color": ROOT.TColor.GetColor("#5aae61") } }

        for proc in sum_rules:
            for iproc in sum_rules[proc]["processes"]:
              mc_list.remove(iproc)

        for proc in sum_rules:

          proc_sum = histManipulate.sum_process_sets(
              [run2.get_process_set(x) for x in sum_rules[proc]["processes"]],proc
          )

          proc_sum.name = proc

          for p in proc_sum:
            p.title = proc
            p.name = proc
            p.color = sum_rules[proc]["color"]
          run2.append_process(proc_sum)

          mc_list.append(proc)

    run2.out_path = "./"
    run2.update_children_parent()
    # external_syst_process = run2.get_process(f"sum({','.join(mc_list)})")
    # ext_pset = run2.get_process_set(f"sum({','.join(mc_list)})")
    # ext_pset.name = "with inclusive TF"
    # for p in ext_pset:
    #     p.title = "with inclusive TF"
    #     p.name = "with inclusive TF"

    # corr_cr = ConfigMgr.open("./iterative_sf/run2.pkl")
    # corr_cr_mc = []
    # for p in mc_list:
    #     try:
    #         corr_cr_mc.append(corr_cr.get_process_set(p))
    #     except:
    #         continue
    # mc_sum = histManipulate.sum_process_sets(
    #
    # )

    # PlotMaker.PLOT_STATUS = "Work In Progress"

    # alt_run2 = ConfigMgr.open("../run2_MC_driven_Jun16_noBjetVeto_ColIncTF/run2.pkl")
    # alt_sum = histManipulate.sum_process_sets(
    #     [alt_run2.get_process_set(p) for p in mc_list]
    # )
    # alt_sum.name = "with collinear TF"
    # # bkgd_sum.legendname = "Others"
    # for p in alt_sum:
    #     print(p.title, p.name)
    #     p.title = "with collinear TF"
    #     p.name = "with collinear TF"
    # run2.append_process(alt_sum)
    # run2.update_children_parent()

    run_PlotMaker.run_stack(
        run2,
        "reco_plot_ZjetsFxFx_Aug16_2023",
        # data="with inclusive TF",
        # mcs=[f"with collinear TF"],
        data='data',
        mcs=mc_list,  # e.g. ['Top']
        low_yrange=(0.3, 1.7),
        yrange=(10, 1e6),
        logy=True,
        workers=None,
        #rname_filter=["*low*rA*"], #ttbar*","*Zjets*","*dijet*"],
        rname_filter=["*Zjets*"], #ttbar*","*Zjets*","*dijet*"],
        # check_region=True,
        # low_ytitle="Col./Inc.",
        low_ytitle="Data/Pred",
        include_systematic_band=True,
        # lookup_systematic_groups = lookup_systematic_groups,
        # compute_systematic_groups = compute_systematic_groups,
        # external_syst_process = external_syst_process,
        hide_process=False,
        #do_sort=False,
        legend_opt="",
        enable_legend=True,
        legend_pos=(0.52, 0.70, 0.85, 0.89),
    )


if __name__ == "__main__":
    main()
