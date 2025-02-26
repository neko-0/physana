import ROOT
from collinearw import ConfigMgr, run_PlotMaker, histManipulate, PlotMaker
from collinearw.serialization import Serialization
from collinearw.histMaker import weight_from_hist
from collinearw.strategies.systematics.core import (
    compute_quadrature_sum,
    compute_systematics,
    compute_process_set_systematics,
)
from collinearw.backends.root import RootBackend
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

# RootBackend.ATLAS_LABELS_STATUS = "Preliminary"
RootBackend.ATLAS_LABELS_STATUS = None

PlotMaker.HIGHT_LIGHT_SIGNAL_COLOR = {}

def rename(config):
    for process in config.processes:
        for region in process:
            for h in region:
                if h.name == "DeltaRLepJetClosest100":
                    h.xtitle = "#DeltaR_{min}(l,jet^{100}_{i})"
                if h.xtitle == "Leading W p_{T} [GeV]":
                    # h.xtitle = "p_{T}(lepton,E_{T}^{miss}) [GeV]"
                    h.xtitle = "p^{l\\nu}_{T} [GeV]"
                if h.xtitle == "W pT/closest jet pT [GeV]":
                    # h.xtitle = "p_{T}(lepton,E_{T}^{miss}) / p_{T}^{closest jet}"
                    h.xtitle = "p^{l#nu}_{T}/p_{T}^{closest jet}"
                 
def main():

    ytitles = {
        "wPt" : "Events/GeV",
        # "wPt/DeltaPhiWJetClosestPt100" : "dN/dp_{T}(lepton,E_{T}^{miss}) / p_{T}^{closest jet}",
        "wPt/DeltaPhiWJetClosestPt100" : "Events/0.01",
        "jet1Pt" : "Events/GeV",
        "DeltaRLepJetClosest100" : "Events/0.18",
        "Ht30" : "Events/GeV",
    }

    mc_list = [
        "wjets_2211",
        "zjets_2211",
        "ttbar",
        "singletop",
        "diboson_powheg",
        "wjets_tau",
        "vgamma_mg_W",
        "vgamma_Z",
        "dijets",
        "wjets_EW_2211",
    ]

    mc_list_2 = [
        "wjets_2211",
        "zjets_2211",
        "ttbar",
        "singletop",
        "diboson_powheg",
        "wjets_tau",
        "vgamma",
        "dijets",
        "wjets_EW_2211",
    ]

    if True:
        run2 = ConfigMgr.open(
            #"/gpfs/slac/atlas/fs1/d/yuzhan/configs_bk/ColWJet_2022/v6b/run2_MC_driven_Aug2/reco_band_full_syst.pkl"
            #"/gpfs/slac/atlas/fs1/d/yuzhan/configs_bk/ColWJet_2022/v6b/run2_MC_driven_Aug2/filter_reco_band_full_syst.pkl" #filter_reco_band_full_syst.pkl"
            #"/gpfs/slac/atlas/fs1/u/mgignac/analysis/collinearw/dev/collinearw/configs/studies/wjets_2211_fake_iter1.pkl"
            #"/gpfs/slac/atlas/fs1/d/yuzhan/configs_bk/ColWJet_2022/v6b/run2_MC_driven_Sep14/filter_reco_band_full_syst.pkl" # pT>500 GeV no MET
            #"/gpfs/slac/atlas/fs1/d/yuzhan/configs_bk/ColWJet_2022/v6b/run2_MC_driven_Sep14_pt800/filter_reco_band_full_syst.pkl"
            #"/gpfs/slac/atlas/fs1/d/yuzhan/configs_bk/ColWJet_2022/v6b/run2_MC_driven_Aug2_ABCD/fakes_2211_nominal_corr.pkl"
            #"/gpfs/slac/atlas/fs1/d/yuzhan/configs_bk/ColWJet_2022/v6b/run2_MC_driven_Jun16_noBjetVeto/merged_run2_full_syst_June15.pkl"
            #"/gpfs/slac/atlas/fs1/d/yuzhan/configs_bk/ColWJet_2022/v6b/run2_MC_driven_2023_Jan8/filter_reco_band_full_syst.pkl"
            #"/gpfs/slac/atlas/fs1/d/yuzhan/configs_bk/ColWJet_2022/v6b/run2_MC_driven_2023_Jan8/filter_reco_band_full_syst.pkl"
            #"/gpfs/slac/atlas/fs1/d/yuzhan/configs_bk/ColWJet_2022/v6b/run2_MC_driven_2023_Jan8/filter_reco_band_full_syst_mask_Jet_EtaCali.pkl"
            #"/gpfs/slac/atlas/fs1/d/yuzhan/configs_bk/ColWJet_2022/v6b/run2_MC_driven_2023_Jan8/filter_reco_band_full_syst_new_vgamma_update_zjets.pkl" 
            #"/gpfs/slac/atlas/fs1/d/yuzhan/configs_bk/ColWJet_2022/v6b/run2_MC_driven_2024_March4/reco_band_split_merged.pkl"
            #"/gpfs/slac/atlas/fs1/d/yuzhan/configs_bk/ColWJet_2022/v6b/run2_MC_driven_2024_March4/reco_band_run2_prefit.pkl"
            #"/gpfs/slac/atlas/fs1/d/yuzhan/configs_bk/ColWJet_2022/v6b/run2_MC_driven_2024_March4/reco_band_split_merged_March8.pkl"
            #"/fs/ddn/sdf/group/atlas/d/yuzhan/configs_bk/ColWJet_2022/v6b/run2_MC_driven_2024_April8/run2_nominal.pkl"
            #"/fs/ddn/sdf/group/atlas/d/yuzhan/configs_bk/ColWJet_2022/v6b/run2_MC_driven_2024_April8/run2_nominal.pkl"
            #"/fs/ddn/sdf/group/atlas/d/yuzhan/configs_bk/ColWJet_2022/v6b/run2_MC_driven_2024_April8/iterative_sf/run2.pkl"
            "/fs/ddn/sdf/group/atlas/d/yuzhan/configs_bk/ColWJet_2022/v6b/run2_MC_driven_2024_April8/merged_reco_band.pkl"

       )  

        # sum_rules = {"W+jets"   : {"processes": ["wjets_2211","wjets_tau","wjets_EW_2211"], "color": ROOT.TColor.GetColor("#ffffff") },
        sum_rules = {"W+jets"   : {"processes": ["wjets_2211","wjets_tau","wjets_EW_2211"], "color": ROOT.TColor.GetColor("#de77ae") },
                     "Z+jets"   : {"processes": ["zjets_2211"], "color": ROOT.TColor.GetColor("#d73027") },
                     "Top"      : {"processes": ["ttbar", "singletop"], "color": ROOT.TColor.GetColor("#fdae61")},
#                     "V+#gamma" : {"processes": ["vgamma_Z","vgamma_mg_W"], "color": ROOT.TColor.GetColor("#4575b4")},
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
    run2.save("temp_file.pkl")
    run2 = ConfigMgr.open("temp_file.pkl")

    rename(run2)

    # Systematic band
    external_syst_process = run2.get_process("sum(wjets_2211,zjets_2211,ttbar,singletop,diboson_powheg,wjets_tau,dijets,wjets_EW_2211)") 
    #sum(wjets_2211,zjets_2211,ttbar,singletop,diboson_powheg,wjets_tau,dijets,wjets_EW_2211,vgamma)") #f"sum({','.join(mc_list_2)})")

    # run_PlotMaker.run_stack(
    #     run2,
    #     "reco_plot_paper_Wmunu_SR",
    #     # data="with inclusive TF",
    #     # mcs=[f"with collinear TF"],
    #     data='data',
    #     mcs=mc_list,  # e.g. ['Top']
    #     low_yrange=(0.3, 1.7),
    #     # yrange=(10, 5e6),
    #     yrange=(1e2, 5e7),
    #     # yrange=(1e-1, 5e5),
    #     logy=True,
    #     workers=None,
    #     #rname_filter=["*ttbar*","*Zjets*","*dijets*"],
    #     # rname_filter=["*inclusive*fake-MU-rA*"],
    #     # rname_filter=["*inclusive*fake-*-rA*"],
    #     rname_filter=["*inclusive*fake-*-rA*"],
    #     # rname_filter=["*collinear*fake-MU-rA*"],
    #     # check_region=True,
    #     # low_ytitle="Col./Inc.",
    #     low_ytitle="Data/Pred",
    #     #systematic=('ttbar_MUR_MUF_Scale', 'NoSys', 'LHE3Weight_MUR0p5_MUF0p5_PDF303200_PSMUR0p5_PSMUF0p5'),
    #     include_systematic_band=True,
    #     # lookup_systematic_groups = lookup_systematic_groups,
    #     # compute_systematic_groups = compute_systematic_groups,
    #     external_syst_process = external_syst_process,
    #     hide_process=False,
    #     #do_sort=False,
    #     legend_opt="",
    #     enable_legend=True,
    #     legend_pos=(0.55, 0.70, 0.85, 0.89),
    #     #text="#splitline{W #rightarrow #mu#nu + #geq 1 jet}{p_{T}^{j1} #geq 500 GeV}",
    #     # text="\\splitline{W \\rightarrow \\mu\\nu + \\geq 1 jet}{p_{T}^{j1} \\geq 500 GeV, \\Delta{R}_{\\text{min}}\\left(lepton,\\text{jet}^{100}_{i}\\right) \\leq 2.6}",
    #     # text="\\splitline{W \\rightarrow \\mu\\nu + \\geq 1 jet}{p_{T}^{j1} \\geq 500 GeV, \\DeltaR_{min}(lepton,jet^{100}_{i}) \\leq 2.6}",
    #     text="\\splitline{W \\rightarrow \\mu\\nu + \\geq 1 jet}{p_{T}^{j1} \\geq 500 GeV}",
    #     # text="\\splitline{W \\rightarrow e\\nu + \\geq 1 jet}{p_{T}^{j1} \\geq 500 GeV}",
    #     # text="\\splitline{W \\rightarrow e\\nu + \\geq 1 jet}{p_{T}^{j1} \\geq 500 GeV, \\DeltaR_{min}(lepton,jet^{100}_{i}) \\leq 2.6}",
    #     # text=[
    #     #     {"text_content" : "W \\rightarrow \\mu\\nu + \\geq 1 jet", "x" : 0.20, "y":0.77, "color":1, "size":30},
    #     #     {"text_content" : "p_{T}^{j1} \\geq 500 GeV, \\DeltaR_{min}(\\mu, jet^{100}_{i}) \\leq 2.6", "x" : 0.20, "y":0.72, "color":1, "size":23},
    #     # ],
    #     #text="#splitline{W #rightarrow e#nu + #geq 1 jet}{p_{T}^{j1} #geq 500 GeV, #DeltaR(lepton,jet_{i}^{100}) #leq 2.6}",
    #     #text="#splitline{#splitline{t#bar{t} control region}{1 muon + #geq 1 #it{b}-jet}}{p_{T}^{j1} #geq 500 GeV}",
    #     #text="#splitline{#splitline{Z+jets control region}{#mu^{#pm}#mu^{#mp} + 60<m_{#mu#mu}<120 GeV}}{p_{T}^{j1} #geq 500 GeV}",
    #     #text="#splitline{#splitline{Multi-jet control region}{1 electron, inverted signal criteria}}{p_{T}^{j1} #geq 500 GeV}",
    #     #text="#splitline{#splitline{Multi-jet validation region}{1 electron, E_{T}^{miss}<100 GeV}}{p_{T}^{j1} #geq 500 GeV}",
    #     show_text=True,
    #     divide_binwidth=True,
    #     ytitles=ytitles,
    #     figfmt="pdf",
    # )
    
    fmt = "pdf"

    run2.get_process("data").title = "Data, stat. unc."

    run_PlotMaker.run_stack(
        run2,
        "reco_plot_paper_Wmunu_SR_inclusive",
        data='data',
        mcs=mc_list,  # e.g. ['Top']
        low_yrange=(0.3, 1.7),
        yrange=(1e2, 5e7),
        logy=True,
        workers=None,
        rname_filter=["*inclusive*fake-MU-rA*"],
        low_ytitle="Data/Pred.",
        include_systematic_band=True,
        external_syst_process = external_syst_process,
        hide_process=False,
        legend_opt="",
        enable_legend=True,
        legend_pos=(0.55, 0.70, 0.85, 0.89),
        text=[
            {"text_content" : "\\splitline{W \\rightarrow \\mu\\nu + \\geq 1 jet}{p_{T}^{j1} \\geq 500 GeV}", "x" : 0.185, "y":0.75, "color":1, "size":30},
        ],
        show_text=True,
        divide_binwidth=True,
        ytitles=ytitles,
        figfmt=fmt,
    )

    run_PlotMaker.run_stack(
        run2,
        "reco_plot_paper_Wenu_SR_inclusive",
        data='data',
        mcs=mc_list,  # e.g. ['Top']
        low_yrange=(0.3, 1.7),
        yrange=(1e2, 5e7),
        logy=True,
        workers=None,
        rname_filter=["*inclusive*fake-EL-rA*"],
        low_ytitle="Data/Pred.",
        include_systematic_band=True,
        external_syst_process = external_syst_process,
        hide_process=False,
        legend_opt="",
        enable_legend=True,
        legend_pos=(0.55, 0.70, 0.85, 0.89),
        text=[
            {"text_content" : "\\splitline{W \\rightarrow e\\nu + \\geq 1 jet}{p_{T}^{j1} \\geq 500 GeV}", "x" : 0.185, "y":0.75, "color":1, "size":30},
        ],
        show_text=True,
        divide_binwidth=True,
        ytitles=ytitles,
        figfmt=fmt,
    )

    run_PlotMaker.run_stack(
        run2,
        "reco_plot_paper_Wmunu_SR_collinear",
        data='data',
        mcs=mc_list,  # e.g. ['Top']
        low_yrange=(0.3, 1.7),
        yrange=(1e-1, 5e5),
        logy=True,
        workers=None,
        rname_filter=["*collinear*fake-MU-rA*"],
        low_ytitle="Data/Pred.",
        include_systematic_band=True,
        external_syst_process = external_syst_process,
        hide_process=False,
        legend_opt="",
        enable_legend=True,
        legend_pos=(0.55, 0.70, 0.85, 0.89),
        text=[
            {"text_content" : "W \\rightarrow \\mu\\nu + \\geq 1 jet", "x" : 0.185, "y":0.77, "color":1, "size":30},
            {"text_content" : "p_{T}^{j1} \\geq 500 GeV, \\DeltaR_{min}(\\mu, jet^{100}_{i}) \\leq 2.6", "x" : 0.185, "y":0.72, "color":1, "size":23},
        ],
        show_text=True,
        divide_binwidth=True,
        ytitles=ytitles,
        figfmt=fmt,
    )

    run_PlotMaker.run_stack(
        run2,
        "reco_plot_paper_Wenu_SR_collinear",
        data='data',
        mcs=mc_list,  # e.g. ['Top']
        low_yrange=(0.3, 1.7),
        yrange=(1e-1, 5e5),
        logy=True,
        workers=None,
        rname_filter=["*collinear*fake-EL-rA*"],
        low_ytitle="Data/Pred.",
        include_systematic_band=True,
        external_syst_process = external_syst_process,
        hide_process=False,
        legend_opt="",
        enable_legend=True,
        legend_pos=(0.55, 0.70, 0.85, 0.89),
        text=[
            {"text_content" : "W \\rightarrow e\\nu + \\geq 1 jet", "x" : 0.185, "y":0.77, "color":1, "size":30},
            {"text_content" : "p_{T}^{j1} \\geq 500 GeV, \\DeltaR_{min}(e, jet^{100}_{i}) \\leq 2.6", "x" : 0.185, "y":0.72, "color":1, "size":23},
        ],
        show_text=True,
        divide_binwidth=True,
        ytitles=ytitles,
        figfmt=fmt,
    )

    run_PlotMaker.run_stack(
        run2,
        "reco_plot_paper_CR",
        data='data',
        mcs=mc_list,  # e.g. ['Top']
        low_yrange=(0.3, 1.7),
        yrange=(10, 5e6),
        logy=True,
        workers=None,
        rname_filter=["*ttbar*"],
        low_ytitle="Data/Pred.",
        include_systematic_band=True,
        external_syst_process = external_syst_process,
        hide_process=False,
        legend_opt="",
        enable_legend=True,
        legend_pos=(0.55, 0.70, 0.85, 0.89),
        text=[
            {"text_content" : "t#bar{t} control region",  "x" : 0.185, "y":0.79, "color":1, "size":23},
            {"text_content" : "1 muon + #geq 1 #it{b}-jet",  "x" : 0.185, "y":0.75, "color":1, "size":23},
            {"text_content" : "p_{T}^{j1} #geq 500 GeV",  "x" : 0.185, "y":0.71, "color":1, "size":23},
        ],  
        show_text=True,
        divide_binwidth=True,
        ytitles=ytitles,
        figfmt=fmt,
    )

    run_PlotMaker.run_stack(
        run2,
        "reco_plot_paper_CR",
        data='data',
        mcs=mc_list,  # e.g. ['Top']
        low_yrange=(0.3, 1.7),
        yrange=(1e-1, 5e3),
        logy=True,
        workers=None,
        rname_filter=["*Zjets*"],
        low_ytitle="Data/Pred.",
        include_systematic_band=True,
        external_syst_process = external_syst_process,
        hide_process=False,
        legend_opt="",
        enable_legend=True,
        legend_pos=(0.55, 0.70, 0.85, 0.89),
        text=[
            {"text_content" : "Z+jets control region",  "x" : 0.185, "y":0.79, "color":1, "size":23},
            {"text_content" : "#mu^{#pm}#mu^{#mp} + 60<m_{#mu#mu}<120 GeV",  "x" : 0.185, "y":0.75, "color":1, "size":23},
            {"text_content" : "p_{T}^{j1} #geq 500 GeV",  "x" : 0.185, "y":0.71, "color":1, "size":23},
        ],  
        show_text=True,
        divide_binwidth=True,
        ytitles=ytitles,
        figfmt=fmt,
    )

    run_PlotMaker.run_stack(
        run2,
        "reco_plot_paper_CR",
        data='data',
        mcs=mc_list,  # e.g. ['Top']
        low_yrange=(0.3, 1.7),
        yrange=(1e-1, 5e4),
        logy=True,
        workers=None,
        rname_filter=["*dijets*"],
        low_ytitle="Data/Pred.",
        include_systematic_band=True,
        external_syst_process = external_syst_process,
        hide_process=False,
        legend_opt="",
        enable_legend=True,
        legend_pos=(0.55, 0.70, 0.85, 0.89),
        text=[
            {"text_content" : "Multi-jet control region",  "x" : 0.185, "y":0.79, "color":1, "size":23},
            {"text_content" : "1 electron, inverted signal criteria",  "x" : 0.185, "y":0.75, "color":1, "size":23},
            {"text_content" : "p_{T}^{j1} #geq 500 GeV",  "x" : 0.185, "y":0.71, "color":1, "size":23},
        ],  
        show_text=True,
        divide_binwidth=True,
        ytitles=ytitles,
        figfmt=fmt,
    )

    run_PlotMaker.run_stack(
        run2,
        "reco_plot_paper_CR",
        data='data',
        mcs=mc_list,  # e.g. ['Top']
        low_yrange=(0.3, 1.7),
        yrange=(1e-1, 5e4),
        logy=True,
        workers=None,
        rname_filter=["*lowMET*"],
        low_ytitle="Data/Pred.",
        include_systematic_band=True,
        external_syst_process = external_syst_process,
        hide_process=False,
        legend_opt="",
        enable_legend=True,
        legend_pos=(0.55, 0.70, 0.85, 0.89),
        text=[
            {"text_content" : "Multi-jet validation region",  "x" : 0.185, "y":0.79, "color":1, "size":23},
            {"text_content" : "1 electron, E_{T}^{miss}<100 GeV",  "x" : 0.185, "y":0.75, "color":1, "size":23},
            {"text_content" : "p_{T}^{j1} #geq 500 GeV",  "x" : 0.185, "y":0.71, "color":1, "size":23},
        ],  
        show_text=True,
        divide_binwidth=True,
        ytitles=ytitles,
        figfmt=fmt, 
    )


if __name__ == "__main__":
    main()
