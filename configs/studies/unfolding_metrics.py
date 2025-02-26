import collinearw

from collinearw.strategies import unfolding
from collinearw.backends.root import RootBackend


# RootBackend.ATLAS_LABELS_STATUS = "Preliminary"
RootBackend.ATLAS_LABELS_STATUS = None
RootBackend.SHOW_LUMI_LABEL = False

def rename(config):
    rename_map = {
        # "jet1TruthPt" : "Leading jet p_{T} [GeV]",
        # "nTruthJet30" : "Inclusive jet multiplicity",
        # "wTruthPt" : "\\text{p}^{\\ell\\nu}_{\\text{T}}\\;\\text{[GeV]}",
        # "HtTruth30" : "S_{T} [GeV]",
        # "DeltaRTruthLepJetClosest100" : "\\Delta{R}_{\\text{min}}\\left(,\\text{jet}^{100}_{i}\\right)",
        "response_matrix_DeltaRLepJetClosest100" : "\\Delta{R}_{\\text{min}}\\left(\\ell,\\text{jet}^{100}_{i}\\right)",
        # "response_matrix_DeltaRLepJetClosest100" : "#DeltaR_{min}(ll,jet^{100}_{i})",
        # "wTruthPt/DeltaPhiTruthWJetClosestPt100" : "\\text{p}^{\\ell\\nu}_{\\text{T}}/\\text{p}_{\\text{T}}^{\\text{closest jet}}",
    }
    for process in config.processes:
        for region in process:
            for h in region:
                # h.xtitle = h.xtitle.replace("Particle level ", "")
                # h.ytitle = h.ytitle.replace("Particle level ", "")
                if h.name in rename_map:
                    h.xtitle = h.xtitle.replace("Particle level ", "")
                    h.ytitle = h.ytitle.replace("Particle level ", "")
                    h.xtitle = rename_map[h.name]
                    h.ytitle = "\\text{Particle level }" + rename_map[h.name]

def main():

    config = collinearw.ConfigMgr.open(
        "/fs/ddn/sdf/group/atlas/d/yuzhan/configs_bk/ColWJet_2022/v6b/run2_MC_driven_2024_April8/run2_nominal_metric_Jul8.pkl"
    )
    config.remove_process("wjets")

    name_map = {
        "wjets_FxFx": "MadGraph+Pythia8 FxFx",
        "wjets_2211": "Sherpa 2.2.11 NLO QCD",
        "wjets_2211_ASSEW": "Sherpa 2.2.11 NLO QCD+EW",
    }

    rename(config)

    # collinearw.run_PlotMaker.plot_purity(config, signal_list = ["wjets_2211","wjets_2211_ASSEW","wjets_FxFx"], name_map=name_map, plot_response=False)

    # collinearw.run_PlotMaker.plot_purity(config, signal_list = ["wjets_2211"], name_map=name_map, plot_response=True)

    unfolding.plot.make_eff_plots(config, name_map=name_map, phasespace="collinear", text_label=["Muon","Collinear region"], fmt="ps")


if __name__ == "__main__":
    main()
