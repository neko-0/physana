import collinearw
from collinearw.strategies.systematics.core import compute_quadrature_sum, compute_systematics

import logging

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

def main():
    config = collinearw.ConfigMgr.open("unfold_run2_2211.pkl")

    jet_experiment = {}
    jet_experiment.update( config.generate_systematic_group("JET", ("*JET*", "*JET*", "")) )
    #group.update( config.generate_systematic_group("JET-JER", ("*JET*JER*", "*JET*", "")) )
    #group.update( config.generate_systematic_group("JET-EtaIntercalibration-NonClosure", ("*JET*Eta*", "*JET*", "")) )

    jet_experiment.update( config.generate_systematic_group("JVT", ("*jvtWeight_JET_JvtEfficiency*", "NoSys", "*")) )
    #group.update( config.generate_systematic_group("JVT-up", ("*jvtWeight_JET_JvtEfficiency*", "NoSys", "*up*")) )
    #group.update( config.generate_systematic_group("JVT-down", ("*jvtWeight_JET_JvtEfficiency*", "NoSys", "*down*")) )

    lepton_experiment = {}
    lepton_experiment.update( config.generate_systematic_group("Lepton-Weight", ("*leptonWeight*", "NoSys", "*leptonWeight*")) )
    #group.update( config.generate_systematic_group("Lepton-EL-Weight-up", ("*leptonWeight*EL*", "NoSys", "*leptonWeight*up*")) )
    #group.update( config.generate_systematic_group("Lepton-EL-Weight-down", ("*leptonWeight*EL*", "NoSys", "*leptonWeight*down*")) )
    #group.update( config.generate_systematic_group("Lepton-MU-Weight-up", ("*leptonWeight*MUON*", "NoSys", "*leptonWeight*up*")) )
    #group.update( config.generate_systematic_group("Lepton-MU-Weight-down", ("*leptonWeight*MUON*", "NoSys", "*leptonWeight*down*")) )


    lepton_experiment.update( config.generate_systematic_group("TrigWeight", ("*trigWeight*", "NoSys", "*")) )
    #group.update( config.generate_systematic_group("Trig-EL-up", ("*trigWeight_EL*_*", "NoSys", "*up*")) )
    #group.update( config.generate_systematic_group("Trig-EL-down", ("*trigWeight_EL*", "NoSys", "*down*")) )
    #group.update( config.generate_systematic_group("Trig-MUON-up", ("*trigWeight_MUON*_*", "NoSys", "*up*")) )
    #group.update( config.generate_systematic_group("Trig-MUON-down", ("*trigWeight_MUON*", "NoSys", "*down*")) )

    btagging = {}
    btagging.update( config.generate_systematic_group("bTagWeight", ("*bTagWeight*", "NoSys", "*")) )
    #group.update( config.generate_systematic_group("bTagWeight-up", ("*bTagWeight*", "NoSys", "*up*")) )
    #group.update( config.generate_systematic_group("bTagWeight-down", ("*bTagWeight*", "NoSys", "*down*")) )

    # computing min max of scaling uncertainties
    compute_systematics(config, "wjets_2211_MUR_MUF_Scale", "min_max")
    compute_systematics(config, "wjets_2211_NNPDF30nnlo_hessian", "hessian")
    compute_systematics(config, "zjets_2211_MUR_MUF_Scale", "min_max")
    compute_systematics(config, "zjets_2211_NNPDF30nnlo_hessian", "hessian")
    compute_systematics(config, "diboson_MUR_MUF_Scale", "min_max")
    compute_systematics(config, "diboson_NNPDF30nnlo_hessian", "hessian")
    compute_systematics(config, "diboson_powheg_PDF", "stdev")
    compute_systematics(config, "ttbar_ren_fac_scale", "min_max")
    compute_systematics(config, "ttbar_ISR_scale", "min_max")
    compute_systematics(config, "ttbar_FSR_scale", "min_max")
    compute_systematics(config, "ttbar_NNPDF30_PDF", "stdev")
    compute_systematics(config, "singletop_ren_fac_scale", "min_max")

    fakes_theory = {}
    bkgd_theory = {}

    ttbar_theory = {}
    ttbar_theory.update( config.generate_systematic_group("ttbar-ren_fac_scale", ("*ttbar*ren_fac_scale*", "min_max", "*")) )
    ttbar_theory.update( config.generate_systematic_group("ttbar-ISR_scale", ("*ttbar*ISR_scale*", "min_max", "*")) )
    ttbar_theory.update( config.generate_systematic_group("ttbar-FSR_scale", ("*ttbar*FSR_scale*", "min_max", "*")) )
    ttbar_theory.update( config.generate_systematic_group("ttbar-NNPDF30_PDF", ("*ttbar*NNPDF30_PDF*", "stdev", "*")) )

    #ttbar_theory.update( config.generate_systematic_group("ttbar-ren_fac_scale-up", ("*ttbar*ren_fac_scale*", "min_max", "max")) )
    #ttbar_theory.update( config.generate_systematic_group("ttbar-ren_fac_scale-down", ("*ttbar*ren_fac_scale*", "min_max", "min")) )
    #ttbar_theory.update( config.generate_systematic_group("ttbar-ISR_scale-up", ("*ttbar*ISR_scale*", "min_max", "max")) )
    #ttbar_theory.update( config.generate_systematic_group("ttbar-ISR_scale-down", ("*ttbar*ISR_scale*", "min_max", "min")) )
    #ttbar_theory.update( config.generate_systematic_group("ttbar-FSR_scale-up", ("*ttbar*FSR_scale*", "min_max", "max")) )
    #ttbar_theory.update( config.generate_systematic_group("ttbar-FSR_scale-down", ("*ttbar*FSR_scale*", "min_max", "min")) )
    #ttbar_theory.update( config.generate_systematic_group("ttbar-NNPDF30_PDF-up", ("*ttbar*NNPDF30_PDF*", "stdev", "std_up")) )
    #ttbar_theory.update( config.generate_systematic_group("ttbar-NNPDF30_PDF-down", ("*ttbar*NNPDF30_PDF*", "stdev", "std_down")) )

    singletop_theory = {}
    singletop_theory.update( config.generate_systematic_group("singletop-ren_fac_scale", ("*singletop_ren_fac_scale*", "min_max", "*")) )
    ##singletop_theory.update( config.generate_systematic_group("singletop-ren_fac_scale-up", ("*singletop_ren_fac_scale*", "min_max", "max")) )
    #singletop_theory.update( config.generate_systematic_group("singletop-ren_fac_scale-down", ("*singletop_ren_fac_scale*", "min_max", "min")) )

    zjets_theory = {}
    zjets_theory.update( config.generate_systematic_group("zjets_2211-MUR_MUF", ("*zjets*", "min_max", "*")) )
    zjets_theory.update( config.generate_systematic_group("zjets_2211-NNPDF30nnlo_hessian", ("*zjets*", "hessian", "*")) )
    #zjets_theory.update( config.generate_systematic_group("zjets_2211-MUR_MUF-up", ("*zjets*", "min_max", "max")) )
    #zjets_theory.update( config.generate_systematic_group("zjets_2211-MUR_MUF-down", ("*zjets*", "min_max", "min")) )
    #zjets_theory.update( config.generate_systematic_group("zjets_2211-NNPDF30nnlo_hessian-up", ("*zjets*", "stdev", "std_up")) )
    #zjets_theory.update( config.generate_systematic_group("zjets_2211-NNPDF30nnlo_hessian-down", ("*zjets*", "stdev", "std_down")) )

    diboson_theory = {}
    diboson_theory.update( config.generate_systematic_group("diboson-MUR_MUF", ("*diboson*", "min_max", "*")) )
    diboson_theory.update( config.generate_systematic_group("diboson-NNPDF30nnlo_hessian", ("*diboson*", "hessian", "*")) )
    diboson_theory.update( config.generate_systematic_group("diboson_powheg-PDF", ("*diboson*", "stdev", "*")) )
    #diboson_theory.update( config.generate_systematic_group("diboson-MUR_MUF-up", ("*diboson*", "min_max", "max")) )
    #diboson_theory.update( config.generate_systematic_group("diboson-MUR_MUF-down", ("*diboson*", "min_max", "min")) )
    #diboson_theory.update( config.generate_systematic_group("diboson-NNPDF30nnlo_hessian-up", ("*diboson*", "stdev", "std_up")) )
    #diboson_theory.update( config.generate_systematic_group("diboson-NNPDF30nnlo_hessian-down", ("*diboson*", "stdev", "std_down")) )

    wjets_theory = {}
    wjets_theory.update( config.generate_systematic_group("wjets_2211-MUR_MUF", ("*wjets*", "min_max", "*")) )
    wjets_theory.update( config.generate_systematic_group("wjets_2211-NNPDF30nnlo_hessian", ("*wjets*", "hessian", "*")) )
    #wjets_theory.update( config.generate_systematic_group("wjets_2211-MUR_MUF-up", ("*wjets*", "min_max", "max")) )
    #wjets_theory.update( config.generate_systematic_group("wjets_2211-MUR_MUF-down", ("*wjets*", "min_max", "min")) )
    #wjets_theory.update( config.generate_systematic_group("wjets_2211-NNPDF30nnlo_hessian-up", ("*wjets*", "stdev", "std_up")) )
    #wjets_theory.update( config.generate_systematic_group("wjets_2211-NNPDF30nnlo_hessian-down", ("*wjets*", "stdev", "std_down")) )

    fakes_theory.update(ttbar_theory)
    fakes_theory.update(singletop_theory)
    fakes_theory.update(wjets_theory)
    fakes_theory.update(zjets_theory)

    bkgd_theory.update(ttbar_theory)
    bkgd_theory.update(singletop_theory)
    bkgd_theory.update(zjets_theory)
    bkgd_theory.update(diboson_theory)

    other_theory = {}
    other_theory.update(zjets_theory)
    other_theory.update(ttbar_theory)
    other_theory.update(diboson_theory)
    other_theory.update(singletop_theory)

    theories = {
        "wjets" : wjets_theory,
        "zjets" : zjets_theory,
        "ttbar" : ttbar_theory,
        "singletop" : singletop_theory,
        "diboson" : diboson_theory,
        "fakes" : fakes_theory,
        "bkgd" : bkgd_theory,
        "other" : other_theory,
    }

    plist = ["wjets_2211", "unfold_closure", "unfold_realthang_fake-MU", "unfold_realthang_fake-EL"]

    for process_name in config.list_processes():
        syst_type = "experimental"
        for name, syst_list in jet_experiment.items():
            print(f"Doing process {process_name}, group {name}")
            compute_quadrature_sum(config, process_name, "Jet", syst_type, syst_list, sub_band_name=name, store_process=False)

        for name, syst_list in btagging.items():
            print(f"Doing process {process_name}, group {name}")
            compute_quadrature_sum(config, process_name, "B-tagging", syst_type, syst_list, sub_band_name=name, store_process=False)

        for name, syst_list in lepton_experiment.items():
            print(f"Doing process {process_name}, group {name}")
            compute_quadrature_sum(config, process_name, "Lepton", syst_type, syst_list, sub_band_name=name, store_process=False)


        if "wjets" in process_name:
            theory = theories["wjets"]
        elif "zjets" in process_name:
            theory = theories["zjets"]
        elif "ttbar" in process_name:
            theory = theories["ttbar"]
        elif "singletop" in process_name:
            theory = theories["singletop"]
        elif "fakes" in process_name:
            theory = theories["fakes"]
        elif "measured" in process_name:
            theory = theories["bkgd"]
        elif "closure" in process_name:
            theory = theories["other"]
        else:
            theory = theories["other"]

        for name, syst_list in theory.items():
            syst_type = "theory"
            print(f"Doing process {process_name}, group {name}")
            if "wjets" in name:
                #theory_legend_name = "W+jets theory"
                if "PDF" in name:
                    theory_legend_name = "NNPDF30 NNLO"
                if "MUR_MUF" in name:
                    theory_legend_name = "Scale(#muR,#muF)"
            elif "zjets" in name:
                theory_legend_name = "Z+jets theory"
            elif "ttbar" in name:
                theory_legend_name = "t#bar{t} theory"
            else:
                theory_legend_name = "others"
            compute_quadrature_sum(config, process_name, theory_legend_name, syst_type, syst_list, sub_band_name=name, store_process=False)

    config.save("band_unfold.pkl")


def plot_things():
    import ROOT
    import pathlib

    config = collinearw.ConfigMgr.open("band_unfoled.pkl")

    plotmaker = collinearw.PlotMaker(config, "check_unfold_plots")

    unfold_process = config.get_process("unfold_realthang_PID-fake-electron")
    wjets_2211 = config.get_process("wjets_2211")

    for r in unfold_process.regions:
        for obs in r.histograms:
            canvas = plotmaker.make_canvas()
            wjets_obs = wjets_2211.get_region(r.name).get_observable(obs.name)
            wjets_theory = wjets_obs.scale_band("theory")

            exp_band = obs.scale_band("experimental")
            theory_band = obs.scale_band("theory")

            wjets_band = wjets_obs.scale_band("theory")

            obs_h = obs.root
            wjets_h = wjets_obs.root

            obs_exp_g = ROOT.TGraphAsymmErrors(obs.root)
            obs_exp_g.SetLineColor(ROOT.kRed)
            obs_exp_g.SetMarkerColor(ROOT.kBlack)
            #obs_exp_g.SetFillColor(ROOT.kRed)
            #obs_exp_g.SetFillStyle(3005)
            for i, (up, down) in enumerate(zip(exp_band["up"], exp_band["down"])):
                if i == 0 or i == len(exp_band["up"])-1:
                    continue
                obs_exp_g.SetPointError(i-1, 0, 0, down, up)

            obs_theory_g = ROOT.TGraphAsymmErrors(obs.root)
            obs_theory_g.SetLineColor(ROOT.kGreen)
            obs_theory_g.SetMarkerColor(ROOT.kBlack)
            #obs_theory_g.SetFillColor(ROOT.kGreen)
            #obs_theory_g.SetFillStyle(3005)
            for i, (up, down) in enumerate(zip(theory_band["up"], theory_band["down"])):
                if i == 0 or i == len(theory_band["up"])-1:
                    continue
                obs_theory_g.SetPointError(i-1, 0.01, 0.01, down, up)

            wjets_theory_g = ROOT.TGraphAsymmErrors(wjets_obs.root)
            #wjets_theory_g.SetLineColor(ROOT.kBlue)
            #wjets_theory_g.SetMarkerColor(ROOT.kBlue)
            #wjets_theory_g.SetMarkerStyle(8)
            #wjets_theory_g.SetMarkerSize(1)
            #wjets_theory_g.SetLineWidth(2)
            wjets_theory_g.SetFillColor(ROOT.kBlue)
            wjets_theory_g.SetFillStyle(3005)
            for i, (up, down) in enumerate(zip(wjets_band["up"], wjets_band["down"])):
                if i == 0 or i == len(wjets_band["up"])-1:
                    continue
                wjets_theory_g.SetPointError(i-1, 0.5, 0.5, down, up)

            obs_h.SetLineColor(ROOT.kBlack)
            obs_h.SetLineWidth(3)
            obs_h.Draw()
            # wjets_h.Draw("same")
            wjets_theory_g.Draw("2")
            obs_exp_g.Draw("p E1")
            obs_theory_g.Draw("p E1")


            legend = plotmaker.make_legend()
            legend.AddEntry(obs_h, "data")
            legend.AddEntry(obs_exp_g, "experimental", "pl")
            legend.AddEntry(obs_theory_g, "theory", "pl")
            legend.AddEntry(wjets_theory_g, "wjet with theory", "f")
            legend.Draw()

            save_path = pathlib.Path(f"check_unfold_plots/{r.name}/{obs.name}.png")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            canvas.SaveAs(f"{save_path.resolve()}")

if __name__ == "__main__":
     main()
    #plot_things()
