import collinearw
from collinearw.strategies.systematics.core import (
    compute_quadrature_sum,
    compute_systematics,
)


def main():
    config = collinearw.ConfigMgr.open("unfold.pkl")

    group = {}
    group.update(
        config.generate_systematic_group(
            "JET-GroupedNP-up", ("*JET_GroupedNP*", "*JET*up*", "")
        )
    )
    group.update(
        config.generate_systematic_group(
            "JET-GroupedNP-down", ("*JET_GroupedNP*", "*JET*down*", "")
        )
    )

    group.update(
        config.generate_systematic_group("JET-JER-up", ("*JET*JER*", "*JET*up*", ""))
    )
    group.update(
        config.generate_systematic_group(
            "JET-JER-down", ("*JET*JER*", "*JET*down*", "")
        )
    )

    group.update(
        config.generate_systematic_group(
            "JET-EtaIntercalibration-NonClosure-up", ("*JET*Eta*", "*JET*up*", "")
        )
    )
    group.update(
        config.generate_systematic_group(
            "JET-EtaIntercalibration-NonClosure-down", ("*JET*Eta*", "*JET*down*", "")
        )
    )

    group.update(
        config.generate_systematic_group(
            "Lepton-EL-Weight-up", ("*leptonWeight*EL*", "NoSys", "*leptonWeight*up*")
        )
    )
    group.update(
        config.generate_systematic_group(
            "Lepton-EL-Weight-down",
            ("*leptonWeight*EL*", "NoSys", "*leptonWeight*down*"),
        )
    )
    group.update(
        config.generate_systematic_group(
            "Lepton-MU-Weight-up", ("*leptonWeight*MUON*", "NoSys", "*leptonWeight*up*")
        )
    )
    group.update(
        config.generate_systematic_group(
            "Lepton-MU-Weight-down",
            ("*leptonWeight*MUON*", "NoSys", "*leptonWeight*down*"),
        )
    )

    group.update(
        config.generate_systematic_group(
            "JVT-up", ("*jvtWeight_JET_JvtEfficiency*", "NoSys", "*up*")
        )
    )
    group.update(
        config.generate_systematic_group(
            "JVT-down", ("*jvtWeight_JET_JvtEfficiency*", "NoSys", "*down*")
        )
    )

    group.update(
        config.generate_systematic_group(
            "Trig-EL-up", ("*trigWeight_EL*_*", "NoSys", "*up*")
        )
    )
    group.update(
        config.generate_systematic_group(
            "Trig-EL-down", ("*trigWeight_EL*", "NoSys", "*down*")
        )
    )

    group.update(
        config.generate_systematic_group(
            "Trig-MUON-up", ("*trigWeight_MUON*_*", "NoSys", "*up*")
        )
    )
    group.update(
        config.generate_systematic_group(
            "Trig-MUON-down", ("*trigWeight_MUON*", "NoSys", "*down*")
        )
    )

    group.update(
        config.generate_systematic_group(
            "bTagWeight-up", ("*bTagWeight*", "NoSys", "*up*")
        )
    )
    group.update(
        config.generate_systematic_group(
            "bTagWeight-down", ("*bTagWeight*", "NoSys", "*down*")
        )
    )

    # computing min max of scaling uncertainties
    compute_systematics(config, "wjets_2211_MUR_MUF_Scale", "min_max")
    compute_systematics(config, "wjets_2211_NNPDF30nnlo_hessian", "stdev")
    compute_systematics(config, "zjets_2211_MUR_MUF_Scale", "min_max")
    compute_systematics(config, "zjets_2211_NNPDF30nnlo_hessian", "stdev")
    compute_systematics(config, "ttbar_ren_fac_scale", "min_max")
    compute_systematics(config, "ttbar_ISR_scale", "min_max")
    compute_systematics(config, "ttbar_FSR_scale", "min_max")
    compute_systematics(config, "ttbar_NNPDF30_PDF", "stdev")
    compute_systematics(config, "singletop_ren_fac_scale", "min_max")

    theory = {}
    theory.update(
        config.generate_systematic_group(
            "ttbar-ren_fac_scale-up", ("*ttbar*ren_fac_scale*", "min_max", "max")
        )
    )
    theory.update(
        config.generate_systematic_group(
            "ttbar-ren_fac_scale-down", ("*ttbar*ren_fac_scale*", "min_max", "min")
        )
    )
    theory.update(
        config.generate_systematic_group(
            "ttbar-ISR_scale-up", ("*ttbar*ISR_scale*", "min_max", "max")
        )
    )
    theory.update(
        config.generate_systematic_group(
            "ttbar-ISR_scale-down", ("*ttbar*ISR_scale*", "min_max", "min")
        )
    )
    theory.update(
        config.generate_systematic_group(
            "ttbar-FSR_scale-up", ("*ttbar*FSR_scale*", "min_max", "max")
        )
    )
    theory.update(
        config.generate_systematic_group(
            "ttbar-FSR_scale-down", ("*ttbar*FSR_scale*", "min_max", "min")
        )
    )
    theory.update(
        config.generate_systematic_group(
            "ttbar-NNPDF30_PDF-up", ("*ttbar*NNPDF30_PDF*", "stdev", "std_up")
        )
    )
    theory.update(
        config.generate_systematic_group(
            "ttbar-NNPDF30_PDF-down", ("*ttbar*NNPDF30_PDF*", "stdev", "std_down")
        )
    )

    theory.update(
        config.generate_systematic_group(
            "singletop-ren_fac_scale-up",
            ("*singletop_ren_fac_scale*", "min_max", "max"),
        )
    )
    theory.update(
        config.generate_systematic_group(
            "singletop-ren_fac_scale-down",
            ("*singletop_ren_fac_scale*", "min_max", "min"),
        )
    )

    wjets_theory = {}
    wjets_theory.update(
        config.generate_systematic_group(
            "wjets_2211-MUR_MUF-up", ("*wjets*", "min_max", "max")
        )
    )
    wjets_theory.update(
        config.generate_systematic_group(
            "wjets_2211-MUR_MUF-down", ("*wjets*", "min_max", "min")
        )
    )
    wjets_theory.update(
        config.generate_systematic_group(
            "wjets_2211-NNPDF30nnlo_hessian-up", ("*wjets*", "stdev", "std_up")
        )
    )
    wjets_theory.update(
        config.generate_systematic_group(
            "wjets_2211-NNPDF30nnlo_hessian-down", ("*wjets*", "stdev", "std_down")
        )
    )

    for process_name in config.list_processes():
        if "wjets" not in process_name:
            for name, syst_list in group.items():
                print(f"Doing process {process_name}, group {name}")
                if "-up" in name:
                    name = name.replace("-up", "")
                    config = compute_quadrature_sum(
                        name,
                        "experimental",
                        config,
                        process_name,
                        syst_list,
                        "up",
                        copy=False,
                    )
                else:
                    name = name.replace("-down", "")
                    config = compute_quadrature_sum(
                        name,
                        "experimental",
                        config,
                        process_name,
                        syst_list,
                        "down",
                        copy=False,
                    )

            for name, syst_list in theory.items():
                print(f"Doing process {process_name}, group {name}")
                if "-up" in name:
                    name = name.replace("-up", "")
                    config = compute_quadrature_sum(
                        name,
                        "theory",
                        config,
                        process_name,
                        syst_list,
                        "up",
                        copy=False,
                    )
                else:
                    name = name.replace("-down", "")
                    config = compute_quadrature_sum(
                        name,
                        "theory",
                        config,
                        process_name,
                        syst_list,
                        "down",
                        copy=False,
                    )
        else:
            for name, syst_list in wjets_theory.items():
                print(f"Doing process {process_name}, group {name}")
                if "-up" in name:
                    name = name.replace("-up", "")
                    config = compute_quadrature_sum(
                        name,
                        "theory",
                        config,
                        process_name,
                        syst_list,
                        "up",
                        copy=False,
                    )
                else:
                    name = name.replace("-down", "")
                    config = compute_quadrature_sum(
                        name,
                        "theory",
                        config,
                        process_name,
                        syst_list,
                        "down",
                        copy=False,
                    )

    config.save("band_unfold.pkl")


def plot_things():
    import ROOT
    import pathlib

    config = collinearw.ConfigMgr.open("band_unfoled_v2.pkl")

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
            # obs_exp_g.SetFillColor(ROOT.kRed)
            # obs_exp_g.SetFillStyle(3005)
            for i, (up, down) in enumerate(zip(exp_band["up"], exp_band["down"])):
                if i == 0 or i == len(exp_band["up"]) - 1:
                    continue
                obs_exp_g.SetPointError(i - 1, 0, 0, down, up)

            obs_theory_g = ROOT.TGraphAsymmErrors(obs.root)
            obs_theory_g.SetLineColor(ROOT.kGreen)
            obs_theory_g.SetMarkerColor(ROOT.kBlack)
            # obs_theory_g.SetFillColor(ROOT.kGreen)
            # obs_theory_g.SetFillStyle(3005)
            for i, (up, down) in enumerate(zip(theory_band["up"], theory_band["down"])):
                if i == 0 or i == len(theory_band["up"]) - 1:
                    continue
                obs_theory_g.SetPointError(i - 1, 0.01, 0.01, down, up)

            wjets_theory_g = ROOT.TGraphAsymmErrors(wjets_obs.root)
            # wjets_theory_g.SetLineColor(ROOT.kBlue)
            # wjets_theory_g.SetMarkerColor(ROOT.kBlue)
            # wjets_theory_g.SetMarkerStyle(8)
            # wjets_theory_g.SetMarkerSize(1)
            # wjets_theory_g.SetLineWidth(2)
            wjets_theory_g.SetFillColor(ROOT.kBlue)
            wjets_theory_g.SetFillStyle(3005)
            for i, (up, down) in enumerate(zip(wjets_band["up"], wjets_band["down"])):
                if i == 0 or i == len(wjets_band["up"]) - 1:
                    continue
                wjets_theory_g.SetPointError(i - 1, 0.5, 0.5, down, up)

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
# plot_things()
