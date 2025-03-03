import collinearw
from collinearw.strategies.systematics.core import compute_quadrature_sum, compute_systematics

import logging

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

def main():
    config = collinearw.ConfigMgr.open("unfold_run2_2211.pkl")

    regions = [r.name for r in config.regions]
    observables = [h.name for h in config.histograms]

    # removing background processes that no longer needed after the unfolding.
    process_filter = [
        "zjets_2211",
        "ttbar",
        "diboson_powheg",
        "singletop",
        "vgamma",
        "dijets",
        "wjets_2211_tau",
        "data_subtracted_bkg_fakes",
    ]
    for proc in process_filter:
        config.remove_process_set(proc)

    print(f"list of processes : {config.list_processes()}")

    # compute PDF and MuR MuF systematics
    # after computing, they can be lookup through the config.generate_systematic_group()
    # compute_systematics(config, "wjets_2211_MUR_MUF_Scale", "min_max")
    compute_systematics(config, "wjets_2211_NNPDF30nnlo_hessian", "hessian")
    compute_systematics(config, "wjets_FxFx_MUR_MUF_Scale", "min_max")
    compute_systematics(config, "wjets_FxFx_PDF", "stdev")
    compute_systematics(config, "wjets_MUR_MUF_Scale", "min_max")
    compute_systematics(config, "wjets_PDF", "stdev")
    compute_systematics(config, "zjets_2211_MUR_MUF_Scale", "min_max")
    compute_systematics(config, "zjets_2211_NNPDF30nnlo_hessian", "hessian")
    # compute_systematics(config, "diboson_NNPDF30nnlo_hessian", "hessian")
    compute_systematics(config, "diboson_powheg_MUR_MUF_Scale", "min_max")
    # compute_systematics(config, "diboson_powheg_PDF", "stdev")
    compute_systematics(config, "ttbar_ren_fac_scale", "min_max")
    compute_systematics(config, "ttbar_ISR_scale", "min_max")
    compute_systematics(config, "ttbar_FSR_scale", "min_max")
    compute_systematics(config, "ttbar_NNPDF30_PDF", "stdev")
    compute_systematics(config, "singletop_ren_fac_scale", "min_max")
    compute_systematics(config, "singletop_A14_Tunning", "min_max")
    compute_systematics(config, "singletop_PDF", "stdev")
    compute_systematics(config, "dijets_ISR_scale", "min_max")
    compute_systematics(config, "dijets_FSR_scale", "min_max")
    compute_systematics(config, "dijets_A14_Tunning", "min_max")

    compute_systematics(config, "JET_GroupedNP_1", "symmetrize_up_down")
    compute_systematics(config, "JET_GroupedNP_2", "symmetrize_up_down")
    compute_systematics(config, "JET_GroupedNP_3", "symmetrize_up_down")
    compute_systematics(config, "JET_JER_EffectiveNP_1", "symmetrize_up_down")
    compute_systematics(config, "JET_JER_EffectiveNP_2", "symmetrize_up_down")
    compute_systematics(config, "JET_JER_EffectiveNP_3", "symmetrize_up_down")
    compute_systematics(config, "JET_JER_EffectiveNP_4", "symmetrize_up_down")
    compute_systematics(config, "JET_JER_EffectiveNP_5", "symmetrize_up_down")
    compute_systematics(config, "JET_JER_EffectiveNP_6", "symmetrize_up_down")
    compute_systematics(config, "JET_JER_EffectiveNP_7restTerm", "symmetrize_up_down")

    # generate dictionary for grouping different systematics

    # these only affect the unfolded data, don't include the signal theory
    # config.generate_systematic_group(name, lookup tuple) create python dict
    experimental = {}
    experimental["Jet"] = {}
    experimental["Jet"].update( config.generate_systematic_group("JET", ("*JET*", "*symmetrize*", "*")) )
    experimental["Jet"].update( config.generate_systematic_group("JVT", ("*jvtWeight_JET_JvtEfficiency*", "NoSys", "*")) )

    experimental["Lepton"] = {}
    experimental["Lepton"].update( config.generate_systematic_group("Lepton-Weight", ("*leptonWeight*", "NoSys", "*leptonWeight*")) )
    experimental["Lepton"].update( config.generate_systematic_group("TrigWeight", ("*trigWeight*", "NoSys", "*")) )

    experimental["B-tagging"] = {}
    experimental["B-tagging"].update( config.generate_systematic_group("bTagWeight", ("*bTagWeight*", "NoSys", "*")) )

    theories = {}
    theories["ttbar"] = {}
    theories["ttbar"].update( config.generate_systematic_group("ttbar-ren_fac_scale", ("*ttbar*ren_fac_scale*", "min_max", "*")) )
    theories["ttbar"].update( config.generate_systematic_group("ttbar-ISR_scale", ("*ttbar*ISR_scale*", "min_max", "*")) )
    theories["ttbar"].update( config.generate_systematic_group("ttbar-FSR_scale", ("*ttbar*FSR_scale*", "min_max", "*")) )
    theories["ttbar"].update( config.generate_systematic_group("ttbar-NNPDF30_PDF", ("*ttbar*NNPDF30_PDF*", "stdev", "*")) )

    theories["singletop"] = {}
    theories["singletop"].update( config.generate_systematic_group("singletop-ren_fac_scale", ("*singletop_ren_fac_scale*", "min_max", "*")) )
    theories["singletop"].update( config.generate_systematic_group("singletop-A14_Tunning", ("*singletop_A14_Tunning*", "min_max", "*")) )
    theories["singletop"].update( config.generate_systematic_group("singletop-PDF", ("*singletop_PDF*", "stdev", "*")) )

    theories["dijets"] = {}
    theories["dijets"].update( config.generate_systematic_group("dijets_FSR_scale", ("dijets_FSR_scale", "min_max", "*")) )
    theories["dijets"].update( config.generate_systematic_group("dijets_ISR_scale", ("dijets_ISR_scale", "min_max", "*")) )
    theories["dijets"].update( config.generate_systematic_group("dijets_A14_Tunning", ("dijets_A14*", "min_max", "*")) )

    theories["zjets"] = {}
    theories["zjets"].update( config.generate_systematic_group("zjets_2211-MUR_MUF", ("*zjets*", "min_max", "*")) )
    theories["zjets"].update( config.generate_systematic_group("zjets_2211-NNPDF30nnlo_hessian", ("*zjets*", "hessian", "*")) )

    theories["diboson"] = {}
    # theories["diboson"].update( config.generate_systematic_group("diboson-MUR_MUF", ("*diboson*", "min_max", "*")) )
    # theories["diboson"].update( config.generate_systematic_group("diboson-NNPDF30nnlo_hessian", ("*diboson*", "hessian", "*")) )
    theories["diboson"].update( config.generate_systematic_group("diboson_powheg-MUR_MUF_Scale", ("*diboson_powheg*MUR_MUF_Scale*", "min_max", "*")) )
    # theories["diboson"].update( config.generate_systematic_group("diboson_powheg-PDF", ("*diboson_powheg*PDF*", "stdev", "*")) )

    theories["wjets_2211"] = {}
    theories["wjets_2211"].update( config.generate_systematic_group("wjets_2211-MUR_MUF", ("*wjets_2211*MUR_MUF_Scale*", "min_max", "*")) )
    theories["wjets_2211"].update( config.generate_systematic_group("wjets_2211-NNPDF30nnlo_hessian", ("*wjets_2211*", "hessian", "*")) )

    theories["wjets_FxFx"] = {}
    theories["wjets_FxFx"].update( config.generate_systematic_group("wjets_FxFx-MUR_MUF", ("*wjets_FxFx*", "min_max", "*")) )
    theories["wjets_FxFx"].update( config.generate_systematic_group("wjets_FxFx-PDF", ("*wjets_FxFx*", "stdev", "*")) )

    theories["wjets"] = {}
    theories["wjets"].update( config.generate_systematic_group("wjets-MUR_MUF", ("wjets_MUR_MUF_Scale", "min_max", "*")) )
    theories["wjets"].update( config.generate_systematic_group("wjets-PDF", ("wjets_PDF", "stdev", "*")) )

    plist = ["wjets_2211", "unfold_closure", "unfold_realthang_fake-MU", "unfold_realthang_fake-EL"]

    signal_processes= ["wjets_2211", "wjets_FxFx", "wjets"]
    data_processes = ["unfold_realthang_fake-EL", "unfold_realthang_fake-MU"]

    for process_name in config.list_processes():
        syst_type = "experimental"
        if process_name not in signal_processes:
            for exp_name in experimental:
                for name, syst_list in experimental[exp_name].items():
                    print(f"Doing process {process_name}, group {name}")
                    compute_quadrature_sum(
                        config,
                        process_name,
                        exp_name,
                        syst_type,
                        syst_list,
                        sub_band_name=name,
                        store_process=False,
                        regions=regions,
                        observables=observables,
                    )

        syst_type = "theory"
        for theory_name in theories:
            # skip signal theory on data
            if process_name in data_processes and theory_name in signal_processes:
                continue
            # make sure W+jets has only theory of it's own type
            if process_name in signal_processes:
                if process_name != theory_name:
                    continue
                name_maping = {
                    "wjets_2211-MUR_MUF" : "Scale(#muR,#muF)",
                    "wjets_2211-NNPDF30nnlo_hessian" : "NNPDF30nnlo",
                    "wjets_FxFx-MUR_MUF" : "Scale(#muR,#muF)",
                    "wjets_FxFx-PDF" : "PDF",
                    "wjets-MUR_MUF" : "Scale(#muR,#muF)",
                    "wjets-PDF" : "PDF",
                }
            else:
                name_maping = {}
            for name, syst_list in theories[theory_name].items():
                print(f"Doing process {process_name}, group {name}")
                compute_quadrature_sum(
                    config,
                    process_name,
                    name_maping.get(name, theory_name),
                    syst_type,
                    syst_list,
                    sub_band_name=name,
                    store_process=False,
                    regions=regions,
                    observables=observables,
                )

    # we only need the nominal with syst bands
    splitted_config = config.self_split("systematic")

    nominal_config = next(splitted_config)

    nominal_config.save("band_unfold.pkl")


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
