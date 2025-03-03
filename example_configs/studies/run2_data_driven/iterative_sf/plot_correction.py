import collinearw
import ROOT

def main():
    dummy_config = collinearw.ConfigMgr()
    dummy_config.set_output_location("./")
    plotmaker = collinearw.PlotMaker(dummy_config, f"plot_corr")

    correction = collinearw.configMgr.CorrectionContainer()
    correction.add_correction_file("run2_wjets_2211_bkgd_correction.shelf")
    correction.add_correction_file("run2_wjets_2211_signal_correction.shelf")

    obs = "nJet30"

    for lepton_type in ["electron", "muon"]:
        #wjets_corr = (lepton_type, 'wjets_2211', obs, None)
        zjets_corr = (lepton_type, 'zjets_2211', obs, None)
        ttbar_corr = (lepton_type, 'ttbar', obs, None)
        dijets_corr = (lepton_type, 'dijets', obs, None)

        #wjets_corr = correction[wjets_corr]
        zjets_corr = correction[zjets_corr]
        ttbar_corr = correction[ttbar_corr]
        dijets_corr = correction[dijets_corr]

        #wjets_corr.ytitle = "Normalization Factor"
        zjets_corr.ytitle = "Normalization Factor"
        ttbar_corr.ytitle = "Normalization Factor"
        dijets_corr.ytitle = "Normalization Factor"

        #wjets_corr_h = wjets_corr.root
        zjets_corr_h = zjets_corr.root
        ttbar_corr_h = ttbar_corr.root
        dijets_el_corr_h = dijets_corr.root

        #wjets_corr_h.SetLineColor(ROOT.TColor.GetColor("#210898"))
        zjets_corr_h.SetLineColor(ROOT.TColor.GetColor("#167f45"))
        ttbar_corr_h.SetLineColor(ROOT.TColor.GetColor("#f7347a"))
        dijets_el_corr_h.SetLineColor(ROOT.TColor.GetColor("#210898"))

        canvas = plotmaker.make_canvas()
        # canvas.SetRightMargin(0.2)
        canvas.cd()

        legend = plotmaker.make_legend(0.5, 0.7, 0.6, 0.85)
        #legend.AddEntry(wjets_corr_h, "W+jets (Sh 2.2.11)")
        legend.AddEntry(zjets_corr_h, "Z+jets (Sh 2.2.11)")
        legend.AddEntry(ttbar_corr_h, "t#bar{t} (Powheg+Pythia8)")
        legend.AddEntry(dijets_el_corr_h, "Dijets (Pythia8)")

        zjets_corr_h.GetYaxis().SetRangeUser(0, 2.0)

        #wjets_corr_h.Draw()
        zjets_corr_h.Draw("")
        ttbar_corr_h.Draw("same")
        dijets_el_corr_h.Draw("same")
        legend.Draw()
        plotmaker.make_atlas_label()

        canvas.SaveAs(f"{plotmaker.output_dir}/{lepton_type}_correction.pdf")

    canvas = plotmaker.make_canvas()
    canvas.cd()
    legend = plotmaker.make_legend(0.5, 0.7, 0.6, 0.85)
    dijets_el_corr = ("electron", 'dijets', obs, None)
    dijets_el_corr = correction[dijets_el_corr]
    dijets_el_corr.ytitle = "Normalization Factor"
    dijets_el_corr_h = dijets_el_corr.root
    dijets_el_corr_h.GetYaxis().SetRangeUser(0, 2.0)
    dijets_el_corr_h.SetLineColor(ROOT.TColor.GetColor("#210898"))
    legend.AddEntry(dijets_el_corr_h, "Electron")

    dijets_mu_corr = ("muon", 'dijets', obs, None)
    dijets_mu_corr = correction[dijets_mu_corr]
    dijets_mu_corr.ytitle = "Normalization Factor"
    dijets_mu_corr_h = dijets_mu_corr.root
    dijets_mu_corr_h.GetYaxis().SetRangeUser(0, 2.0)
    dijets_mu_corr_h.SetLineColor(ROOT.TColor.GetColor("#b5e3af"))
    legend.AddEntry(dijets_mu_corr_h, "Muon")

    dijets_el_corr_h.Draw()
    dijets_mu_corr_h.Draw("same")
    legend.Draw()
    plotmaker.make_atlas_label()
    canvas.SaveAs(f"{plotmaker.output_dir}/dijets_correction.pdf")


if __name__ == "__main__":
    main()
