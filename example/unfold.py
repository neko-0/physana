import RooUnfold
import ROOT
import pickle
import pathlib

ROOT.gROOT.SetBatch(True)
config = pickle.load(open('unfoldTest_v0/unfold.pkl', 'rb'))

# number of times to unfold
for n_unfolds in range(1, 11):
    for phasespace in ['inclusive', 'collinear', 'backtoback']:
        for variables in [
            # ('jet1Pt', 'jet1TruthPt', 'response_matrix_jet1Pt'),
            # ('met', 'missingTruth', 'response_matrix_met'),
            ('lep1Pt', 'lep1TruthPt', 'response_matrix_lep1Pt'),
            # ('nJet25', 'nTruthJet25', 'response_matrix_nJets'),
            # ('Ht', 'HtTruth', 'response_matrix_Ht'),
        ]:
            hMeas = (
                config.get_process('wjets')
                .get_region(f'{phasespace}_truth_reco_matching_ROOT')
                .get_histogram(variables[0])
                .histogram
            )
            hTrue = (
                config.get_process('wjets')
                .get_region(f'{phasespace}_truth_ROOT')
                .get_histogram(variables[1])
                .histogram
            )
            hResponse = (
                config.get_process('wjets')
                .get_region(f'{phasespace}_truth_reco_matching_ROOT')
                .get_histogram(variables[2])
                .histogram
            )

            response = ROOT.RooUnfoldResponse(hMeas, hTrue, hResponse)

            unfold = ROOT.RooUnfoldBayes(response, hMeas, n_unfolds)

            hReco = unfold.Hreco()

            unfold.PrintTable(ROOT.cout, hTrue)

            hTrue.SetLineWidth(2)
            hTrue.SetLineColor(2)
            hMeas.SetLineWidth(2)
            hMeas.SetLineColor(4)
            hReco.SetLineWidth(2)
            hReco.SetLineColor(4)

            canvas = ROOT.TCanvas("RooUnfold", "bayes")
            leg = ROOT.TLegend(0.55, 0.65, 0.76, 0.82)
            hTrue.Draw()
            leg.AddEntry(hTrue, "Truth")
            hReco.Draw("SAME")
            leg.AddEntry(hReco, "Unfolded", "le")
            hMeas.Draw("SAME")
            leg.AddEntry(hMeas, "Measured")
            leg.Draw("SAME")
            canvas.SaveAs(
                str(
                    pathlib.Path(config.out_path)
                    .joinpath(f"RooUnfold_{phasespace}_{variables[0]}_n{n_unfolds}.pdf")
                    .resolve()
                )
            )
