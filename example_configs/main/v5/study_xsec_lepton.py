from collinearw import ConfigMgr, core, utils
from collinearw.strategies import unfolding
from collinearw.backends import RootBackend

import ROOT
import pathlib
import logging

logging.basicConfig()
logging.getLogger().setLevel(logging.CRITICAL)

log = logging.getLogger(__name__)

configMgr = ConfigMgr.open(
    '/gpfs/slac/atlas/fs1/d/yuzhan/configs_bk/ColWJet_2022/v5/run2_noVgamma_tauBkg_updateMuon/unfold_merged_fxfx.pkl'
)

lumi = 138.861
configMgr.out_path = pathlib.Path('./figures/')
signal = configMgr.get_process_set("wjets_2211")
unfoldeds = [
    procset
    for procset in configMgr.process_sets
    if procset.process_type == 'unfolded' and procset.name == 'unfold_realthang_fake-MU'
]

eles = [
    procset
    for procset in configMgr.process_sets
    if procset.process_type == 'unfolded' and procset.name == 'unfold_realthang_fake-EL'
]

mu_data = unfoldeds[0]
ele_data = eles[0]

regions = ['inclusive', 'inclusive_2j', 'collinear', 'backtoback']

for region in regions:
    for observable in signal.nominal.get_region(f'electron_{region}_truth'):
        if not isinstance(observable, core.Histogram):
            continue

        base_output_path = (
            pathlib.Path(configMgr.out_path)
            .joinpath(
                "plots",
                "study_el_vs_mu_xsec",
                region,
                observable.name,
            )
            .resolve()
        )
        output_path = base_output_path.with_suffix('.pdf')

        observable_el = unfolding.plot.scale_to_xsec(
            signal.nominal.get_region(f'electron_{region}_truth').get_histogram(
                observable.name
            ),
            lumi,
        )

        data_ele = unfolding.plot.scale_to_xsec(
            ele_data.nominal.get_region(f'electron_{region}_truth').get_histogram(
                observable.name
            ),
            lumi,
        )

        observable_mu = unfolding.plot.scale_to_xsec(
            signal.nominal.get_region(f'muon_{region}_truth').get_histogram(
                observable.name
            ),
            lumi,
        )

        data_mu = unfolding.plot.scale_to_xsec(
            mu_data.nominal.get_region(f'muon_{region}_truth').get_histogram(
                observable.name
            ),
            lumi,
        )

        hTrue_el = data_ele.root  # observable_el.root
        hTrue_mu = data_mu.root  # observable_mu.root

        hAverage = data_ele.root.Clone("average")
        hAverage.Reset()

        bin_min = 1e6
        bin_max = 1e-4
        for b in range(0, hAverage.GetNbinsX() + 1):
            avg = 0.5 * (hTrue_el.GetBinContent(b) + hTrue_mu.GetBinContent(b))
            avgErr = 0.5 * (hTrue_el.GetBinError(b) + hTrue_mu.GetBinError(b))
            hAverage.SetBinContent(b, avg)
            hAverage.SetBinError(b, avgErr)
            if avg < bin_min and avg > 0:
                bin_min = avg
            if avg > bin_max:
                bin_max = avg

        RootBackend.apply_process_styles(hTrue_el, signal.nominal)
        RootBackend.apply_process_styles(hTrue_mu, signal.nominal)
        RootBackend.apply_process_styles(hAverage, signal.nominal)

        RootBackend.apply_styles(hAverage, color=1)
        RootBackend.apply_styles(hTrue_el, color=2)
        RootBackend.apply_styles(hTrue_mu, color=4)

        RootBackend.set_range(hAverage, yrange=(0.5 * bin_min, 50 * bin_max))

        canvas = RootBackend.make_canvas(f"RooUnfold_{output_path}", num_ratio=1)
        canvas.GetPad(1).SetLogy()
        leg = RootBackend.make_legend(text_size=0.055)

        # top pad
        canvas.cd(1)

        canvas.SetTitle("")

        hAverage.Draw("lp")
        hTrue_el.Draw("lp same")
        hTrue_mu.Draw("lp same")
        leg.AddEntry(hAverage, f"Average", "lpe0")
        leg.AddEntry(hTrue_el, f"Electron", "lpe0")
        leg.AddEntry(hTrue_mu, f"Muon", "lpe0")

        leg.Draw("SAME")
        RootBackend.make_atlas_label()

        # bottom pad
        canvas.cd(2)

        ratio = hAverage.Clone()
        ratioel = hAverage.Clone()
        ratio.Divide(hTrue_mu)
        ratioel.Divide(hTrue_el)
        RootBackend.apply_styles(ratio, color=4)
        RootBackend.apply_styles(ratioel, color=2)
        # ratio.SetMarkerStyle(8)
        # ratio.SetMarkerSize(0)
        # ratio.SetLineWidth(3)
        # ratio.SetLineStyle(1)
        ratio.SetTitle("")
        ratio.GetYaxis().SetTitle("e or #mu / average")
        ratio.GetYaxis().SetRangeUser(0.82, 1.18)
        RootBackend.apply_font_styles(ratio.GetXaxis(), titleoffset=3)
        ratio.SetNdivisions(505, "Y")
        ratio.Draw('E')
        ratioel.Draw('E same')
        _draw_opts = 'E SAME'

        # add dashed line through midpoint 1.0
        # NB: requires canvas.Update() to get the right Uxmin/Uxmax values
        canvas.Update()
        line = ROOT.TLine(
            canvas.GetPad(1).GetUxmin(), 1, canvas.GetPad(1).GetUxmax(), 1
        )
        line.SetLineColor(ROOT.kBlack)
        line.SetLineStyle(ROOT.kDashed)
        line.Draw('SAME')

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with utils.all_redirected():
            canvas.SaveAs(str(output_path))
