from collinearw import ConfigMgr, core, utils
from collinearw.strategies import unfolding
from collinearw.backends import RootBackend

import ROOT
import pathlib
import logging

logging.basicConfig()
logging.getLogger().setLevel(logging.CRITICAL)

log = logging.getLogger(__name__)

configMgr = ConfigMgr.open('band_unfold.pkl')

# lumi = 36.1  # ifb
lumi = 138.861
# configMgr.out_path = pathlib.Path('/sdf/home/g/gstark/collinearw/output/unfolding_v4')

signal = configMgr.get_process_set("wjets_2211")

inclusiveXSec = "nTruthBJet30"

region_el = 'electron_inclusive_truth'
region_mu = 'muon_inclusive_truth'

for observable in signal.nominal.get_region(region_el):
    if not isinstance(observable, core.Histogram):
        continue

    base_output_path = (
        pathlib.Path(configMgr.out_path)
        .joinpath(
            "plots",
            "study_el_vs_mu_xsec",
            observable.name,
        )
        .resolve()
    )
    output_path = base_output_path.with_suffix('.pdf')

    observable_el = unfolding.plot.scale_to_xsec(
        signal.nominal.get_region(region_el).get_histogram(observable.name),
        lumi,
    )

    observable_mu = unfolding.plot.scale_to_xsec(
        signal.nominal.get_region(region_mu).get_histogram(observable.name),
        lumi,
    )

    hTrue_el = observable_el.root
    hTrue_mu = observable_mu.root

    RootBackend.apply_process_styles(hTrue_el, signal.nominal)
    RootBackend.apply_process_styles(hTrue_mu, signal.nominal)

    RootBackend.apply_styles(hTrue_el, color=2)
    RootBackend.apply_styles(hTrue_mu, color=4)

    # RootBackend.set_range(hTrue_el, top_room_scale=1e2)
    # RootBackend.set_range(hTrue_mu, top_room_scale=1e2)
    RootBackend.set_range(hTrue_el, yrange=(1e-4, 1e4))
    RootBackend.set_range(hTrue_mu, yrange=(1e-4, 1e4))

    canvas = RootBackend.make_canvas(f"RooUnfold_{output_path}", with_ratio=True)
    canvas.GetPad(1).SetLogy()
    leg = RootBackend.make_legend()

    # top pad
    canvas.cd(1)

    canvas.SetTitle("")

    hTrue_el.Draw("lp")
    hTrue_mu.Draw("lp same")
    leg.AddEntry(hTrue_el, f"{hTrue_el.GetTitle()} -- e^{{-}}", "lp")
    leg.AddEntry(hTrue_mu, f"{hTrue_el.GetTitle()} -- #mu^{{-}}", "lp")

    leg.Draw("SAME")
    RootBackend.make_atlas_label()

    # bottom pad
    canvas.cd(2)

    ratio = hTrue_el.Clone()
    ratio.Divide(hTrue_mu)
    RootBackend.apply_styles(ratio, color=1)
    # ratio.SetMarkerStyle(8)
    # ratio.SetMarkerSize(0)
    # ratio.SetLineWidth(3)
    # ratio.SetLineStyle(1)
    ratio.SetTitle("")
    ratio.GetYaxis().SetTitle("e^{-}/#mu^{-}")
    ratio.GetYaxis().SetRangeUser(-0.05, 2.05)
    RootBackend.apply_font_styles(ratio.GetXaxis(), titleoffset=3)
    ratio.SetNdivisions(5, "Y")
    ratio.Draw('E')
    _draw_opts = 'E SAME'

    # add dashed line through midpoint 1.0
    # NB: requires canvas.Update() to get the right Uxmin/Uxmax values
    canvas.Update()
    line = ROOT.TLine(canvas.GetPad(1).GetUxmin(), 1, canvas.GetPad(1).GetUxmax(), 1)
    line.SetLineColor(ROOT.kBlack)
    line.SetLineStyle(ROOT.kDashed)
    line.Draw('SAME')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with utils.all_redirected():
        canvas.SaveAs(str(output_path))
