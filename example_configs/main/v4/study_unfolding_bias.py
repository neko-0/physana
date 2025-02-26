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
    #    '/gpfs/slac/atlas/fs1/d/yuzhan/configs_bk/slac_configs/V4_production_ver7/sherpa2211_run2_tight/band_unfoled_v2.pkl'
    #'output/unfolding_v4_Aug2021Talk_nominal/unfold.pkl'
    #'output/unfolding_v3_crackTest/unfold.pkl'
    '/gpfs/slac/atlas/fs1/d/yuzhan/configs_bk/slac_configs/Giordon_Aug2021_Talks/V5_production/track_calo_with_more_sys_2211/unfold.pkl',
)

# lumi = 36.1  # ifb
lumi = 138.861
# configMgr.out_path = pathlib.Path('/sdf/home/g/gstark/collinearw/output/unfolding_v4')
configMgr.out_path = pathlib.Path(
    '/sdf/home/g/gstark/collinearw/output/unfolding_v4_bias'
)

comparables = {
    'electron_inclusive_truth': (
        'unfold_realthang_fake-EL',
        'unfold_alt_realthang_fake-EL_wjets',
    ),
    'muon_inclusive_truth': (
        'unfold_realthang_fake-MU',
        'unfold_alt_realthang_fake-MU_wjets',
    ),
}

p2211_el = configMgr.get_process_set('unfold_realthang_fake-EL').nominal
p221_el = configMgr.get_process_set('unfold_alt_realthang_fake-EL_wjets').nominal
p2211_mu = configMgr.get_process_set('unfold_realthang_fake-MU').nominal
p221_mu = configMgr.get_process_set('unfold_alt_realthang_fake-MU_wjets').nominal


r2211_el = p2211_el.get_region('electron_inclusive_truth')
r221_el = p221_el.get_region('electron_inclusive_truth')
r2211_mu = p2211_mu.get_region('muon_inclusive_truth')
r221_mu = p221_mu.get_region('muon_inclusive_truth')

for observable in r2211_el:
    if not isinstance(observable, core.Histogram):
        continue

    base_output_path = (
        pathlib.Path(configMgr.out_path)
        .joinpath(
            "plots",
            "study_unfolding_bias",
            observable.name,
        )
        .resolve()
    )
    output_path = base_output_path.with_suffix('.pdf')

    obs2211_el = unfolding.plot.scale_to_xsec(
        r2211_el.get_histogram(observable.name), lumi
    )
    obs221_el = unfolding.plot.scale_to_xsec(
        r221_el.get_histogram(observable.name), lumi
    )
    obs2211_mu = unfolding.plot.scale_to_xsec(
        r2211_mu.get_histogram(observable.name), lumi
    )
    obs221_mu = unfolding.plot.scale_to_xsec(
        r221_mu.get_histogram(observable.name), lumi
    )

    hTrue2211_el = obs2211_el.root
    hTrue221_el = obs221_el.root
    hTrue2211_mu = obs2211_mu.root
    hTrue221_mu = obs221_mu.root

    RootBackend.apply_process_styles(hTrue2211_el, p2211_el)
    RootBackend.apply_process_styles(hTrue221_el, p221_el)
    RootBackend.apply_process_styles(hTrue2211_mu, p2211_mu)
    RootBackend.apply_process_styles(hTrue221_mu, p221_mu)

    RootBackend.apply_styles(hTrue2211_el, color=1)
    RootBackend.apply_styles(hTrue221_el, color=2)
    RootBackend.apply_styles(hTrue2211_mu, color=4)
    RootBackend.apply_styles(hTrue221_mu, color=6)

    # RootBackend.set_range(hTrue2211_el, top_room_scale=1e2)
    # RootBackend.set_range(hTrue_221, top_room_scale=1e2)
    RootBackend.set_range(hTrue2211_el, yrange=(1e-4, 1e6))
    RootBackend.set_range(hTrue221_el, yrange=(1e-4, 1e6))
    RootBackend.set_range(hTrue2211_mu, yrange=(1e-4, 1e6))
    RootBackend.set_range(hTrue221_mu, yrange=(1e-4, 1e6))

    canvas = RootBackend.make_canvas(f"RooUnfold_{output_path}", with_ratio=True)
    canvas.GetPad(1).SetLogy()
    leg = RootBackend.make_legend()

    # top pad
    canvas.cd(1)

    canvas.SetTitle("")

    hTrue2211_el.Draw("lp")
    hTrue221_el.Draw("lp same")
    hTrue2211_mu.Draw("lp same")
    hTrue221_mu.Draw("lp same")

    leg.AddEntry(hTrue2211_el, f"{hTrue2211_el.GetTitle()} - 2.2.11, e^{{-}}", "lp")
    leg.AddEntry(hTrue221_el, f"{hTrue2211_el.GetTitle()} - 2.2.1, e^{{-}}", "lp")
    leg.AddEntry(hTrue2211_mu, f"{hTrue2211_el.GetTitle()} - 2.2.11, #mu^{{-}}", "lp")
    leg.AddEntry(hTrue221_mu, f"{hTrue2211_el.GetTitle()} - 2.2.1, #mu^{{-}}", "lp")

    leg.Draw("SAME")
    RootBackend.make_atlas_label()

    # bottom pad
    canvas.cd(2)

    ratio_el = hTrue221_el.Clone()
    ratio_el.Divide(hTrue2211_el)

    ratio_mu = hTrue221_mu.Clone()
    ratio_mu.Divide(hTrue2211_mu)

    ratio_elmu = hTrue221_el.Clone()
    ratio_elmu.Divide(hTrue2211_mu)

    ratio_muel = hTrue221_mu.Clone()
    ratio_muel.Divide(hTrue2211_el)

    RootBackend.apply_styles(ratio_elmu, markerstyle=5)
    RootBackend.apply_styles(ratio_muel, markerstyle=5)

    # RootBackend.apply_styles(ratio, color=2)
    # RootBackend.apply_styles(ratio_mu, color=4)
    # RootBackend.apply_styles(ratio_mu_221, color=6)
    # ratio.SetMarkerStyle(8)
    # ratio.SetMarkerSize(0)
    # ratio.SetLineWidth(3)
    # ratio.SetLineStyle(1)
    ratio_el.SetTitle("")
    ratio_el.GetYaxis().SetTitle(f"2.2.1/2.2.11")
    ratio_el.GetYaxis().SetRangeUser(-0.05, 2.05)
    RootBackend.apply_font_styles(ratio_el.GetXaxis(), titleoffset=3)
    ratio_el.SetNdivisions(5, "Y")

    ratio_el.Draw('E')
    _draw_opts = 'E SAME'
    ratio_mu.Draw(_draw_opts)
    ratio_elmu.Draw(_draw_opts)
    ratio_muel.Draw(_draw_opts)

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
