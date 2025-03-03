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
    '/gpfs/slac/atlas/fs1/d/yuzhan/configs_bk/ColWJet_2022/v5i/run2_MC_driven/ave_band_unfold.pkl'
)

lumi = 138.861
lumiunc = 0.02
configMgr.out_path = pathlib.Path(
    '/sdf/home/g/gstark/collinearw/output/v5i_run2_MC_driven_average'
)

regions = ['inclusive', 'inclusive_2j', 'collinear', 'backtoback']

for flavour in ['electron', 'muon']:

    if flavour == 'electron':
        nominal = configMgr.get_process_set('unfold_realthang_fake-EL').nominal
        Alt_Sh221 = configMgr.get_process_set(
            'unfold_alt_realthang_fake-EL_wjets'
        ).nominal
        Alt_FxFx = configMgr.get_process_set(
            'unfold_alt_realthang_fake-EL_wjets_FxFx'
        ).nominal
    else:
        nominal = configMgr.get_process_set('unfold_realthang_fake-MU').nominal
        Alt_Sh221 = configMgr.get_process_set(
            'unfold_alt_realthang_fake-MU_wjets'
        ).nominal
        Alt_FxFx = configMgr.get_process_set(
            'unfold_alt_realthang_fake-MU_wjets_FxFx'
        ).nominal

    for region in regions:
        r_nominal = nominal.get_region(f'{flavour}_{region}_truth')
        r_Sh221 = Alt_Sh221.get_region(f'{flavour}_{region}_truth')
        r_FxFx = Alt_FxFx.get_region(f'{flavour}_{region}_truth')

        for observable in r_nominal:
            if not isinstance(observable, core.Histogram):
                continue

            base_output_path = (
                pathlib.Path(configMgr.out_path)
                .joinpath(
                    "study_unfolding_bias",
                    flavour,
                    region,
                    observable.name,
                )
                .resolve()
            )
            output_path = base_output_path.with_suffix('.pdf')

            obs_nominal = unfolding.plot.scale_to_xsec(
                r_nominal.get_histogram(observable.name), lumi
            )
            obs_Sh221 = unfolding.plot.scale_to_xsec(
                r_Sh221.get_histogram(observable.name), lumi
            )
            obs_FxFx = unfolding.plot.scale_to_xsec(
                r_FxFx.get_histogram(observable.name), lumi
            )

            div_Sh221 = obs_Sh221 / obs_nominal
            div_FxFx = obs_FxFx / obs_nominal

            hNominal = obs_nominal.root_graph
            hSh221 = obs_Sh221.root_graph
            hFxFx = obs_FxFx.root_graph

            # RootBackend.apply_process_styles(hNominal,nominal)
            # RootBackend.apply_process_styles(hSh221, Alt_Sh221)
            # RootBackend.apply_process_styles(hFxFx, Alt_FxFx)

            RootBackend.apply_styles(hNominal, color=1, markerstyle=8, markersize=1.2)
            RootBackend.apply_styles(hSh221, color=2, markerstyle=25, markersize=1.2)
            RootBackend.apply_styles(hFxFx, color=4, markerstyle=22, markersize=1.2)

            RootBackend.set_range(hNominal)

            canvas = RootBackend.make_canvas(f"RooUnfold_{output_path}", num_ratio=1)
            canvas.GetPad(1).SetLogy()
            leg = RootBackend.make_legend()

            # top pad
            canvas.cd(1)

            canvas.SetTitle("")

            hNominal.Draw("AP")
            hFxFx.Draw("P same")
            hSh221.Draw("P same")

            leg.AddEntry(hNominal, f"Sherpa 2.2.11 (nom)", "lp")
            leg.AddEntry(hFxFx, f"MG_aMC@NLO+Py8 FxFx", "lp")
            leg.AddEntry(hSh221, f"Sherpa 2.2.1", "lp")

            leg.Draw("SAME")
            RootBackend.make_atlas_label()

            # bottom pad
            canvas.cd(2)

            ratio_Sh221 = div_Sh221.root_graph
            ratio_FxFx = div_FxFx.root_graph

            RootBackend.apply_styles(
                ratio_Sh221, color=2, markerstyle=25, markersize=1.2
            )
            RootBackend.apply_styles(
                ratio_FxFx, color=4, markerstyle=22, markersize=1.2
            )

            ratio_Sh221.SetTitle("")
            ratio_Sh221.GetYaxis().SetTitle("Alt/Nom")
            ratio_Sh221.GetYaxis().SetRangeUser(0.85, 1.15)
            RootBackend.apply_font_styles(ratio_Sh221.GetXaxis(), titleoffset=3)
            RootBackend.apply_font_styles(ratio_Sh221.GetYaxis(), titleoffset=1)
            ratio_Sh221.GetYaxis().SetNdivisions(5)

            ratio_Sh221.Draw('AP')
            ratio_FxFx.Draw('P same')
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
