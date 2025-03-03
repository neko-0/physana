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
    '/gpfs/slac/atlas/fs1/d/yuzhan/configs_bk/ColWJet_2022/v5/run2_nominal_noVgamma/unfold_nominal.pkl'
)

lumi = 138.861
configMgr.out_path = pathlib.Path(
    '/gpfs/slac/atlas/fs1/u/mgignac/analysis/collinearw/dev/collinearw/configs/main/v5/figures/'
)

signal = configMgr.get_process_set("wjets_2211")
target = 'unfold_closure'

for procset in configMgr.process_sets:
    if not procset.process_type == 'unfolded':
        continue
    if procset.name == target:
        unfolded = procset
        break

regions = ['inclusive', 'inclusive_2j', 'collinear', 'backtoback']

flavor = 'electron'

for region in regions:
    for observable in signal.nominal.get_region(f'electron_{region}_truth'):
        if not isinstance(observable, core.Histogram):
            continue

        base_output_path = (
            pathlib.Path(configMgr.out_path)
            .joinpath(
                "plots",
                "study_closure",
                region,
                observable.name,
            )
            .resolve()
        )
        output_path = base_output_path.with_suffix('.pdf')

        # Particle-level selection
        particle = unfolding.plot.scale_to_xsec(
            signal.nominal.get_region(f'{flavor}_{region}_truth').get_histogram(
                observable.name
            ),
            lumi,
        )

        # Reco selection
        reco = unfolding.plot.scale_to_xsec(
            signal.nominal.get_region(
                f'{flavor}_{region}_truth_reco_matching'
            ).get_histogram(observable.name),
            lumi,
        )

        # Unfolded
        meas = unfolding.plot.scale_to_xsec(
            unfolded.nominal.get_region(f'{flavor}_{region}_truth').get_histogram(
                observable.name
            ),
            lumi,
        )

        hTrue = particle.root
        hReco = reco.root
        hMeas = meas.root

        RootBackend.apply_process_styles(hTrue, signal.nominal)
        RootBackend.apply_process_styles(hReco, signal.nominal)
        RootBackend.apply_process_styles(hMeas, signal.nominal)

        RootBackend.apply_styles(hMeas, color=1)
        RootBackend.apply_styles(hReco, color=2)
        RootBackend.apply_styles(hTrue, color=4)

        RootBackend.set_range(hMeas, yrange=(1e-4, 1e6))

        canvas = RootBackend.make_canvas(f"RooUnfold_{output_path}", num_ratio=1)
        canvas.GetPad(1).SetLogy()
        leg = RootBackend.make_legend(text_size=0.055)

        # top pad
        canvas.cd(1)

        canvas.SetTitle("")

        hTrue.Draw("lp")
        hReco.Draw("lp same")
        hMeas.Draw("lp same")
        leg.AddEntry(hMeas, f"Unfolded", "lpe0")
        leg.AddEntry(hTrue, f"Generator", "lpe0")
        leg.AddEntry(hReco, f"Reco", "lpe0")

        leg.Draw("SAME")
        RootBackend.make_atlas_label()

        # bottom pad
        canvas.cd(2)

        ratio = hTrue.Clone()
        ratio.Divide(hMeas)
        RootBackend.apply_styles(ratio, color=4)
        # ratio.SetMarkerStyle(8)
        # ratio.SetMarkerSize(0)
        # ratio.SetLineWidth(3)
        # ratio.SetLineStyle(1)
        ratio.SetTitle("")
        ratio.GetYaxis().SetTitle("Unfolded/generator")
        ratio.GetYaxis().SetRangeUser(0.99, 1.01)
        RootBackend.apply_font_styles(ratio.GetXaxis(), titleoffset=3)
        ratio.SetNdivisions(505, "Y")
        ratio.Draw('E')
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
