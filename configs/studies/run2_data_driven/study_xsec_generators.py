from collinearw import ConfigMgr, backends, utils
from collinearw.strategies import unfolding
from collinearw.backends import RootBackend
from collinearw.strategies import systematics as sys_handler
import numpy as np

import ROOT
import pathlib
import logging

logging.basicConfig()
logging.getLogger().setLevel(logging.CRITICAL)

configMgr = ConfigMgr.open("band_unfold.pkl")

lumi = 138.861
configMgr.out_path = "./"

main_proc = configMgr.get_process_set('unfold_realthang_fake-EL')
# configMgr.get_process_set('data').nominal.color = 1  # set it to black
# configMgr.get_process_set('data').nominal.binerror = 0  # set it to 0

unfoldeds_el = [
    configMgr.get_process_set(name)
    for name in (
        'unfold_realthang_fake-EL',
        'unfold_alt_realthang_fake-EL_wjets_FxFx',
        'unfold_alt_realthang_fake-EL_wjets',
        'unfold_alt_realthang_fake-EL_wjets_2211_ASSEW',
    )
]

stylers = [
    configMgr.get_process_set(name)
    for name in ('wjets_2211', 'wjets_FxFx', 'wjets', 'wjets_2211_ASSEW')
]

colors = [
    1,
    *(ROOT.TColor.GetColor(color) for color in ['#1b7837', '#af8dc3', '#7fbf7b']),
]

assert len(unfoldeds_el) == len(stylers)
assert len(colors) == len(stylers)

region_cats = ['inclusive', 'inclusive_2j', 'collinear', 'backtoback']

for region_cat in region_cats:
    for observable in main_proc.nominal.get_region(f'electron_{region_cat}_truth'):

        hMain = None
        hAlternatives = []

        for unfolded_el, style_proc, color in zip(unfoldeds_el, stylers, colors):
            unfolded_mu = configMgr.get_process_set(
                unfolded_el.name.replace('EL', 'MU')
            )

            # get the unfolded for electron and muons
            observable_el = unfolding.plot.scale_to_xsec(
                unfolded_el.nominal.get_region(
                    f'electron_{region_cat}_truth'
                ).get_histogram(observable.name),
                lumi,
            )

            observable_mu = unfolding.plot.scale_to_xsec(
                unfolded_mu.nominal.get_region(
                    f'muon_{region_cat}_truth'
                ).get_histogram(observable.name),
                lumi,
            )

            observable = (observable_el + observable_mu) / 2.0
            hReco = observable.root

            hReco_error = None
            if (
                observable_el.systematic_band is not None
                and observable_mu.systematic_band is not None
            ) and False:
                x_values = observable.bins[:-1] + observable.bin_width[1:-1] / 2.0
                y_values = observable.bin_content[1:-1]
                ex_l = observable.bin_width / 2.0
                ex_h = observable.bin_width / 2.0
                ey_l = np.sqrt(
                    observable_el.scale_band('experimental')['down'][1:-1] ** 2
                    + observable_mu.scale_band('experimental')['down'][1:-1] ** 2
                )
                ey_h = np.sqrt(
                    observable_el.scale_band('experimental')['up'][1:-1] ** 2
                    + observable_mu.scale_band('experimental')['up'][1:-1] ** 2
                )
                hReco_error = ROOT.TGraphAsymmErrors(
                    len(x_values), x_values, y_values, ex_l, ex_h, ey_l, ey_h
                )

                RootBackend.apply_process_styles(hReco_error, style_proc.nominal)
                RootBackend.apply_styles(hReco_error, color=color)

                # see https://root.cern.ch/doc/master/classTGraphPainter.html#GP03
                hReco_error.SetFillColorAlpha(13, 0.35)  # red, transparent
                hReco_error.SetFillStyle(1001)  # solid
                hReco_error.SetTitle("Bkg th. + Expt. unct.")

            RootBackend.apply_process_styles(hReco, style_proc.nominal)
            RootBackend.apply_styles(hReco, color=color)

            if unfolded_el == main_proc:
                hMain = (hReco, hReco_error)
            else:
                hAlternatives.append((hReco, hReco_error))

        base_output_path = (
            pathlib.Path(configMgr.out_path)
            .joinpath(
                "plots",
                "study_xsec_generators",
                f'lepton_{region_cat}_truth',
                observable.name,
            )
            .resolve()
        )
        output_path = base_output_path.with_suffix('.pdf')

        unfolding.plot.plot_results(
            None,
            [i[0] for i in hAlternatives],
            hMain[0],
            output_path,
            reco_error=hMain[1],
            truths_error=[i[1] for i in hAlternatives],
            # logy=False,
            # yrange=(0,500),
        )

# unfolding.plot.make_eff_plots(configMgr)
# unfolding.plot.make_debugging_plots(configMgr, output_folder=f"unfolding")
