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

configMgr = ConfigMgr.open(
    #    '/gpfs/slac/atlas/fs1/d/yuzhan/configs_bk/slac_configs/V4_production_ver7/sherpa2211_run2_tight/band_unfoled_v2.pkl'
    'band_unfold.pkl'
)

lumi = 138.861
# configMgr.out_path = pathlib.Path(
#     '/sdf/home/g/gstark/collinearw/output/unfolding_v5_nominal_noVgamma'
# )

signals = [
    procset
    for procset in configMgr.process_sets
    if procset.process_type in ["signal", "signal_alt"]
]
unfoldeds_el = [
    procset
    for procset in configMgr.process_sets
    if procset.process_type == 'unfolded' and procset.name.endswith('EL')
]

""" do not need to do this for experimental systs
for systematic_name in unfoldeds[1].systematic_names:
    sys_handler.compute_systematics(configMgr, systematic_name, "min_max", process_sets=unfoldeds)
"""

region_cats = ['inclusive', 'inclusive_2j', 'collinear', 'backtoback']

for unfolded_el in unfoldeds_el:
    unfolded_mu = configMgr.get_process_set(unfolded_el.name.replace('EL', 'MU'))
    for region_cat in region_cats:
        for observable in unfolded_el.nominal.get_region(
            f'electron_{region_cat}_truth'
        ):
            base_output_path = (
                pathlib.Path(configMgr.out_path)
                .joinpath(
                    "unfolding_plots",
                    "unfolded",
                    unfolded_el.name.replace('EL', 'LEP'),
                    f'lepton_{region_cat}_truth',
                    observable.name,
                )
                .resolve()
            )
            output_path = base_output_path.with_suffix('.pdf')

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

            # override some weird defaults
            unfolded_el.nominal.color = 1  # set it to black
            unfolded_el.nominal.binerror = 0  # set it to 0

            hReco_error = None
            if (
                observable_el.systematic_band is not None
                and observable_mu.systematic_band is not None
            ):
                x_values = observable.bins[:-1] + observable.bin_width / 2.0
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

                RootBackend.apply_process_styles(hReco_error, unfolded_el.nominal)
                # see https://root.cern.ch/doc/master/classTGraphPainter.html#GP03
                hReco_error.SetFillColorAlpha(13, 0.35)  # red, transparent
                hReco_error.SetFillStyle(1001)  # solid
                hReco_error.SetTitle("Bkg th. + Expt. unct.")

            RootBackend.apply_process_styles(hReco, unfolded_el.nominal)

            truths_el = [
                unfolding.plot.scale_to_xsec(
                    signal.nominal.get_region(
                        f'electron_{region_cat}_truth'
                    ).get_histogram(observable.name),
                    lumi,
                )
                for signal in signals
            ]

            truths_mu = [
                unfolding.plot.scale_to_xsec(
                    signal.nominal.get_region(f'muon_{region_cat}_truth').get_histogram(
                        observable.name
                    ),
                    lumi,
                )
                for signal in signals
            ]

            hTruths = []
            truths_error = []
            for truth_obs_el, truth_obs_mu, pset in zip(truths_el, truths_mu, signals):
                # first build the combined histogram
                truth_obs = (truth_obs_el + truth_obs_mu) / 2.0
                hTrue = truth_obs.root
                RootBackend.apply_process_styles(hTrue, pset.nominal)
                hTruths.append(hTrue)

                if (
                    truth_obs_el.systematic_band is not None
                    and truth_obs_mu.systematic_band is not None
                ):
                    x_values = truth_obs.bins[:-1] + truth_obs.bin_width / 2.0
                    y_values = truth_obs.bin_content[1:-1]
                    ex_l = truth_obs.bin_width / 2.0
                    ex_h = truth_obs.bin_width / 2.0
                    try:
                        ey_l = np.sqrt(
                            truth_obs_el.scale_band('theory')['down'][1:-1] ** 2
                            + truth_obs_mu.scale_band('theory')['down'][1:-1] ** 2
                        )
                        ey_h = np.sqrt(
                            truth_obs_el.scale_band('theory')['up'][1:-1] ** 2
                            + truth_obs_mu.scale_band('theory')['up'][1:-1] ** 2
                        )
                    except KeyError:
                        ey_l = truth_obs.bin_content[1:-1]
                        ey_h = truth_obs.bin_content[1:-1]
                    truth_error = ROOT.TGraphAsymmErrors(
                        len(x_values), x_values, y_values, ex_l, ex_h, ey_l, ey_h
                    )

                    RootBackend.apply_process_styles(truth_error, pset.nominal)
                    # see https://root.cern.ch/doc/master/classTGraphPainter.html#GP03
                    truth_error.SetFillColorAlpha(pset.nominal.color, 0.35)
                    truth_error.SetFillStyle(1001)  # solid
                    truth_error.SetTitle("Theory Uncrt.")
                    truths_error.append(truth_error)
                else:
                    truths_error.append(None)

            unfolding.plot.plot_results(
                None,
                hTruths,
                hReco,
                output_path,
                reco_error=hReco_error,
                truths_error=truths_error,
                yrange=(1e-3,1e4),
                layout=[[0,1], [2]],
                # logy=False,
                # yrange=(-500,500),
            )

# unfolding.plot.make_eff_plots(configMgr)
# unfolding.plot.make_debugging_plots(configMgr, output_folder=f"unfolding")
