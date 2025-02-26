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
    'output/unfolding_v4_Aug2021Talk_nominal/unfold.pkl'
)

# configMgr.out_path = pathlib.Path('/sdf/home/g/gstark/collinearw/output/unfolding_v4')

# lumi = 36.1  # ifb
lumi = 138.861

signals = [
    procset
    for procset in configMgr.process_sets
    if procset.process_type in ["signal", "signal_alt"]
]
unfoldeds = [
    procset for procset in configMgr.process_sets if procset.process_type == 'unfolded'
]

""" do not need to do this for experimental systs
for systematic_name in unfoldeds[1].systematic_names:
    sys_handler.compute_systematics(configMgr, systematic_name, "min_max", process_sets=unfoldeds)
"""

for unfolded in unfoldeds:
    for region in unfolded.nominal:
        for observable in region:
            base_output_path = (
                pathlib.Path(configMgr.out_path)
                .joinpath(
                    "plots", "unfolded", unfolded.name, region.name, observable.name
                )
                .resolve()
            )
            output_path = base_output_path.with_suffix('.pdf')

            # grab all relevant histograms and style them
            observable = unfolding.plot.scale_to_xsec(observable, lumi)
            hReco = observable.root

            # override some weird defaults
            unfolded.nominal.color = 1  # set it to black
            unfolded.nominal.binerror = 0  # set it to 0

            _do_plot_systs = observable.systematic_band is not None
            hReco_error = None
            if observable.systematic_band is not None:
                x_values = observable.bins[:-1] + observable.bin_width / 2.0
                y_values = observable.bin_content[1:-1]
                ex_l = observable.bin_width / 2.0
                ex_h = observable.bin_width / 2.0
                ey_l = np.sqrt(
                    observable.scale_band('experimental')['down'][1:-1] ** 2
                    + observable.scale_band('theory')['down'][1:-1] ** 2
                )
                ey_h = np.sqrt(
                    observable.scale_band('experimental')['up'][1:-1] ** 2
                    + observable.scale_band('theory')['up'][1:-1] ** 2
                )
                hReco_error = ROOT.TGraphAsymmErrors(
                    len(x_values), x_values, y_values, ex_l, ex_h, ey_l, ey_h
                )

                RootBackend.apply_process_styles(hReco_error, unfolded.nominal)
                # see https://root.cern.ch/doc/master/classTGraphPainter.html#GP03
                hReco_error.SetFillColorAlpha(13, 0.35)  # red, transparent
                hReco_error.SetFillStyle(1001)  # solid
                hReco_error.SetTitle("Bkg th. + Expt. unct.")

            RootBackend.apply_process_styles(hReco, unfolded.nominal)

            truths = [
                unfolding.plot.scale_to_xsec(
                    signal.nominal.get_region(region.name).get_histogram(
                        observable.name
                    ),
                    lumi,
                )
                for signal in signals
            ]
            hTruths = [truth.root for truth in truths]

            for hTrue, pset in zip(hTruths, signals):
                RootBackend.apply_process_styles(hTrue, pset.nominal)

            truths_error = []
            for truth_observable, pset in zip(truths, signals):
                if truth_observable.systematic_band is not None:
                    x_values = (
                        truth_observable.bins[:-1] + truth_observable.bin_width / 2.0
                    )
                    y_values = truth_observable.bin_content[1:-1]
                    ex_l = truth_observable.bin_width / 2.0
                    ex_h = truth_observable.bin_width / 2.0
                    ey_l = truth_observable.scale_band('theory')['down'][1:-1]
                    ey_h = truth_observable.scale_band('theory')['up'][1:-1]
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
            )

# unfolding.plot.make_eff_plots(configMgr)
# unfolding.plot.make_debugging_plots(configMgr, output_folder=f"unfolding")
