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


def explore_unfold(ifile, output="unfolding_plots", skip_processes=[]):
    configMgr = ConfigMgr.open(ifile)

    # lumi = 36.1  # ifb
    lumi = 138.861

    signals = [
        procset
        for procset in configMgr.process_sets
        if procset.process_type in ["signal", "signal_alt"] and procset.name not in skip_processes
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
                        output, "unfolded", unfolded.name, region.name, observable.name
                    )
                    .resolve()
                )
                output_path = base_output_path.with_suffix('.pdf')

                # grab all relevant histograms and style them
                c_observable = observable.copy()
                observable = unfolding.plot.scale_to_xsec(observable, lumi)
                hReco = observable.root

                # override some weird defaults
                unfolded.nominal.color = 1  # set it to black
                unfolded.nominal.binerror = 0  # set it to 0

                #print(f"syst band {observable.name} {observable.systematic_band}")

                _do_plot_systs = observable.systematic_band is not None
                hReco_error = None
                _error_band = observable.total_band()
                if _error_band is not None:
                    bin_width = np.diff(observable.bins) / 2.0
                    x_values = observable.bins[:-1] + bin_width
                    y_values = observable.bin_content[1:-1]
                    ex_l = (observable.bin_width / 2.0)[1:-1]
                    ex_h = (observable.bin_width / 2.0)[1:-1]
                    _error_band = c_observable.total_band()
                    _error = _error_band.scale_nominal(c_observable.bin_content)
                    _error["down"] = _error["down"]*( observable.bin_content / c_observable.bin_content)
                    _error["up"] = _error["up"]*( observable.bin_content / c_observable.bin_content)
                    ey_l = np.nan_to_num(_error["down"][1:-1])
                    ey_h = np.nan_to_num(_error["up"][1:-1])
                    assert x_values.shape == y_values.shape
                    assert y_values.shape == ey_l.shape == ey_h.shape
                    assert x_values.shape == ex_l.shape == ex_h.shape
                    x_values = x_values.astype("float")
                    y_values = y_values.astype("float")
                    ex_l = ex_l.astype("float")
                    ex_h = ex_h.astype("float")
                    ey_l = ey_l.astype("float")
                    ey_h = ey_h.astype("float")
                    hReco_error = ROOT.TGraphAsymmErrors(
                        len(x_values), x_values, y_values, ex_l, ex_h, ey_l, ey_h
                    )

                    RootBackend.apply_process_styles(hReco_error, unfolded.nominal)
                    # see https://root.cern.ch/doc/master/classTGraphPainter.html#GP03
                    hReco_error.SetFillColorAlpha(13, 0.35)  # red, transparent
                    hReco_error.SetFillStyle(1001)  # solid
                    hReco_error.SetTitle("Bkg th. + Expt. unct.")

                RootBackend.apply_process_styles(hReco, unfolded.nominal)

                c_truths = [
                    signal.nominal.get_region(region.name).get_histogram(
                        observable.name
                    ).copy() for signal in signals
                ]

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
                for truth_observable, pset, c_truth in zip(truths, signals, c_truths):
                    #print(f"truth syst band {truth_observable.name} {truth_observable.systematic_band}")
                    if pset.process_type != "signal":
                        _error_band = None
                    else:
                        _error_band = c_truth.total_band(exclude_types={"experimental"})
                    if _error_band is not None:
                        bin_width = np.diff(truth_observable.bins) / 2.0
                        x_values = truth_observable.bins[:-1] + bin_width
                        y_values = observable.bin_content[1:-1]
                        y_values = truth_observable.bin_content[1:-1]
                        ex_l = (truth_observable.bin_width / 2.0)[1:-1]
                        ex_h = (truth_observable.bin_width / 2.0)[1:-1]
                        try:
                            _error = _error_band.scale_nominal(c_truth.bin_content)
                            _error["down"] = _error["down"]*( truth_observable.bin_content / c_truth.bin_content)
                            _error["up"] = _error["up"]*( truth_observable.bin_content / c_truth.bin_content)
                            ey_l = np.nan_to_num(_error["down"][1:-1])
                            ey_h = np.nan_to_num(_error["up"][1:-1])
                            assert x_values.shape == y_values.shape
                            assert y_values.shape == ey_l.shape == ey_h.shape
                            assert x_values.shape == ex_l.shape == ex_h.shape
                            x_values = x_values.astype("float")
                            y_values = y_values.astype("float")
                            ex_l = ex_l.astype("float")
                            ex_h = ex_h.astype("float")
                            ey_l = ey_l.astype("float")
                            ey_h = ey_h.astype("float")
                        except KeyError:
                            ey_l = truth_observable.bin_content[1:-1]
                            ey_h = truth_observable.bin_content[1:-1]
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
                    #logy=False,
                    #yrange=(0,500),
                )

if __name__ == "__main__":
    explore_unfold("band_unfold.pkl", "unfold_plots", skip_processes=["wjets_2211_EXPASSEW", "wjets_2211_MULTIASSEW"])
    # unfolding.plot.make_eff_plots(configMgr)
    # unfolding.plot.make_debugging_plots(configMgr, output_folder=f"unfolding")
