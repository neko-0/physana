import logging

# import warnings
# import pathlib
# import os
import numpy as np
from pathlib import Path

# from . import metadata
# from .. import abcd
# from ..systematics.core import create_lumi_band
from ... import ConfigMgr
from ...backends import RootBackend
from . import plot

logger = logging.getLogger(__name__)


_reco_truth_map = {
    "electron_inclusive_reco_ABCD-fake-EL-rA_": "electron_inclusive_truth",
    "muon_inclusive_reco_ABCD-fake-MU-rA_": "muon_inclusive_truth",
    "electron_inclusive_2j_reco_ABCD-fake-EL-rA_": "electron_inclusive_2j_truth",
    "muon_inclusive_2j_reco_ABCD-fake-MU-rA_": "muon_inclusive_2j_truth",
    "electron_collinear_reco_ABCD-fake-EL-rA_": "electron_collinear_truth",
    "muon_collinear_reco_ABCD-fake-MU-rA_": "muon_collinear_truth",
    "electron_backtoback_reco_ABCD-fake-EL-rA_": "electron_backtoback_truth",
    "muon_backtoback_reco_ABCD-fake-MU-rA_": "muon_backtoback_truth",
}
_match_map = {
    "electron_inclusive_reco_ABCD-fake-EL-rA_": "electron_inclusive_truth_reco_matching",
    "muon_inclusive_reco_ABCD-fake-MU-rA_": "muon_inclusive_truth_reco_matching",
    "electron_inclusive_2j_reco_ABCD-fake-EL-rA_": "electron_inclusive_2j_truth_reco_matching",
    "muon_inclusive_2j_reco_ABCD-fake-MU-rA_": "muon_inclusive_2j_truth_reco_matching",
    "electron_collinear_reco_ABCD-fake-EL-rA_": "electron_collinear_truth_reco_matching",
    "muon_collinear_reco_ABCD-fake-MU-rA_": "muon_collinear_truth_reco_matching",
    "electron_backtoback_reco_ABCD-fake-EL-rA_": "electron_backtoback_truth_reco_matching",
    "muon_backtoback_reco_ABCD-fake-MU-rA_": "muon_backtoback_truth_reco_matching",
}
_th_name_map = {
    "jet1Pt": "jet1TruthPt",
    "jet2Pt": "jet2TruthPt",
    "lep1Pt": "lep1TruthPt",
    "lep1Eta": "lep1TruthEta",
    "nJet30": "nTruthJet30",
    "nJet100": "nTruthJet100",
    "nBJet30": "nTruthBJet30",
    "wPt": "wTruthPt",
    "Ht30": "HtTruth30",
    "Ht100": "HtTruth100",
    "DeltaRLepJetClosest100": "DeltaRTruthLepJetClosest100",
    "mjj": "mjjTruth",
    "wPt/DeltaPhiWJetClosestPt100": "wTruthPt/DeltaPhiTruthWJetClosestPt100",
    "wPt/DeltaRLepJetClosestPt100": "wTruthPt/DeltaRTruthLepJetClosestPt100",
}


def basic_rescaling(
    input,
    signal="wjets_2211",
    scale_process="data",
    scaler=1.0,
    debug_plot=True,
    prefix="basic_bias",
    reco_truth_map=None,
    th_name_map=None,
    match_map=None,
    rename_signal="wjets_scaled",
    subtract_notmatch=True,
):
    """
    Perform rescaling using the signal MC generators (and data) to prepare
    configuration needed for studying the unfolding bias.

    The generated process (default 'wjets_scaled') is used in the unfolding
    process to estimate the bias.

    """
    config = ConfigMgr.open(input)

    if isinstance(signal, str):
        wjets = config.get_process(signal)
    else:
        wjets = signal

    if isinstance(scale_process, str):
        scale_process = config.get_process(scale_process)
    else:
        scale_process = scale_process

    if reco_truth_map is None:
        reco_truth_map = _reco_truth_map
    if th_name_map is None:
        th_name_map = _th_name_map
    if match_map is None:
        match_map = _match_map

    # calculate the scaling factor. Note signal is the numerator.
    ratio_process = scale_process / wjets
    ratio_process.name = "kfactor"
    ratio_process.process_type = "kfactor"

    # create copy to use for storing the corrected RECO wjets
    wjets_corrected = wjets.copy()
    wjets_corrected.name = rename_signal
    wjets_corrected.process_type = 'signal_alt'
    for region in wjets_corrected.regions:
        # skipping non reco type
        if region.type in {"reco_match", "reco_match_not", "truth"}:
            continue
        # skip if not in the reco truth map
        if region.name not in reco_truth_map:
            logger.warning(f"reco truth map cannot find {region.name}")
            continue
        for histo in region.histograms:
            # if not 1D, don't touch. i.e. don't change the response matrix etc
            if histo.hist_type != "1d":
                continue
            try:
                th_name = th_name_map[histo.name]
            except KeyError:
                logger.warning(f"th name map cannot find {histo.name}")
                continue

            if subtract_notmatch:
                sf = scale_process.get(region.name).get(histo.name).copy()
                mes_wjets = wjets.get(region.name).get(histo.name)
                # th_wjets = wjets.get(reco_truth_map[region.name]).get(th_name)
                res_wjets = wjets.get(match_map[region.name]).get(
                    f"response_matrix_{histo.name}"
                )
                hNotmatch = mes_wjets - res_wjets.project_x()
                sf.sub(hNotmatch)
                sf.div(res_wjets.project_x())
                # reweighted_reco_wjets = (res_wjets.bin_content * sf.bin_content).sum(axis=0)
                # print(reweighted_reco_wjets)
                # breakpoint()
                # print(mes_wjets.bin_content)
                # print(scale_process.get(region.name).get(histo.name).bin_content)
            else:
                # this is the scaling factors. apply it to the reco histogram
                sf = ratio_process.get(region.name).get(histo.name)
                sf.mul(scaler)  # additional constant scaler, default is 1.0

            sf.nan_to_num()
            histo.mul(sf)
            histo.nan_to_num()

            # if subtract_notmatch:
            #     histo.bin_content = reweighted_reco_wjets
            #     histo.nan_to_num()

            # transform the scaling factors from reco to truth level.
            th = wjets_corrected.get(reco_truth_map[region.name]).get(th_name)
            res = (
                wjets.get(match_map[region.name])
                .get(f"response_matrix_{histo.name}")
                .bin_content.T
            )

            res_wjets = wjets.get(match_map[region.name]).get(
                f"response_matrix_{histo.name}"
            )
            # STILL TESTING
            if subtract_notmatch:
                origin_th_project = res_wjets.project_y().bin_content
                scaled_res_wjets = res_wjets.copy()
                xshape = scaled_res_wjets.bin_content.shape[0]
                for x in range(xshape):
                    scaled_res_wjets.bin_content[x, :] *= sf.bin_content[x]
                # scaled_res_wjets.bin_content *= sf.bin_content
                new_th_project = scaled_res_wjets.project_y().bin_content
                th_sf = new_th_project / origin_th_project
                th.bin_content *= th_sf

                # r_th_match = res_wjets.project_y().bin_content / th.bin_content
                # r_reweight_reco = th.bin_content / mes_wjets.bin_content
                # breakpoint()

            res_sum = res.sum(axis=1)
            unit_res = res / res_sum[:, None]
            unit_res = np.nan_to_num(unit_res)
            transform_sf = th.copy()
            transform_sf.bin_content = unit_res.dot(sf.bin_content)
            transform_sf.sumW2 = unit_res.dot(sf.sumW2)
            transform_sf.nan_to_num()

            # apply the transformed scaling factor to truth level.
            th.mul(transform_sf)
            th.nan_to_num()
            # ratio_th = ratio_process.get(reco_truth_map[region.name]).get(th_name)
            # ratio_th.bin_content = transform_sf.bin_content

            if debug_plot:
                parent_path = f"{prefix}_{signal}_basic_bias_debug_plot/{region.name}"
                sf_output = Path(f"{parent_path}/sf_{histo.name}.png")
                res_output = Path(f"{parent_path}/res_{histo.name}.png")
                transform_sf_output = Path(
                    f"{parent_path}/transform_sf_{histo.name}.png"
                )
                diff_output = Path(f"{parent_path}/diff_sf_{histo.name}.png")
                truth_output = Path(f"{parent_path}/truth_{histo.name}.png")

                sf_output.parent.mkdir(parents=True, exist_ok=True)

                canvas = RootBackend.make_canvas(str(sf_output))
                canvas.cd()
                rh_sf = sf.root
                rh_sf.GetYaxis().SetTitle("SF_{reco}")
                rh_sf.GetYaxis().SetTitleOffset(1.5)
                rh_sf.GetYaxis().SetRangeUser(0, 2.0)
                rh_sf.SetLineWidth(2)
                rh_sf.Draw("HIST")
                canvas.SaveAs(str(sf_output))

                canvas = RootBackend.make_canvas(str(res_output))
                canvas.cd()
                unit_res_hist = (
                    wjets.get(match_map[region.name])
                    .get(f"response_matrix_{histo.name}")
                    .copy()
                )
                # unit_res_hist.bin_content = unit_res
                zrange = (
                    np.min(unit_res_hist.bin_content),
                    np.max(unit_res_hist.bin_content),
                )
                plot.plot_response(
                    unit_res_hist.root,
                    output_path=res_output,
                    normalise=False,
                    zrange=zrange,
                )

                canvas = RootBackend.make_canvas(str(transform_sf_output))
                canvas.cd()
                rh_transform_sf = transform_sf.root
                rh_transform_sf.GetYaxis().SetTitle("SF_{truth}")
                rh_transform_sf.GetYaxis().SetTitleOffset(1.5)
                rh_transform_sf.GetYaxis().SetRangeUser(0, 2.0)
                rh_transform_sf.SetLineWidth(2)
                rh_transform_sf.Draw("H")
                canvas.SaveAs(str(transform_sf_output))

                canvas = RootBackend.make_canvas(str(diff_output))
                canvas.cd()
                diff_h = (transform_sf - sf) / sf
                diff_h.nan_to_num()
                diff_h.bin_content = np.abs(diff_h.bin_content)
                rh_diff_h = diff_h.root
                rh_diff_h.GetYaxis().SetTitle("(SF_{truth}-SF_{reco})/SF_{reco}")
                rh_diff_h.GetYaxis().SetTitleOffset(1.5)
                rh_diff_h.GetYaxis().SetRangeUser(-1.0, 1.0)
                rh_diff_h.SetLineWidth(2)
                rh_diff_h.Draw("H")
                canvas.SaveAs(str(diff_output))

                canvas = RootBackend.make_canvas(str(truth_output))
                canvas.cd()
                origin_th = wjets.get(reco_truth_map[region.name]).get(th_name).copy()
                rh_origin_th = origin_th.root
                rh_origin_th.SetLineColor(2)
                rh_origin_th.Draw("H")
                rh_th = th.root
                rh_th.SetLineColor(3)
                rh_th.Draw("Hsame")
                canvas.SaveAs(str(truth_output))

    config.append_process(wjets_corrected)
    config.append_process(ratio_process)

    return config.save(f"basic_unfolding_scaling_{input.ofilename}.pkl")


def cross_rescaling(
    input,
    signal="wjets_2211",
    alt_signal="wjets_FxFx",
    debug_plot=True,
    prefix="hidden_bias",
    reco_truth_map=None,
    th_name_map=None,
    match_map=None,
    reco_to_truth=True,
    rename_signal="wjets_scaled",
):
    """
    Perform rescaling using althernative MC generators and prepare configuration
    needed for studying the unfolding bias.

    The generated process (default 'wjets_scaled') is used in the unfolding
    process to estimate the bias.

    """
    config = ConfigMgr.open(input)

    if isinstance(signal, str):
        wjets = config.get_process(signal)
    else:
        wjets = signal

    if isinstance(alt_signal, str):
        alt_signal = config.get_process(alt_signal)
    else:
        alt_signal = alt_signal

    if reco_truth_map is None:
        reco_truth_map = _reco_truth_map
    if th_name_map is None:
        th_name_map = _th_name_map
    if match_map is None:
        match_map = _match_map

    # calculate the scaling factor. Note signal is the numerator.
    ratio_process = alt_signal / wjets
    ratio_process.name = "kfactor"
    ratio_process.process_type = "kfactor"

    # create copy to use for storing the corrected RECO wjets
    wjets_corrected = wjets.copy()
    wjets_corrected.name = rename_signal
    wjets_corrected.process_type = 'signal_alt'
    for region in wjets_corrected.regions:
        # skipping non reco type
        if region.type in {"reco_match", "reco_match_not", "truth"}:
            continue
        # skip if not in the reco truth map
        if region.name not in reco_truth_map:
            continue
        for histo in region.histograms:
            # if not 1D, don't touch. i.e. don't change the response matrix etc
            if histo.hist_type != "1d":
                continue
            try:
                th_name = th_name_map[histo.name]
            except KeyError:
                continue

            # this is the scaling factors. apply it to the reco histogram
            if reco_to_truth:
                sf = ratio_process.get(region.name).get(histo.name)
                sf.nan_to_num()
                histo.mul(sf)
                histo.nan_to_num()

                # transform the scaling factors from reco to truth level.
                th = wjets_corrected.get(reco_truth_map[region.name]).get(th_name)
                res = (
                    alt_signal.get(match_map[region.name])
                    .get(f"response_matrix_{histo.name}")
                    .bin_content.T
                )
                res_sum = res.sum(axis=1)
                unit_res = res / res_sum[:, None]
                unit_res = np.nan_to_num(unit_res)
                transform_sf = th.copy()
                transform_sf.bin_content = unit_res.dot(sf.bin_content)
                transform_sf.sumW2 = unit_res.dot(sf.sumW2)
                transform_sf.nan_to_num()

                # apply the transformed scaling factor to truth level.
                th.mul(transform_sf)
                th.nan_to_num()
                ratio_th = ratio_process.get(reco_truth_map[region.name]).get(th_name)
                ratio_th.bin_content = transform_sf.bin_content
            else:
                sf = ratio_process.get(reco_truth_map[region.name]).get(th_name)
                sf.nan_to_num()

                # apply scaling on truth
                th = wjets_corrected.get(reco_truth_map[region.name]).get(th_name)
                th.mul(sf)
                th.nan_to_num()

                res = (
                    alt_signal.get(match_map[region.name])
                    .get(f"response_matrix_{histo.name}")
                    .bin_content
                )
                res_sum = res.sum(axis=1)
                unit_res = res / res_sum[:, None]
                unit_res = np.nan_to_num(unit_res)
                transform_sf = sf.copy()
                transform_sf.bin_content = unit_res.dot(sf.bin_content)
                transform_sf.nan_to_num()

                histo.mul(transform_sf)
                histo.nan_to_num()

            if debug_plot:
                parent_path = f"{prefix}_{signal}_hidden_bias_debug_plot/{region.name}"
                sf_output = Path(f"{parent_path}/sf_{histo.name}.png")
                res_output = Path(f"{parent_path}/res_{histo.name}.png")
                transform_sf_output = Path(
                    f"{parent_path}/transform_sf_{histo.name}.png"
                )
                diff_output = Path(f"{parent_path}/diff_sf_{histo.name}.png")
                truth_output = Path(f"{parent_path}/truth_{histo.name}.png")

                sf_output.parent.mkdir(parents=True, exist_ok=True)

                canvas = RootBackend.make_canvas(str(sf_output))
                canvas.cd()
                rh_sf = sf.root
                rh_sf.GetYaxis().SetTitle("SF_{reco}")
                rh_sf.GetYaxis().SetTitleOffset(1.5)
                rh_sf.GetYaxis().SetRangeUser(0, 2.0)
                rh_sf.SetLineWidth(2)
                rh_sf.Draw("HIST")
                canvas.SaveAs(str(sf_output))

                canvas = RootBackend.make_canvas(str(res_output))
                canvas.cd()
                unit_res_hist = (
                    wjets.get(match_map[region.name])
                    .get(f"response_matrix_{histo.name}")
                    .copy()
                )
                # unit_res_hist.bin_content = unit_res
                zrange = (
                    np.min(unit_res_hist.bin_content),
                    np.max(unit_res_hist.bin_content),
                )
                plot.plot_response(
                    unit_res_hist.root,
                    output_path=res_output,
                    normalise=False,
                    zrange=zrange,
                )

                canvas = RootBackend.make_canvas(str(transform_sf_output))
                canvas.cd()
                rh_transform_sf = transform_sf.root
                rh_transform_sf.GetYaxis().SetTitle("SF_{truth}")
                rh_transform_sf.GetYaxis().SetTitleOffset(1.5)
                rh_transform_sf.GetYaxis().SetRangeUser(0, 2.0)
                rh_transform_sf.SetLineWidth(2)
                rh_transform_sf.Draw("H")
                canvas.SaveAs(str(transform_sf_output))

                canvas = RootBackend.make_canvas(str(diff_output))
                canvas.cd()
                diff_h = (transform_sf - sf) / sf
                diff_h.nan_to_num()
                diff_h.bin_content = np.abs(diff_h.bin_content)
                rh_diff_h = diff_h.root
                rh_diff_h.GetYaxis().SetTitle("(SF_{truth}-SF_{reco})/SF_{reco}")
                rh_diff_h.GetYaxis().SetTitleOffset(1.5)
                rh_diff_h.GetYaxis().SetRangeUser(-1.0, 1.0)
                rh_diff_h.SetLineWidth(2)
                rh_diff_h.Draw("H")
                canvas.SaveAs(str(diff_output))

                canvas = RootBackend.make_canvas(str(truth_output))
                canvas.cd()
                origin_th = wjets.get(reco_truth_map[region.name]).get(th_name).copy()
                rh_origin_th = origin_th.root
                rh_origin_th.SetLineColor(2)
                rh_origin_th.Draw("H")
                rh_th = th.root
                rh_th.SetLineColor(3)
                rh_th.Draw("Hsame")
                canvas.SaveAs(str(truth_output))

    config.append_process(wjets_corrected)
    config.append_process(ratio_process)

    return config.save(f"hidden_unfolding_scaling_{input.ofilename}.pkl")
