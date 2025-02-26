import logging
import warnings
import pathlib
import numpy as np
import os
from . import metadata
from .. import abcd
from ..systematics.core import create_lumi_band
from ... import core, utils, plotMaker
from ...backends import RootBackend
from ...histManipulate import rbin_merge
from . import plot

try:
    import ROOT
except ImportError:
    warnings.warn("Cannot import ROOT module!")

log = logging.getLogger(__name__)
log.setLevel(logging.WARNING)


# https://twiki.cern.ch/twiki/bin/view/Atlas/LuminosityForPhysics#2018_13_TeV_proton_proton_Morion
def scale_to_xsec(hist, lumi, lumiunc=0.0, binwidth=True):
    # divide through by lumi and bin-width
    if lumi <= 0:
        return hist

    _y_title = "d#sigma/dx [fb / binning]"

    if isinstance(hist, core.HistogramBase):
        newhist = hist.copy()
        newhist.ytitle = _y_title
        if binwidth:
            _scalefactor = lumi * newhist.bin_width
        else:
            _scalefactor = lumi
        newhist.bin_content = newhist.bin_content / _scalefactor

        # With luminosity uncertainty (spelt out...)

        create_lumi_band(newhist, lumiunc, "xsec")
        newhist.sumW2 = (
            newhist.bin_content * (np.sqrt(hist.sumW2) / hist.bin_content)
        ) ** 2

        # newhist.sumW2 = (
        #     newhist.bin_content
        #     * np.sqrt(
        #         (np.sqrt(hist.sumW2) / hist.bin_content) ** 2
        #         + ((lumiunc * lumi) / lumi) ** 2
        #     )
        # ) ** 2
        # newhist.sumW2 = (
        #    newhist.bin_content * (np.sqrt(hist.sumW2) / hist.bin_content)
        # ) ** 2
        return newhist

    elif isinstance(hist, ROOT.TH1):
        newhist = hist.Clone()
        newhist.GetYaxis().SetTitle(_y_title)
        _min_newval = 1.0
        for idx, val in enumerate(hist):
            if binwidth:
                _scalefactor = lumi * hist.GetBinWidth(idx)
            else:
                _scalefactor = lumi
            newval = val / _scalefactor
            newerr = hist.GetBinError(idx) / _scalefactor
            newhist.SetBinContent(idx, newval)
            newhist.SetBinError(idx, newerr)
            if newval > 0:
                _min_newval = min(newval, _min_newval)

        newhist.GetYaxis().SetRangeUser(_min_newval / 10.0, max(newhist) * 10.0)
        return newhist
    else:
        ValueError(f'No idea how to handle a histogram of type {type(hist)}')


def unpack_results(
    region,
    varname,
    measured,
    truths,
    output_path,
    lumi,
    lumi_unc,
    layout,
    style,
    title_map=None,
    exclude_names=None,  # exclude systematic bands
    ratio_ytitle="Pred./Data",
    data_label=None,
    include_stats=True,
    normalise=False,
    legend_args=(0.45, 0.45, 0.85, 0.85, 0.05),
    ratio_legend_args=(0.45, 0.45, 0.85, 0.85, 0.05),
    fixed_pos=None,
    hist_modfunc=None,
    disable_top=None,
    overlay_systematics=None,
):
    """
    Plot the unfolded data spectrum and truth spectra, with a ratio panel.

    Parameters
    ----------
    region : str
        Region name.
    varname : str
        Variable name.
    measured : CollinearUnfold.core.Measurement
        Measured spectrum.
    truths : list[CollinearUnfold.core.Measurement]
        List of truth spectra.
    output_path : str
        Output path.
    lumi : float
        Luminosity in fb^-1.
    lumi_unc : float
        Relative luminosity uncertainty.
    layout : list[list[int]]
        Layout of the ratio panels.
    style : dict
        Style dictionary.
    title_map : dict, optional
        Title map (default is None).
    exclude_names : list[str], optional
        List of systematic names to exclude (default is None).
    ratio_ytitle : str, optional
        Y-axis title of the ratio panels (default is "Pred./Data").
    data_label : str, optional
        Data label (default is None).
    include_stats : bool, optional
        Include statistical uncertainty in the ratio panels (default is True).
    normalise : bool, optional
        Normalise the spectra (default is False).
    legend_args : tuple, optional
        Legend arguments (default is (0.45, 0.45, 0.85, 0.85, 0.05)).
    ratio_legend_args : tuple, optional
        Ratio legend arguments (default is (0.45, 0.45, 0.85, 0.85, 0.05)).
    fixed_pos : list[int], optional
        List of fixed positions (default is None).
    hist_modfunc : callable, optional
        Histogram modification function (default is None).
    disable_top : list[str], optional
        List of truth names to disable in the top panel (default is None).
    overlay_systematics : dict, optional
        Dictionary of single systematics component to overlay on total systematics (default is None).
    """
    logy = True
    minMain = 0.001
    maxMain = 1000.0
    minRatio = 0.6
    maxRatio = 1.4
    ylabel = ""
    bin_labels = None
    xrange = None
    disable_top_syst = False

    if overlay_systematics is None:
        overlay_systematics = {}

    key = f'{region}_{varname}'
    if style.get(key):
        logy = style[key]['log']
        minMain = style[key]['minMain']
        maxMain = style[key]['maxMain']
        minRatio = style[key]['minRatio']
        maxRatio = style[key]['maxRatio']
        ylabel = style[key].get("yaxis")
        bin_labels = style[key].get("bin_labels")
        xrange = style[key].get("xrange", None)
        disable_top_syst = style[key].get("disable_top_syst", False)

    if not ylabel:
        ylabel = "d#sigma/dx [fb / binning]"

    for truth in truths:
        truth_h = truth.get_histogram(varname)
        while len(truth_h.bins) > len(measured.get_histogram(varname).bins):
            t_last_bins = truth_h.bins[-2:]
            m_last_bins = measured.get_histogram(varname).bins[-2:]
            name_str = f"{truth.name}(T){t_last_bins} vs {measured.name}(M){m_last_bins}, {varname}"
            log.warning(f"{name_str} BIN SIZE MISMATCH (T>M), MERGING!")
            rbin_merge(truth_h, True)
        while len(truth_h.bins) < len(measured.get_histogram(varname).bins):
            t_last_bins = truth_h.bins[-2:]
            m_last_bins = measured.get_histogram(varname).bins[-2:]
            name_str = f"{truth.name}(T){t_last_bins} vs {measured.name}(M){m_last_bins}, {varname}"
            log.warning(f"{name_str} BIN SIZE MISMATCH (T<M), MERGING!")
            rbin_merge(measured.get_histogram(varname), True)

    if normalise:
        _h = measured.get_histogram(varname)
        _h.div(_h.integral("width"))
        for truth in truths:
            _h = truth.get_histogram(varname)
            _h.div(_h.integral("width"))

    # We need a few things:
    #   1) Data points, containing the statistical uncertainty
    #   2) The uncertainty in (1) converted into a relative value, for plotting in the ratio panels
    #   3) Hashed band with systematic uncertainties associated to the data measurement
    #   4) The uncertainty in (3) coverted into a relative value, for plotting in the ratio panels
    observable = scale_to_xsec(measured.get_histogram(varname), lumi, lumi_unc)

    xlabel = observable.xtitle

    # This is (1)
    hMeasured = observable.root_graph(
        "Statistical", exclude_names=exclude_names, bin_labels=bin_labels
    )
    RootBackend.apply_styles(
        hMeasured, color=1, linewidth=1, linestyle=2, markerstyle=8, markersize=1.5
    )
    hMeasured.GetYaxis().SetTitleOffset(1.65)
    hMeasured.GetYaxis().SetTitle(ylabel)
    hMeasured.GetXaxis().SetTitle(xlabel)

    # This is (2)
    hMeasured_relStatError = observable.root_graph(
        "Relative-Statistical", exclude_names=exclude_names, bin_labels=bin_labels
    )
    RootBackend.apply_styles(hMeasured_relStatError, color=1, linewidth=1, markersize=0)

    # This is (3)
    hMeasured_sysError = observable.root_graph(
        "Systematic",
        exclude_names=exclude_names,
        include_stats=include_stats,
        bin_labels=bin_labels,
    )
    RootBackend.apply_styles(hMeasured_sysError, color=17)
    hMeasured_sysError.SetFillColor(17)

    # This is (4)
    hMeasured_relSysError = observable.root_graph(
        "Relative-Systematic",
        exclude_names=exclude_names,
        include_stats=include_stats,
        bin_labels=bin_labels,
    )
    hMeasured_relSysError.SetFillColor(17)
    hMeasured_relSysError.SetFillStyle(1001)

    canvas = RootBackend.make_canvas(f"RooUnfold_{output_path}", num_ratio=len(layout))
    ROOT.gStyle.SetErrorX(0)

    # top pad
    canvas.cd(1)
    if logy:
        canvas.GetPad(1).SetLogy()

    RootBackend.set_range(hMeasured, yrange=(minMain, maxMain), top_room_scale=1e2)
    RootBackend.clip_graph_value(hMeasured, 0)  # remove all zeros from the graph

    leg = RootBackend.make_legend(*legend_args)

    leg.AddEntry(hMeasured, data_label or "Data (stat. error)", "PE1X0")
    if include_stats:
        leg.AddEntry(hMeasured_sysError, "Stat. #oplus syst. unc.", "F")
    else:
        leg.AddEntry(hMeasured_sysError, "Systematic uncertainty", "F")

    if xrange:
        hMeasured.GetXaxis().SetRangeUser(*xrange)

    hMeasured.Draw("APE")
    hMeasured_sysError.Draw("same2")

    hTruths = []
    rdump = []
    for truth in truths:
        _hTruth = scale_to_xsec(truth.get_histogram(varname), lumi)

        hTruths.append(_hTruth)

        if hist_modfunc:
            _hTruth = hist_modfunc(_hTruth)

        pos = truths.index(truth) + 1
        multiplier = len(truths)
        # fixed the position if specified.
        if fixed_pos and pos - 1 in fixed_pos:
            pos = -1
            multiplier = 1
        elif fixed_pos:
            pos += 1

        hTruth = _hTruth.root_graph(
            "Systematic" if not disable_top_syst else "",
            exclude_names=exclude_names,
            multiplier=multiplier,
            position=pos,
            bin_labels=bin_labels,
            include_stats=include_stats,
        )
        if title_map:
            _default_title = _hTruth.parent.parent.title
            _title = title_map.get(_hTruth.parent.parent.name, _default_title)
        RootBackend.apply_process_styles(hTruth, _hTruth.parent.parent)
        RootBackend.clip_graph_value(hTruth, 0)  # remove all zeros from the graph
        hTruth.SetLineColor(_hTruth.parent.parent.color)
        hTruth.SetLineWidth(2)
        hTruth.SetMarkerSize(1.5)
        rdump.append(hTruth)
        if disable_top and truth.parent.name not in disable_top:
            hTruth.Draw("PE same")
            leg.AddEntry(hTruth, _title, "PE1X0")

    hMeasured.Draw("PE same")

    leg.Draw("SAME")
    RootBackend.make_atlas_label()

    # these are default RootBackend.make_canvas setting
    canvas.cd()
    top_heights = 800
    ratio_heights = (len(layout) + 1) * 200
    pad_frac = ratio_heights / (ratio_heights + top_heights)
    newpad = ROOT.TPad("newpad", "a transparent pad", 0, 0, 1, 1)
    newpad.SetFillStyle(4000)
    newpad.Draw()
    newpad.cd()
    ratio_label_t = ROOT.TLatex()
    ratio_label_t.SetTextFont(42)
    ratio_label_t.SetTextAngle(90)
    ratio_label_t.SetTextSize(0.03)
    if len(layout) == 1:
        label_t_loc = pad_frac * 0.65
    else:
        label_t_loc = pad_frac * 0.45
    ratio_label_t.DrawLatex(0.065, label_t_loc, ratio_ytitle)

    ratios = []
    dashed_lines = []
    ratio_legends = {}
    for pad_idx, truth_idxs in enumerate(layout, 2):
        if key in style:
            m_minRatio = style[key].get(f'minRatio{pad_idx-1}', minRatio)
            m_maxRatio = style[key].get(f'maxRatio{pad_idx-1}', maxRatio)
        else:
            m_minRatio = minRatio
            m_maxRatio = maxRatio
        canvas.cd(pad_idx)
        hMeasured_relSysError.SetTitle("")
        hMeasured_relSysError.GetYaxis().SetTitle("")
        # if pad_idx == len(layout) + 1:
        #     hMeasured_relSysError.GetYaxis().SetTitle(ratio_ytitle)
        #     hMeasured_relSysError.GetYaxis().SetTitleSize(30)
        hMeasured_relSysError.GetYaxis().SetRangeUser(m_minRatio, m_maxRatio)
        hMeasured_relSysError.GetYaxis().SetNdivisions(5)
        if xrange:
            hMeasured_relSysError.GetXaxis().SetRangeUser(*xrange)
        RootBackend.apply_font_styles(hMeasured_relSysError.GetXaxis(), titleoffset=1.5)
        hMeasured_relSysError.DrawClone("a2")
        hMeasured_relStatError.Draw("Psame")

        if disable_top and pad_idx not in ratio_legends:
            canvas.GetPad(pad_idx).cd()
            if isinstance(ratio_legend_args, dict):
                rleg_args = ratio_legend_args.get(
                    pad_idx, list(ratio_legend_args.values())[0]
                )
            else:
                rleg_args = ratio_legend_args
            ratio_legends[pad_idx] = RootBackend.make_legend(*rleg_args)
            ratio_legends[pad_idx].Draw()

        for truth_idx in truth_idxs:
            try:
                hratio = hTruths[truth_idx] / observable
            except ValueError as _err:
                if len(hTruths[truth_idx].bins) == len(observable.bins):
                    raise _err
                log.warning(f"RATIO BINNING MISMATCH {observable.name}")
                while len(hTruths[truth_idx].bins) > len(observable.bins):
                    last_bin = (
                        f"{hTruths[truth_idx].bins[-3:]} vs {observable.bins[-3:]}"
                    )
                    log.warning(f"T > M, {last_bin}")
                    rbin_merge(hTruths[truth_idx], True)
                while len(hTruths[truth_idx].bins) < len(observable.bins):
                    last_bin = (
                        f"{hTruths[truth_idx].bins[-3:]} vs {observable.bins[-3:]}"
                    )
                    log.warning(f"T < M, {last_bin}")
                    rbin_merge(observable, True)
                hratio = hTruths[truth_idx] / observable

            """
            up = observable.total_band(exclude_names=exclude_names)["up"]
            down = observable.total_band(exclude_names=exclude_names)["down"]

            for b, (b_content, b_sumw2, b_down, b_up) in enumerate(
              zip(observable.bin_content[1:-1], observable.sumW2[1:-1], down[1:-1], up[1:-1])
            ):
              err = (b_down * b_content)
              alt = hTruths[truth_idx].bin_content[b+1]
              print(f"Combined {b_content} and individual alt {alt} and combined uncertainty {err}")
              hratio.bin_content[b] = (b_content - alt) / err
            ratio = hratio.root_graph("", exclude_names=exclude_names)
            """

            if hist_modfunc:
                hratio = hist_modfunc(hratio)

            pos = truth_idx + 1
            multiplier = len(truths)
            # fixed the position if specified.
            if fixed_pos and pos - 1 in fixed_pos:
                pos = -1
                multiplier = 1
            elif fixed_pos:
                pos += 1

            hratio.nan_to_num()

            truth_parent = hTruths[truth_idx].parent.parent
            # handle overlaying of single systematics component on total systematics.
            _ratio = None
            ratio_draw_opts = "PE SAME"
            lcolor = truth_parent.color
            if truth_parent.name in overlay_systematics:
                lwidth = 1
                lcolor = 2
                markersize = 0.8
                ratio_draw_opts = "PZ SAME"
                # first draw the total systematics
                # mean_binwidth = np.min(hratio.bin_width) * 0.07
                mean_binwidth = 0
                _ratio = hratio.copy().root_graph(
                    "Systematic",
                    exclude_names=exclude_names,
                    multiplier=multiplier,
                    position=pos,
                    bin_labels=bin_labels,
                    include_stats=include_stats,
                    fixed_multiplier_binwidth=mean_binwidth,
                )
                RootBackend.apply_process_styles(
                    _ratio, truth_parent, alpha=0.05, fillstyle=4000, fillcolor=False
                )
                _ratio.SetMarkerSize(0.8)
                _ratio.SetLineWidth(3)
                ratios.append(_ratio)
                _ratio.Draw("EZ SAME")
                _exclude = [
                    x
                    for x in hratio.systematic_band
                    if x not in overlay_systematics[truth_parent.name]
                ]
                if exclude_names:
                    _exclude += exclude_names
            else:
                lwidth = 2
                markersize = 1.5
                _exclude = exclude_names

            ratio = hratio.root_graph(
                "Systematic",
                exclude_names=_exclude,
                multiplier=multiplier,
                position=pos,
                bin_labels=bin_labels,
                include_stats=False if _ratio else include_stats,
                fixed_multiplier_binwidth=0 if _ratio else None,
            )
            RootBackend.apply_process_styles(ratio, truth_parent)
            ratio.SetFillColorAlpha(lcolor, 0.25)
            ratio.SetLineColor(lcolor)
            ratio.SetLineWidth(lwidth)
            ratio.SetMarkerSize(markersize)
            ratios.append(ratio)
            ratio.Draw(ratio_draw_opts)

            if disable_top and truth_parent.name in disable_top:
                ratio_legends[pad_idx].AddEntry(
                    _ratio or ratio, truth_parent.title, "PE1X0"
                )

        hMeasured_relStatError.Draw("Psame")

        canvas.Update()
        line = ROOT.TLine(
            canvas.GetPad(1).GetUxmin(), 1, canvas.GetPad(1).GetUxmax(), 1
        )
        line.SetLineColor(ROOT.kBlack)
        line.SetLineStyle(ROOT.kDashed)
        line.Draw('SAME')
        dashed_lines.append(line)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with utils.all_redirected():
        canvas.SaveAs(str(output_path))


def plot_results(
    measured,
    truths,
    reco,
    output_path,
    draw_meas=False,
    additionalRegions=None,
    reco_error=None,
    truths_error=None,
    logy=True,
    yrange=None,
    layout=None,
):
    """
    layout is a mapping of index of truth to ratio panels
    """

    additionalRegions = additionalRegions or []
    truths_error = truths_error or []
    layout = layout or [list(range(len(truths)))]

    hMeas = measured.Clone() if measured else None
    hTruths = [truth.Clone() for truth in truths]
    hReco = reco.Clone()

    canvas = RootBackend.make_canvas(f"RooUnfold_{output_path}", num_ratio=len(layout))
    if logy:
        canvas.GetPad(1).SetLogy()
    leg = RootBackend.make_legend()

    # top pad
    canvas.cd(1)
    RootBackend.set_range(hReco, yrange=yrange, top_room_scale=1e2)

    canvas.SetTitle("")

    for truth_error in truths_error:
        if not truth_error:
            continue
        truth_error.Draw("2p")

    for hTrue in hTruths:
        hTrue.Draw("SAME")

    if truths_error:
        for hTrue, truth_error in zip(hTruths, truths_error):
            leg.AddEntry(hTrue)
            if not truth_error:
                continue
            leg.AddEntry(truth_error, truth_error.GetTitle(), "f")
    else:
        for hTrue in hTruths:
            leg.AddEntry(hTrue)

    if draw_meas and hMeas:
        hMeas.Draw("SAME")
        leg.AddEntry(hMeas, f"Measured {hMeas.GetTitle()}")

    for additionalRegion in additionalRegions:
        additionalRegion.Draw("SAME")
        leg.AddEntry(additionalRegion, f"Reco: {additionalRegion.GetTitle()}")

    if reco_error:
        reco_error.Draw("2p")
        hReco.Draw("l same")
    else:
        hReco.Draw("lpe same")

    if reco_error:
        leg.AddEntry(hReco, hReco.GetTitle(), "lp")
        leg.AddEntry(reco_error, reco_error.GetTitle(), "f")
    else:
        leg.AddEntry(hReco, hReco.GetTitle(), "lpe")

    leg.Draw("SAME")
    RootBackend.make_atlas_label()

    ratios = []
    dashed_lines = []
    for pad_idx, truth_idxs in enumerate(layout, 2):
        _draw_opts = 'E'
        canvas.cd(pad_idx)
        for truth_idx in truth_idxs:
            hTrue = hTruths[truth_idx]
            ratio = hTrue.Clone()
            ratio.Divide(hReco)
            ratio.SetTitle("")
            ratio.GetYaxis().SetTitle("Pred./Data")
            ratio.GetYaxis().SetRangeUser(0.85, 1.15)
            RootBackend.apply_font_styles(ratio.GetXaxis(), titleoffset=5)
            ratio.SetNdivisions(5, "Y")
            ratios.append(ratio)
            ratio.Draw(_draw_opts)
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
        dashed_lines.append(line)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with utils.all_redirected():
        canvas.SaveAs(str(output_path))


def plot_response(
    response,
    output_path,
    logz=True,
    add_blue=False,
    draw_opt="colz",
    normalise=True,
    zrange=(0.001, 1),
    text_label=None,
):
    hResponse = response.Clone()
    canvas = RootBackend.make_canvas(f"RooUnfoldResponse_{output_path}", mr=0.15)
    if logz:
        canvas.SetLogz()
    red = np.array([1.0, 1.0])
    blue = np.array([1.0, 0.0])
    green = np.array([1.0, 0.0])
    stops = np.array([0.0, 1.0])
    ROOT.TColor.CreateGradientColorTable(2, stops, red, green, blue, 255)
    # ROOT.TColor.InvertPalette()

    if not add_blue:
        red = np.array([1.0, 1.0, 1.0])
        blue = np.array([1.0, 0.5, 0.0])
        green = np.array([1.0, 0.5, 0.0])
        stops = np.array([0.0, 0.5, 1.0])
        ROOT.TColor.CreateGradientColorTable(3, stops, red, green, blue, 100)
    else:
        nlevels = 100
        red = np.concatenate(
            [np.linspace(0.0, 1.0, nlevels // 2), np.ones(nlevels // 2)]
        )
        blue = np.array(red[::-1])
        green = np.concatenate([red[: nlevels // 2], blue[nlevels // 2 :]])
        stops = np.linspace(0.0, 1.0, nlevels)

        minmax = np.min(np.abs([hResponse.GetMinimum(), hResponse.GetMaximum()]))
        import array

        levels = array.array('d', np.linspace(-minmax, minmax, nlevels))
        ROOT.TColor.CreateGradientColorTable(nlevels, stops, red, green, blue, nlevels)
        hResponse.SetContour(nlevels, levels)

    ROOT.gStyle.SetPaintTextFormat("4.2f")

    rsum = hResponse.Integral()
    for x in range(0, hResponse.GetNbinsX() + 1):
        if normalise:
            rsum = 0.0
            for y in range(0, hResponse.GetNbinsY() + 1):
                rsum += hResponse.GetBinContent(x, y)
            if rsum == 0.0:
                continue
        for y in range(0, hResponse.GetNbinsY() + 1):
            val = hResponse.GetBinContent(x, y) / rsum
            hResponse.SetBinContent(x, y, val)

    if zrange:
        hResponse.GetZaxis().SetRangeUser(*zrange)
    hResponse.GetYaxis().SetTitleOffset(1.5)
    hResponse.Draw(draw_opt)
    RootBackend.make_atlas_label(text_label)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with utils.all_redirected():
        canvas.SaveAs(str(output_path))


def plot_chi2(chi2, ndof, region, observable, output_path):
    chi2 = chi2.Clone()

    canvas = RootBackend.make_canvas(f"RooUnfold_chi2_{output_path}")
    RootBackend.apply_font_styles(chi2.GetXaxis())
    RootBackend.apply_font_styles(chi2.GetYaxis())

    chi2.GetXaxis().SetTitle("num. unfold iterations")
    chi2.GetYaxis().SetTitle(f"#chi^{{2}} / {ndof} DOF")
    chi2.SetTitle(f"Region={region}, Observable={observable}")
    chi2.SetMarkerStyle(8)
    chi2.SetMarkerSize(1)
    chi2.SetMarkerColor(2)

    chi2.Draw("P0")
    RootBackend.make_atlas_label()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with utils.all_redirected():
        canvas.SaveAs(str(output_path))


"""

  Make plots of the selection efficiency (using particle level quantities):

    eff = N_{reco,matched} / N_{truth}

  and purity (using reconstructed level quantities):

    purity = N_{reco,not matched} / N_{reco}

"""


def make_eff_plots(
    configMgr,
    skip_process=[],
    name_map={},
    phasespace="collinear",
    text_label=None,
    fmt="pdf",
):
    signals = [
        process
        for process in configMgr.processes
        if process.process_type in ["signal", "signal_alt"]
        and process.name not in skip_process
    ]

    regions = metadata.regions(configMgr)
    observables = metadata.observables(configMgr)
    print(observables)

    base_output_path = os.path.join(configMgr.out_path, "metrices")
    os.makedirs(base_output_path, exist_ok=True)

    for unfold_name, region in regions.items():

        if phasespace not in region["reco_match"]:
            continue

        for abcd_tag in abcd.get_abcd_tags(configMgr):
            fRegion = region["reco_match"]
            nRegion = region["reco_match_not"]
            rRegion = abcd.abcd_signal_region_name(configMgr, region["reco"], abcd_tag)

            if not rRegion:
                continue

            print(
                f"Calculating and plotting metrics for reco {rRegion}, not matched {nRegion} and matched {fRegion}"
            )

            for observable, histogram in observables.items():
                print(f"--> Analyzing observable {observable}")
                histos_eff = []
                histos_fakes = []
                # responses = []

                for s in signals:
                    print(f"----> Adding prediction from {s.name}")
                    hTrue = (
                        configMgr.get_process(s.name)
                        .get_region(region["particle"])
                        .get_histogram(histogram['truth'])
                        .root
                    )

                    hRecoMatch = (
                        configMgr.get_process(s.name)
                        .get_region(fRegion)
                        .get_histogram(f"response_matrix_{histogram['reco']}")
                        .project_y()
                        .root
                    )

                    hRecoMatch_tP = (
                        configMgr.get_process(s.name)
                        .get_region(fRegion)
                        .get_histogram(f"response_matrix_{histogram['reco']}")
                        .project_x()
                        .root
                    )

                    # hMatchFakes = (
                    #     configMgr.get_process(s.name)
                    #     .get_region(nRegion)
                    #     .get_histogram(histogram['reco'])
                    #     .root
                    # )

                    hReco = (
                        configMgr.get_process(s.name)
                        .get_region(rRegion)
                        .get_histogram(histogram['reco'])
                        .root
                    )

                    hResponse = (
                        configMgr.get_process(s.name)
                        .get_region(fRegion)
                        .get_histogram(histogram['response'])
                        .extend_bins()
                        .root
                    )

                    plot.plot_response(
                        hResponse,
                        pathlib.Path(configMgr.out_path)
                        .joinpath(f'metrices/{s.name}', f"{fRegion}_{observable}.{fmt}")
                        .resolve(),
                        text_label=text_label,
                    )

                    title = s.title
                    if name_map.get(s.name):
                        title = name_map.get(s.name)

                    selEff = hRecoMatch.Clone(f'selEff_{s.name}_{fRegion}_{observable}')
                    selEff.SetTitle(title)
                    selEff.SetLineColor(s.color)
                    selEff.SetMarkerColor(s.color)
                    selEff.SetMarkerSize(1.2)
                    selEff.SetMarkerStyle(s.markerstyle)
                    selEff.GetYaxis().SetRangeUser(0, 1.5)
                    selEff.GetYaxis().SetTitle('Selection efficiency')
                    selEff.Divide(hTrue)
                    selEff.SetLineColor(s.color)
                    histos_eff.append(selEff)

                    mFakes = hRecoMatch_tP.Clone(
                        f'fakes_{s.name}_{fRegion}_{observable}'
                    )
                    mFakes.SetTitle(title)
                    mFakes.SetLineColor(s.color)
                    mFakes.SetMarkerColor(s.color)
                    mFakes.SetMarkerSize(1.2)
                    mFakes.SetMarkerStyle(s.markerstyle)
                    mFakes.GetYaxis().SetRangeUser(0, 1.0)
                    mFakes.GetYaxis().SetTitle('Unmatched rate')
                    mFakes.Divide(hReco)
                    mFakes.SetLineColor(s.color)

                    for bin in range(0, mFakes.GetNbinsX() + 1):
                        mFakes.SetBinContent(bin, 1.0 - mFakes.GetBinContent(bin))

                    histos_fakes.append(mFakes)

                my_plotMaker = plotMaker.PlotMaker(configMgr, "unfolding/plots/")

                """
                Plot selection efficiency
                """
                cmd = 'e0'
                c = my_plotMaker.make_canvas()
                leg = my_plotMaker.make_legend(0.45, 0.70, 0.85, 0.85, 0.035)
                for hs in histos_eff:
                    leg.AddEntry(hs, hs.GetTitle(), "PL")
                    hs.SetTitle('')
                    hs.Draw(cmd)
                    cmd = 'e0 same'

                my_plotMaker.make_atlas_label()
                cap = ROOT.TF1('cap', '1', 1e-5, 1e5)
                cap.SetLineColor(ROOT.kBlack)
                cap.SetLineStyle(ROOT.kDashed)
                cap.Draw('same')
                leg.Draw()
                c.SaveAs(
                    f'{base_output_path}/SelectionEfficiency_{fRegion}_{histogram["truth"]}.{fmt}'
                )

                """
                Plot unmatched rate
                """
                cmd = 'e0'
                cf = my_plotMaker.make_canvas()
                legf = my_plotMaker.make_legend(0.45, 0.70, 0.85, 0.85, 0.035)
                for hs in histos_fakes:
                    legf.AddEntry(hs, hs.GetTitle(), "PL")
                    hs.SetTitle('')
                    hs.Draw(cmd)
                    cmd = 'e0 same'

                my_plotMaker.make_atlas_label()
                legf.Draw()
                cf.SaveAs(
                    f'{base_output_path}/MatchingFakes_{fRegion}_{histogram["truth"]}.{fmt}'
                )


def make_debugging_plots(configMgr, yrange=(1e-1, 1e8), output_folder="unfolding"):
    signals = [
        process
        for process in configMgr.processes
        if process.process_type in ["signal", "signal_alt"]
    ]
    bkgs = [process for process in configMgr.processes if process.process_type == "bkg"]
    fakes = [
        process for process in configMgr.processes if process.process_type == 'fakes'
    ]
    data = [
        process for process in configMgr.processes if process.process_type == 'data'
    ]

    my_plotMaker = plotMaker.PlotMaker(configMgr)

    for my_data in data:
        for region in my_data.regions:
            for observable in region.histograms:
                if isinstance(observable, core.Histogram2D):
                    continue

                legend = my_plotMaker.make_legend(text_size=None)

                get_hist = lambda process: process.get_region(
                    region.name
                ).get_histogram(observable.name)

                data_hists = {
                    process.name: get_hist(process).root.Clone() for process in data
                }
                for _, data_hist in data_hists.items():
                    my_plotMaker.data_style(data_hist)
                RootBackend.apply_styles(data_hists['data'], color=ROOT.kGray)

                data_plotJobs = [
                    plotMaker.PlotJob(data_name, data_hist.Clone(), "pE")
                    for data_name, data_hist in data_hists.items()
                ]

                signal_hists = {
                    process.name: get_hist(process).root.Clone() for process in signals
                }
                for process in signals:
                    RootBackend.apply_process_styles(
                        signal_hists[process.name], process
                    )

                signal_plotJobs = [
                    plotMaker.PlotJob(signal_name, signal_hist.Clone(), "HISTE")
                    for signal_name, signal_hist in signal_hists.items()
                ]

                for plotJob in data_plotJobs + signal_plotJobs:
                    my_plotMaker.update_legend(legend, "int", plotJob)

                bkg_obs_dict = {process.name: get_hist(process) for process in bkgs}
                try:
                    for process in fakes:
                        bkg_obs_dict[process.name] = get_hist(process)
                except:
                    # skip adding fakes in, if it does not exist for this region
                    pass

                stack_plotJob = my_plotMaker.stack(
                    bkg_obs_dict, f"bkg_{observable.name}_stack", legend, "int"
                )

                label_txt = region.name.split('_ABCD-')
                region_name = label_txt.pop(0)

                label_txt = None if len(label_txt) == 0 else label_txt[0].split('_')
                _texts = ["unfolding", *region_name.split('_')]
                text = [", ".join(_texts[i : i + 3]) for i in range(0, len(_texts), 3)]

                my_plotMaker.save_plot(
                    plot_job_list=[*data_plotJobs, *signal_plotJobs, stack_plotJob],
                    legend=legend,
                    odir=str(
                        pathlib.Path(f"{output_folder}/debug").joinpath(region.name)
                    ),
                    oname=observable.name,
                    logy=True,
                    format="pdf",
                    text=text,
                    label_txt=label_txt,
                    yrange=yrange,
                )


def plot_xsec_integral(
    configMgr,
    region_names,
    names,
    processes,
    output_path,
    ratio_base_idx=0,
    legend_order=None,
    legend_styles=None,
    yrange=None,
    yrange_ratio=None,
    nspace=1,
    logy=True,
    ratio_ndivisions=5,
    alpha_syst=0.4,
    legend_args=None,
    legend_add_stat_error=False,
    numerator_name=None,
    denominator_name=None,
    pull_data=None,
    **kwargs,
):
    """
    Plot cross section in each region for various processes.

    Parameters
    ----------
    configMgr : ConfigMgr
        Configuration manager object.
    region_names : list
        List of region names.
    names : list
        List of sample names.
    processes : list
        List of process names.
    output_path : str
        Output path for the plots.
    ratio_base_idx : int, optional
        Index of the ratio base. Defaults to 0.
    legend_order : list, optional
        List of index to order the legend. Defaults to None.
    legend_styles : list, optional
        List of legend styles. Defaults to None.
    yrange : tuple, optional
        Tuple of y-range. Defaults to None.
    yrange_ratio : tuple, optional
        Tuple of y-range for ratio. Defaults to None.
    nspace : int, optional
        Number of spaces. Defaults to 1.
    logy : bool, optional
        Whether to use log scale. Defaults to True.
    ratio_ndivisions : int, optional
        Number of divisions for ratio. Defaults to 5.
    alpha_syst : float, optional
        Alpha value for systematics. Defaults to 0.4.
    legend_args : dict, optional
        Dictionary of legend arguments. Defaults to None.
    legend_add_stat_error : bool, optional
        Whether to add statistical error to legend. Defaults to False.
    numerator_name : str, optional
        Name of the numerator. Defaults to None.
    denominator_name : str, optional
        Name of the denominator. Defaults to None.
    pull_data : dict, optional
        Dictionary of pull data. Defaults to None.
    **kwargs
        Additional keyword arguments.

    Notes
    -----
    The `kwargs` parameter is where all the data is specified.
    For example, the following is a valid input:

    >>> plot_xsec_integral(
    ...     configMgr,
    ...     region_names=['inclusive_reco_electron', ...],
    ...     names=['wjets_2211_EL', 'wjets_2211_MU', 'wjets_EL', ...],
    ...     processes=['wjets', ...],
    ...     output_path='path/to/output',
    ...     kwargs={'Inclusive':
    ...             {
    ...                 'wjets': (xsec, stat_low, stat_high, syst_low, syst_high, tot_low, tot_high, extra_low, extra_high),
                        'wjets_2211': (...)
                    },
            ...},
    ... )

    The `kwargs` parameter is a dictionary of dictionaries, where each key is a
    region name, and the value is another dictionary with the following keys:

    * `xsec`: the cross section value
    * `stat_low`: the statistical error on the cross section value (lower)
    * `stat_high`: the statistical error on the cross section value (upper)
    * `syst_low`: the systematic error on the cross section value (lower)
    * `syst_high`: the systematic error on the cross section value (upper)
    * `tot_low`: the total error on the cross section value (lower)
    * `tot_high`: the total error on the cross section value (upper)
    * `extra_low`: any additional error on the cross section value (lower)
    * `extra_high`: any additional error on the cross section value (upper)

    The `kwargs` parameter is required.
    """
    missing_region_names = set(region_names) - kwargs.keys()
    if missing_region_names:
        raise KeyError(f'Missing regions not provided: {missing_region_names}')

    # Here, we build up prep to start counting and figure out how to arrange the graphs
    nregions = len(region_names)
    nprocs = len(names)
    nbins = nregions
    legend_styles = legend_styles or nprocs * ["p"]

    legend_order = legend_order or list(range(nprocs))
    yrange = yrange or (1e2, 1e4)
    yrange_ratio = yrange_ratio or (-0.05, 2.05)

    unskewed = np.ones((nbins, nprocs))
    mask = np.ones_like(unskewed, dtype=bool)
    graphs_data = []
    for i in range(nprocs):
        mask[:, i] = 0
        unskewed_masked = np.copy(unskewed)
        unskewed_masked[mask] = 0
        mask[:, i] = 1
        graphs_data.append(
            np.pad(
                unskewed_masked, ((0, 0), (nspace, nspace)), mode='constant'
            ).flatten()
        )

    if pull_data is None:
        canvas = RootBackend.make_canvas(f"XSec_Integral_{output_path}", num_ratio=1)
    else:
        canvas = RootBackend.make_canvas(
            f"XSec_Integral_{output_path}",
            num_ratio=2,
            bottom_margin_fudge_factor=0.8,
            ratio_split_y=0.25,
        )
    canvas.SetTitle("")
    # set the grid style
    canvas.GetPad(1).SetGrid(1, 0)
    canvas.GetPad(2).SetGrid(1, 0)
    if pull_data:
        canvas.GetPad(3).SetGrid(1, 0)
    if logy:
        canvas.GetPad(1).SetLogy()
    legend_args = {} if legend_args is None else legend_args
    legend = RootBackend.make_legend(**legend_args)

    # top pad
    canvas.cd(1)
    hist = ROOT.TH1D(f"base_{output_path}", f"base_{output_path}", nbins, 0, nbins)

    hist.GetXaxis().SetNdivisions(nbins + 1)
    hist.GetXaxis().CenterLabels()
    hist.GetXaxis().SetTitle("region")
    hist.GetYaxis().SetTitle("#sigma_{fid} [fb]")
    RootBackend.apply_font_styles(hist.GetYaxis(), titleoffset=1.3)

    hist.SetAxisRange(*yrange, "Y")
    hist.Draw("0")

    graphs_stat = []
    graphs_syst = []
    graphs_extra = []
    for name, process, graph_data in zip(names, processes, graphs_data):
        graph_mask = np.where(graph_data != 0.0)

        edges = np.linspace(0, nbins, 1 + graph_data.size, dtype='float64')
        centers = 0.5 * (edges[1:] + edges[:-1])
        widths = np.diff(edges)

        # generate data for our graph for the process
        x = centers[graph_mask]
        statxl = np.zeros_like(x, dtype='float64')
        statxh = statxl
        systxl = widths / 4.0
        systxh = systxl

        y = []
        statyl = []
        statyh = []
        systyl = []
        systyh = []
        extray = []
        extrayl = []
        extrayh = []
        for region_name in region_names:
            data = kwargs[region_name][name]
            y.append(data[0])
            statyl.append(data[1])
            statyh.append(data[2])
            if len(data) > 3:
                systyl.append(data[3])
                systyh.append(data[4])
            else:
                systyl.append(0.0)
                systyh.append(0.0)
            if len(data) > 7:  # 5 and 6 are the upper/lower total
                extray.append(data[0])
                extrayl.append(data[7])
                extrayh.append(data[8])
            else:
                extray.append(0.0)

        y = np.array(y, dtype='float64')
        statyl = np.array(statyl, dtype='float64')
        statyh = np.array(statyh, dtype='float64')
        systyl = np.array(systyl, dtype='float64')
        systyh = np.array(systyh, dtype='float64')
        extray = np.array(extray, dtype='float64') if np.any(extray) else None
        extrayl = np.array(extrayl, dtype='float64') if extrayl else None
        extrayh = np.array(extrayh, dtype='float64') if extrayh else None

        # build and style the graph
        graph_syst = ROOT.TGraphAsymmErrors(
            x.size, x, y, systxl, systxh, systyl, systyh
        )
        RootBackend.apply_process_styles(
            graph_syst, process, alpha=alpha_syst, fillstyle=1001, fillcolor=True
        )
        graphs_syst.append(graph_syst)

        graph_stat = ROOT.TGraphAsymmErrors(
            x.size, x, y, statxl, statxh, statyl, statyh
        )
        RootBackend.apply_process_styles(graph_stat, process)
        graphs_stat.append(graph_stat)

        # plot
        graph_syst.Draw("2 same")
        graph_stat.Draw("p same")

        if extray is not None:
            graph_extra = ROOT.TGraphAsymmErrors(
                x.size, x, extray, statxl, statxh, extrayl, extrayh
            )
            RootBackend.apply_process_styles(graph_extra, process)
            graph_extra.SetLineColor(2)
            graph_extra.SetLineWidth(3)
            graph_extra.SetMarkerColor(2)
            graphs_extra.append(graph_extra)
            graph_extra.Draw("EZ same")
        else:
            graphs_extra.append(None)

    proxy_syst_error = graphs_syst[-1].Clone()

    for i, idx in enumerate(legend_order):
        if i == 1:
            legend.AddEntry(proxy_syst_error, "Stat. #oplus syst. unc.", "f")
            # legend.AddEntry(proxy_syst_error, "\\text{Stat.} \\oplus \\text{Sys. uncert.}", "f")
        legend.AddEntry(graphs_stat[idx], processes[idx].title, legend_styles[idx])

    legend.Draw("SAME")
    RootBackend.make_atlas_label()

    denominator = graphs_stat[ratio_base_idx]
    if denominator_name is None:
        denominator_name = processes[ratio_base_idx].title
    if numerator_name is None:
        numerator_name = "Pred."

    canvas.cd(2)
    # set up the second pad
    hist_ratio = hist.Clone()
    hist_ratio.SetTitle("")
    hist_ratio.GetYaxis().SetTitle(f"{numerator_name}/{denominator_name}")
    hist_ratio.GetYaxis().SetRangeUser(*yrange_ratio)
    RootBackend.apply_font_styles(hist_ratio.GetXaxis(), titleoffset=5, labeloffset=0.1)
    hist_ratio.SetNdivisions(ratio_ndivisions, "Y")

    for i, label in enumerate(region_names, 1):
        # second argument control the X-label rotation angle.
        hist_ratio.GetXaxis().ChangeLabel(i, -1, -1, -1, -1, -1, label)

    hist_ratio.Draw('0')

    def update_ratio_using_graph(ratio, graph, idx, denominator):
        x = graph.GetX()[idx]
        y = graph.GetY()[idx]
        exl = graph.GetEXlow()[idx]
        exh = graph.GetEXhigh()[idx]
        eyl = graph.GetEYlow()[idx]
        eyh = graph.GetEYhigh()[idx]

        try:
            ratio.SetPoint(idx, x, y / denominator.GetY()[idx])
            ratio.SetPointError(idx, exl, exh, eyl / y, eyh / y)
        except ZeroDivisionError:
            print(
                f'caught zero division error. will not set ratio or error for index={idx}, x={x}, y={y}, denominator={denominator.GetY()[idx]}'
            )

    def update_ratio_as_pull(ratio, graph, idx):
        x = graph.GetX()[idx]
        y = graph.GetY()[idx]
        ratio.SetPoint(idx, x, y)

    # draw all other ratios
    ratios = []
    for i, (graph_stat, graph_syst) in enumerate(zip(graphs_stat, graphs_syst)):
        ratio_syst = graph_syst.Clone()
        ratio_syst.SetTitle("")

        ratio_stat = graph_stat.Clone()
        ratio_stat.SetTitle("")

        graph_extra, ratio_extra = graphs_extra[i], None
        if graph_extra is not None:
            ratio_extra = graph_extra.Clone()
            ratio_extra.SetTitle("")

        for idx in range(nbins):
            update_ratio_using_graph(ratio_syst, graph_syst, idx, denominator)
            update_ratio_using_graph(ratio_stat, graph_stat, idx, denominator)
            if ratio_extra:
                update_ratio_using_graph(ratio_extra, graph_extra, idx, denominator)

        ratios.append(ratio_syst)
        ratios.append(ratio_stat)
        ratios.append(ratio_extra)

        ratio_syst.Draw('2 SAME')
        ratio_stat.Draw('P SAME')
        if ratio_extra:
            ratio_extra.Draw('EZ SAME')

    # add dashed line through midpoint 1.0
    # NB: requires canvas.Update() to get the right Uxmin/Uxmax values
    canvas.Update()
    line = ROOT.TLine(canvas.GetPad(1).GetUxmin(), 1, canvas.GetPad(1).GetUxmax(), 1)
    line.SetLineColor(ROOT.kBlack)
    line.SetLineStyle(ROOT.kDashed)
    line.Draw('SAME')

    proxy_stat_error = graphs_stat[-1].Clone()
    RootBackend.apply_styles(proxy_stat_error, color=1)
    if legend_add_stat_error:
        legend.AddEntry(proxy_stat_error, "statistical error", "e")

    RootBackend.apply_styles(
        proxy_syst_error, color=1, alpha=alpha_syst, fillcolor=True
    )

    if pull_data:
        canvas.cd(3)
        c_hist = hist_ratio.Clone()
        c_hist.Draw('0')
        c_hist.GetYaxis().SetTitle("Pull")

        pull_ratios = []
        for i, (graph_stat, pull) in enumerate(zip(graphs_stat, pull_data)):
            ratio_pull = graph_stat.Clone()
            ratio_pull.SetTitle("")
            for idx in range(nbins):
                update_ratio_as_pull(ratio_pull, pull, idx)
            pull_ratios.append(ratio_pull)
            ratio_pull.Draw('2 SAME')

        # line = ROOT.TLine(canvas.GetPad(2).GetUxmin(), 1, canvas.GetPad(2).GetUxmax(), 1)
        canvas.Update()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with utils.all_redirected():
        canvas.SaveAs(str(output_path))
