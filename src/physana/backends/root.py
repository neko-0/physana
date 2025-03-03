from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
import warnings
import logging

try:
    import ROOT

    ROOT.PyConfig.IgnoreCommandLineOptions = True
    ROOT.gROOT.SetBatch(True)
except ImportError:
    warnings.warn("Cannot import ROOT module!")

if TYPE_CHECKING:
    from ..core import Histogram, Histogram2D

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class RootBackend:
    ATLAS_LABELS_STATUS = "Internal"
    SHOW_LUMI_LABEL = True

    @staticmethod
    def apply_pad_margins(pad, mt=0, mr=0, mb=0, ml=0):
        pad.SetTopMargin(mt)
        pad.SetRightMargin(mr)
        pad.SetBottomMargin(mb)
        pad.SetLeftMargin(ml)

    @staticmethod
    def apply_pad_styles(pad, bordersize=0, fillcolor=0, xticks=True, yticks=True):
        pad.SetBorderSize(bordersize)
        pad.SetFillColor(fillcolor)
        if xticks:
            pad.SetTickx()
        if yticks:
            pad.SetTicky()

    @staticmethod
    def make_canvas(
        name,
        width=1000,
        height=800,
        num_ratio=0,
        ml=0.13,
        mr=0.05,
        mb=0.13,
        mt=0.10,
        ratio_pad_height=200,
        ratio_split_y=0.35,
        ratio_padding_y=0.00,
        # how big we typically want the axis spacing wrt to the original height
        bottom_margin_axis_pad=100,
        # fucking root
        bottom_margin_fudge_factor=0.9,
        **kwargs,
    ):
        ROOT.gStyle.SetOptStat(0)
        ROOT.gStyle.SetOptTitle(0)

        if num_ratio == 0:
            c = ROOT.TCanvas(f"canvas_{name}", f"canvas_{name}", width, height)
            c.SetPad(0, 0, 1, 1)
            RootBackend.apply_pad_margins(c, mt=mt, mr=mr, mb=mb, ml=ml)
            RootBackend.apply_pad_styles(c, **kwargs)
        else:
            # extend height as needed
            height += (num_ratio - 1) * ratio_pad_height
            c = ROOT.TCanvas(f"canvas_{name}", f"canvas_{name}", width, height)
            ratio_split_y = num_ratio / (height / ratio_pad_height + num_ratio - 1)
            ratio_axis_y = bottom_margin_axis_pad / height

            pad_top_below = ratio_split_y + ratio_padding_y / 0.5 + ratio_axis_y
            pad_bottom_above = ratio_split_y - ratio_padding_y / 0.5
            pad_bottom_above /= num_ratio

            c.Divide(1, num_ratio + 1)

            # set up spacing for main pad
            main_pad = c.GetPad(1)
            main_pad.SetPad(0, pad_top_below, 1, 1)
            RootBackend.apply_pad_margins(main_pad, mt=mt, mr=mr, mb=0.0, ml=ml)
            RootBackend.apply_pad_styles(main_pad, **kwargs)

            ratio_pad_indices = list(range(2, num_ratio + 2))
            # keep track of how much padding to give the pad from below
            pad_ylow = ratio_axis_y
            # build it from bottom up, easier
            for pad_idx in ratio_pad_indices:
                pad = c.GetPad(pad_idx)
                if pad_idx == 2:  # bottom pad
                    pad.SetPad(0, 0, 1, pad_ylow + pad_bottom_above)
                else:
                    pad.SetPad(0, pad_ylow, 1, pad_ylow + pad_bottom_above)
                pad_ylow += pad_bottom_above
                RootBackend.apply_pad_margins(pad, mt=0.0, mr=mr, mb=0.0, ml=ml)
                RootBackend.apply_pad_styles(pad, **kwargs)

            # figure out what fraction of the ratio pad will need to be padded for the axis
            bottom_ratio_pad = c.GetPad(ratio_pad_indices[0])
            bottom_ratio_pad.SetBottomMargin(
                bottom_margin_fudge_factor * bottom_margin_axis_pad / ratio_pad_height
            )

            top_ratio_pad = c.GetPad(ratio_pad_indices[-1])
            top_ratio_pad.SetTopMargin(0.05)

        c.Update()
        return c

    @staticmethod
    def apply_font_styles(
        axis,
        fontstyle=43,
        titlesize=32,
        labelsize=28,
        titleoffset=1.35,
        labeloffset=0.005,
    ):
        """
        Default fontstyle is 43, which corresponds to regular Arial + precision 3 (specifying fontsize is in pixels and not NDC).

        See TAttText: https://root.cern.ch/doc/master/classTAttText.html
        """
        axis.SetTitleFont(fontstyle)
        axis.SetTitleSize(titlesize)
        axis.SetLabelFont(fontstyle)
        axis.SetLabelSize(labelsize)
        axis.SetTitleOffset(titleoffset)
        axis.SetLabelOffset(labeloffset)

    @staticmethod
    def make_legend(x1=0.65, y1=0.60, x2=0.85, y2=0.85, text_size=0.03):
        legend = ROOT.TLegend(x1, y1, x2, y2)
        if text_size is not None:
            legend.SetTextSize(text_size)
        legend.SetBorderSize(0)
        legend.SetFillColor(ROOT.kWhite)
        return legend

    @staticmethod
    def make_text(text_content, x=0.5, y=0.6, color=1, font=43, size=30):
        if text_content is None:
            return
        text = ROOT.TLatex()
        text.SetTextColor(color)
        text.SetTextFont(font)
        text.SetTextSize(size)
        text.DrawLatexNDC(x, y, text_content)

    @staticmethod
    def make_math_text(text_content, x=0.5, y=0.6, color=1, font=43, size=30):
        text = ROOT.TMathText()
        text.SetTextColor(color)
        text.SetTextFont(font)
        text.SetTextSize(size)
        text.DrawMathText(x, y, text_content)

    @staticmethod
    def make_data_style(ihisto):
        if isinstance(ihisto, ROOT.TH1):
            ihisto.SetLineWidth(2)
            # ihisto.root.SetMarkerStyle(ROOT.kStar)
            ihisto.SetBinErrorOption(ROOT.TH1.kPoisson)
            # ROOT.gStyle.SetErrorX(0.0001)
            ihisto.SetMarkerStyle(8)
            ihisto.SetMarkerSize(1.2)
            ihisto.SetMarkerColor(ROOT.kBlack)
            ihisto.SetLineColor(ROOT.kBlack)
        else:
            ihisto.SetLineStyle(2)
            ihisto.SetMarkerStyle(ROOT.kStar)
            ihisto.SetMarkerColor(ROOT.kBlack)
            ihisto.SetLineColor(ROOT.kBlack)

    @staticmethod
    def make_atlas_label(
        texts=None, status=None, x=0.165, y=0.8, *, energy=13, lumi=140
    ):
        """
        Draw the ATLAS label, optional status, and luminosity label.

        Args:
            texts (list[str]): additional text lines
            status (str): e.g. "Preliminary"
            x (float): x position of the label
            y (float): y position of the label
            energy (int): 13 for 13 TeV
            lumi (int): 140 for 140 fb^{-1}
        """
        # Text coordinates
        y_gap = 0.04 * ROOT.gPad.GetCanvas().GetHNDC() / ROOT.gPad.GetHNDC()

        # ATLAS label
        RootBackend.make_text("ATLAS", x=x, y=y, font=73, size=30)

        # Status (e.g. "Preliminary")
        if status:
            RootBackend.make_text(status, x=x + 0.125, y=y, size=28)

        # Luminosity label
        if RootBackend.SHOW_LUMI_LABEL:
            lumi_label = f"#sqrt{{s}} = {energy} TeV, {lumi} fb^{{-1}}"
        else:
            lumi_label = f"#sqrt{{s}} = {energy} TeV"
        RootBackend.make_text(lumi_label, x=x, y=y - y_gap, size=28)

        # Additional text lines
        if texts is None:
            return
        for i, text in enumerate(texts):
            RootBackend.make_text(text, x=x, y=y - y_gap * (3 + i))

    @staticmethod
    def set_range(histo, *, xrange=None, yrange=None, top_room_scale=10.0):
        histo.Draw("goff")
        if yrange:
            histo.SetMaximum(yrange[1])
            histo.SetMinimum(yrange[0])
            histo.GetYaxis().SetLimits(yrange[0], yrange[1])
        else:
            max_y = histo.GetMaximum()
            min_y = histo.GetMinimum()
            if isinstance(histo, ROOT.TGraph):
                max_y = histo.GetHistogram().GetMaximum()
                min_y = histo.GetHistogram().GetMinimum()

            if min_y <= 0:
                min_y = 0.01

            new_max_y = max_y * top_room_scale
            new_min_y = min_y - 10.0 * min_y
            histo.SetMaximum(new_max_y)
            histo.SetMinimum(new_min_y)
            histo.GetYaxis().SetLimits(new_min_y, new_max_y)
        if xrange:
            histo.GetXaxis().SetLimits(xrange[0], xrange[1])
            histo.GetXaxis().SetRangeUser(xrange[0], xrange[1])

    @staticmethod
    def clip_graph_value(graph, clip_value):
        """Remove points where y == clip_value"""
        n_points = graph.GetN()
        for i in range(n_points - 1, -1, -1):
            if graph.GetY()[i] == clip_value:
                graph.RemovePoint(i)

    @staticmethod
    def update_legend(legend, legend_opt, plot_job):
        if plot_job.histogram:
            legend_tag = plot_job.tag
            legend_opt = legend_opt.lower().replace(" ", "").split(",")
            if 'int' in legend_opt:
                legend_tag += f", {int(plot_job.histogram.Integral())}"
            if "prediction" in legend_opt:
                legend_tag = f"prediction: {legend_tag}"
            elif "mc" in legend_opt:
                legend_tag = f"mc: {legend_tag}"
            else:
                pass
            legend_style = plot_job.legend_style or ""
            legend.AddEntry(plot_job.histogram, legend_tag, legend_style)

    @staticmethod
    def apply_styles(
        histogram: ROOT.TObject,
        color: int | str = None,
        alpha: float = None,
        title: str = None,
        linestyle: int = None,
        linewidth: int = None,
        markerstyle: int = None,
        markersize: int = None,
        binerror: int = None,
        fillstyle: int = None,
        fillcolor: bool = False,
    ) -> None:
        """
        Apply a set of style options to a histogram.

        Parameters
        ----------
        color : int | str
            Color of the histogram. If str, interpreted as a ROOT color name.
        alpha : float
            Transparency of the histogram (0-1).
        title : str
            Title of the histogram.
        linestyle : int
            Style of the line (0-10).
        linewidth : int
            Width of the line.
        markerstyle : int
            Style of the marker (1-30).
        markersize : int
            Size of the marker.
        binerror : int
            Bin error option.
        fillstyle : int
            Fill style (0-4000).
        fillcolor : bool
            If True, use the same color for filling as for the line.
        """

        # Apply generic style options
        if color is not None:
            ROOT.TColor.SetColorThreshold(0)
            if isinstance(color, str):
                color = ROOT.TColor.GetColor(color)
            alpha = alpha if alpha is not None else 1.0
            histogram.SetLineColorAlpha(color, alpha)
            histogram.SetMarkerColorAlpha(color, alpha)
            if fillcolor:
                histogram.SetFillColorAlpha(color, alpha)
        if title is not None:
            histogram.SetTitle(title)
        if linestyle is not None:
            histogram.SetLineStyle(linestyle)
        if linewidth is not None:
            histogram.SetLineWidth(linewidth)
        if markerstyle is not None:
            histogram.SetMarkerStyle(markerstyle)
        if markersize is not None:
            histogram.SetMarkerSize(markersize)
        if fillstyle is not None:
            histogram.SetFillStyle(fillstyle)

        # Apply TH1-specific style options
        if isinstance(histogram, ROOT.TH1):
            if binerror is not None:
                histogram.SetBinErrorOption(binerror)

    @staticmethod
    def apply_process_styles(histogram, process, **kwargs):
        process_kwargs = {
            k: getattr(process, k, None)
            for k in [
                'color',
                'alpha',
                'title',
                'linestyle',
                'linewidth',
                'markerstyle',
                'markersize',
                'binerror',
                'fillstyle',
                'fillcolor',
            ]
        }
        return RootBackend.apply_styles(histogram, **{**process_kwargs, **kwargs})

    @staticmethod
    def apply_systematic_style(ihisto, style="total", color=1, alpha=0.25):
        line_styles = {
            "total": 1,
            "stats": 2,
            "experimental": 1,
            "theory": 3,
            "others": 6,
            "unfold": 3,
        }
        style_number = line_styles.get(style, 1)
        ihisto.SetLineWidth(3)
        ihisto.SetLineStyle(style_number)
        ihisto.SetFillColorAlpha(color, alpha)
        if style == "total":
            ihisto.SetLineColor(1)
            ihisto.SetLineWidth(3)
            ihisto.SetFillStyle(0)
            ihisto.SetFillColorAlpha(0, 0)
        elif style == "stats":
            ihisto.SetLineColor(1)
            ihisto.SetFillStyle(0)
            ihisto.SetFillColorAlpha(0, 0)
        else:
            ihisto.SetLineColorAlpha(color, alpha)
        for i, b in enumerate(ihisto):
            ihisto.SetBinError(i, 0.0)

    @staticmethod
    def make_line(x1, y1, x2, y2):
        line = ROOT.TLine(x1, y1, x2, y2)
        line.SetLineColor(1)  # black
        line.SetLineWidth(3)
        line.SetLineStyle(2)  # dash
        return line

    @staticmethod
    def hist_1d_from_np(
        content, error=None, bins=None, name=None, xtitle=None, ytitle=None
    ):
        if name is None:
            name = str(content)
        if bins is None:
            bins = list(range(content.shape - 1))
        bins = np.array(bins, dtype="double")
        hist = ROOT.TH1D(name, name, len(bins) - 1, bins)
        if xtitle:
            hist.GetXaxis().SetTitle(xtitle)
        if ytitle:
            hist.GetYaxis().SetTitle(ytitle)
        for b in range(len(bins) + 1):
            hist.SetBinContent(b, content[b])
            if error:
                hist.SetBinError(b, error[b])
        return hist

    @staticmethod
    def hist_2d_from_np(
        content, error=None, bins=None, name=None, xtitle=None, ytitle=None
    ):
        if name is None:
            name = str(content)
        if bins is None:
            xbins = list(range(content.shape[0]))
            ybins = list(range(content.shape[1]))
        else:
            xbins, ybins = bins
        xbins = np.array(xbins, dtype="double")
        ybins = np.array(ybins, dtype="double")
        hist = ROOT.TH2D(name, name, len(xbins) - 1, xbins, len(ybins) - 1, ybins)
        if xtitle:
            hist.GetXaxis().SetTitle(xtitle)
        if ytitle:
            hist.GetYaxis().SetTitle(ytitle)
        for xb in range(len(xbins)):
            for yb in range(len(ybins)):
                hist.SetBinContent(xb, yb, content[xb, yb])
                if error:
                    hist.SetBinError(xb, yb, error[xb, yb])
        return hist


def to_root_graph(
    histo: Histogram,
    sys_type="Statistical",
    multiplier=None,
    position=None,
    include_syst=True,
    bin_labels=None,
    scale_multiplier_binwidth=0,
    fixed_multiplier_binwidth=None,
    **band_opts,
):
    logger.debug(f"making TGraphAsymmErrors for {histo.full_name}")

    _unique = histo.unique_name()

    # Data
    if multiplier:
        multiplier = multiplier + 1

    if position and position > (multiplier / 2.0):
        position = position + 1

    existing_obj = ROOT.gROOT.FindObject(_unique)
    if existing_obj:
        return existing_obj

    _gr = ROOT.TGraphAsymmErrors()
    _gr.SetName(_unique)
    _gr.SetMarkerStyle(8)
    _gr.SetLineWidth(2)

    width = np.diff(histo.bins) / 2.0
    bins = histo.bins[:-1] + width

    #  Systematic band
    syst_band = histo.total_band(**band_opts) if include_syst else None
    if syst_band is None:
        up = np.zeros(histo.bin_content.shape)
        down = np.zeros(histo.bin_content.shape)
    else:
        up = syst_band.up
        down = syst_band.down

    for b, (b_content, b_sumw2, b_down, b_up) in enumerate(
        zip(histo.bin_content[1:-1], histo.sumW2[1:-1], down[1:-1], up[1:-1])
    ):
        n = _gr.GetN()

        bin_point = bins[b]
        bin_width = width[b]
        if multiplier and position:
            if position > 0:
                bin_width = (width[b] * 2.0) / multiplier
                bin_point = (
                    (bins[b] - width[b]) + position * (bin_width) - (bin_width / 2.0)
                )
            if fixed_multiplier_binwidth is not None:
                bin_width = fixed_multiplier_binwidth
            else:
                bin_width *= scale_multiplier_binwidth  # bin_width / 2.0

        if 'Relative' in sys_type:
            _gr.SetPoint(n, bin_point, 1.0)
            if 'Statistical' in sys_type:
                _gr.SetPointError(
                    n,
                    bin_width,
                    bin_width,
                    np.sqrt(b_sumw2) / b_content if b_content else 0,
                    np.sqrt(b_sumw2) / b_content if b_content else 0,
                )
            elif 'Systematic' in sys_type:
                _gr.SetPointError(n, bin_width, bin_width, b_down, b_up)
        else:
            _gr.SetPoint(n, bin_point, b_content)
            if "Statistical" in sys_type:
                _gr.SetPointError(
                    n, bin_width, bin_width, np.sqrt(b_sumw2), np.sqrt(b_sumw2)
                )
            elif 'Systematic' in sys_type:
                _gr.SetPointError(
                    n, bin_width, bin_width, b_down * b_content, b_up * b_content
                )
            else:
                _gr.SetPointError(n, bin_width, bin_width, 0.0, 0.0)

    RootBackend.apply_font_styles(_gr.GetXaxis())
    RootBackend.apply_font_styles(_gr.GetYaxis())
    _gr.GetXaxis().SetTitle(histo.xtitle)
    _gr.GetYaxis().SetTitle(histo.ytitle)

    if bin_labels:
        xaxis = _gr.GetXaxis()
        # xaxis.SetNdivisions(_gr.GetN())
        for b in range(len(bins)):
            # nlabel = xaxis.FindBin(bins[b])
            # xaxis.SetBinLabel(nlabel, f"{bin_labels[b]}")
            # xaxis.SetBinLabel(b, f"{bin_labels[b]}")
            xaxis.ChangeLabel(b + 1, -1, -1, -1, -1, -1, f"{bin_labels[b]}")
        xaxis.LabelsOption("hc")
        xaxis.SetLabelOffset(0.02)
        xaxis.SetNdivisions(_gr.GetN() * 2)
        # xaxis.SetTickLength(1.0)
        # xaxis.SetTickSize(1.0)
        xaxis.CenterLabels()

    return _gr


def to_th1(histo: Histogram):
    logger.debug(f"making TH1 for {histo.full_name}")

    _unique = histo.unique_name()

    existing_obj = ROOT.gROOT.FindObject(_unique)
    if existing_obj:
        return existing_obj

    # make sure no nan or inf when converting
    histo.nan_to_num()

    if histo._bins is not None:
        bins_args = (len(histo._bins) - 1, histo._bins)
    else:
        bins_args = (histo.nbin, histo.xmin, histo.xmax)

    _hist = ROOT.TH1D(_unique, *histo.observable, *bins_args)
    # ROOT.gROOT.GetListOfSpecials().Add(_hist)
    # _hist.AddDirectory(0)
    # _hist.SetDirectory(0)
    _hist.SetTitle("")

    RootBackend.apply_font_styles(_hist.GetXaxis())
    RootBackend.apply_font_styles(_hist.GetYaxis())
    _hist.GetXaxis().SetTitle(histo.xtitle)
    _hist.GetYaxis().SetTitle(histo.ytitle)
    _hist.SetLineColor(ROOT.kBlack)
    for b, (b_content, b_sw2) in enumerate(zip(histo.bin_content, histo.sumW2)):
        logger.debug(f"{histo.bin_index[b]} {b} {b_content}")
        _hist.SetBinContent(b, b_content)
        _hist.SetBinError(b, np.sqrt(b_sw2))

    binerror = getattr(histo, 'binerror', None)
    if binerror:
        _hist.SetBinErrorOption(binerror)
    return _hist


def to_th1_error(histo: Histogram, *args, **kwargs):

    total_band = histo.total_band(*args, **kwargs)

    if total_band is None:
        return to_th1(histo)

    total_avg = (total_band.up + total_band.down) * 0.5 * histo.bin_content

    _unique = histo.unique_name() + "_error"

    existing_obj = ROOT.gROOT.FindObject(_unique)
    if existing_obj:
        return existing_obj

    # make sure no nan or inf when converting
    histo.nan_to_num()

    if histo._bins is not None:
        bins_args = (len(histo._bins) - 1, histo._bins)
    else:
        bins_args = (histo.nbin, histo.xmin, histo.xmax)
    _hist = ROOT.TH1D(_unique, *histo.observable, *bins_args)
    # ROOT.gROOT.GetListOfSpecials().Add(_hist)
    # _hist.AddDirectory(0)
    # _hist.SetDirectory(0)
    _hist.SetTitle("")

    RootBackend.apply_font_styles(_hist.GetXaxis())
    RootBackend.apply_font_styles(_hist.GetYaxis())
    _hist.GetXaxis().SetTitle(histo.xtitle)
    _hist.GetYaxis().SetTitle(histo.ytitle)
    _hist.SetLineColor(ROOT.kBlack)
    for b, (b_content, b_sw2) in enumerate(zip(histo.bin_content, total_avg)):
        logger.debug(f"{histo.bin_index[b]} {b} {b_content}")
        _hist.SetBinContent(b, b_content)
        _hist.SetBinError(b, np.sqrt(b_sw2))

    binerror = getattr(histo, 'binerror', None)
    if binerror:
        _hist.SetBinErrorOption(binerror)

    return _hist


def from_th1(histo: Histogram, roothist):
    logger.debug(f"loading TH1 into {histo.full_name}")

    nbins = roothist.GetNbinsX()
    assert nbins == len(histo.bins) - 1
    for i in range(0, nbins + 2):
        histo.bin_content[i] = roothist.GetBinContent(i)
        histo.sumW2[i] = roothist.GetBinError(i) ** 2
    histo.xtitle = roothist.GetXaxis().GetTitle()
    histo.ytitle = roothist.GetYaxis().GetTitle()
    # set styles and store in the histogram metadata
    style = {
        "color": roothist.GetLineColor(),
        "linestyle": roothist.GetLineStyle(),
        "linewidth": roothist.GetLineWidth(),
        "markerstyle": roothist.GetMarkerStyle(),
        "markersize": roothist.GetMarkerSize(),
        "binerror": 0,
        "fillstyle": roothist.GetFillStyle(),
        "alpha": None,
    }
    histo.metadata["style"] = style


def to_th2(histo: Histogram2D):
    logger.debug(f"making TH2 for {histo.full_name}")

    _unique = histo.unique_name()

    existing_obj = ROOT.gROOT.FindObject(_unique)
    if existing_obj:
        return existing_obj

    # make sure no nan or inf when converting
    histo.nan_to_num()

    if histo._bins is not None:
        bins = histo.bins
        bins_args = (len(bins[0]) - 1, bins[0], len(bins[1]) - 1, bins[1])

    else:
        bins_args = (
            histo.xbin,
            histo.xmin,
            histo.xmax,
            histo.ybin,
            histo.ymin,
            histo.ymax,
        )
    _hist = ROOT.TH2D(_unique, histo.name, *bins_args)
    # _hist.AddDirectory(0)
    # _hist.SetDirectory(0)
    _hist.GetXaxis().SetTitle(histo.xtitle)
    _hist.GetYaxis().SetTitle(histo.ytitle)
    RootBackend.apply_font_styles(_hist.GetXaxis())
    RootBackend.apply_font_styles(_hist.GetYaxis())
    RootBackend.apply_font_styles(_hist.GetZaxis())
    for coord in np.ndindex(histo.bin_content.shape):
        _hist.SetBinContent(*coord, histo.bin_content[coord])
        _hist.SetBinError(*coord, np.sqrt(histo.sumW2[coord]))
    return _hist


def from_th2(histo, roothist):
    logger.debug(f"loading TH2 into {histo.full_name}")

    nbinsx = roothist.GetNbinsX()
    nbinsy = roothist.GetNbinsY()
    assert nbinsx == len(histo.bins[0]) - 1
    assert nbinsy == len(histo.bins[1]) - 1
    for coord in np.ndindex(histo.bin_content.shape):
        histo.bin_content[coord] = roothist.GetBinContent(*coord)
        histo.sumW2[coord] = roothist.GetBinError(*coord) ** 2
    histo.xtitle = roothist.GetXaxis().GetTitle()
    histo.ytitle = roothist.GetYaxis().GetTitle()
    # set styles and store in the histogram metadata
    style = {
        "color": roothist.GetLineColor(),
        "linestyle": roothist.GetLineStyle(),
        "linewidth": roothist.GetLineWidth(),
        "markerstyle": roothist.GetMarkerStyle(),
        "markersize": roothist.GetMarkerSize(),
        "binerror": roothist.GetBinErrorOption(),
        "fillstyle": roothist.GetFillStyle(),
        "alpha": None,
    }
    histo.metadata["style"] = style
