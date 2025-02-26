'''
class for making plots
'''

import os
import numpy as np
import pathlib
import copy
import logging

from .core import Histogram, Histogram2D
from .backends import RootBackend

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


try:
    import ROOT

    ROOT.PyConfig.IgnoreCommandLineOptions = True
    ROOT.gROOT.SetBatch(True)
    ROOT.gStyle.SetOptStat(0)
except ImportError:
    pass

COLOR_HEX = [
    "#41d106",  # light green
    "#ffcc00",  # yellow
    "#e93524",  # red
    "#1a73e8",  # blue
    "#F39C12",  # orange
    "#9B59B6",  # purple
    "#717D7E",  # grey
    "#7fe4b4",  # light green-ish
    "#8c510a",
    "#1d91c0",
    "#c7e9b4",
]

STACK_PLOT_HEX = [
    # "#FFFFFF",
    "#d73027",
    "#fdae61",
    "#4575b4",
    "#8073ac",
    "#5aae61",
    "#de77ae",
    "#bf812d",
    "#35978f",
]


class PlotJob:
    def __init__(self, tag, histo, draw_opt, name=None, divide_binwidth=False):
        self.tag = tag
        self._histo = histo
        self.draw_opt = draw_opt
        self.name = name  # observable name
        self._divide_binwidth = False
        self.legend_style = None
        if divide_binwidth:
            self.divide_binwidth()

    @property
    def histogram(self):
        if isinstance(self._histo, (Histogram, Histogram2D)):
            return self._histo.root
        else:
            return self._histo

    def divide_binwidth(self):
        if self._divide_binwidth:
            return
        if isinstance(self._histo, (Histogram, Histogram2D)):
            self._histo.bin_content /= self._histo.bin_width
            self._histo.sumW2 /= self._histo.bin_width**2
            self._histo.ytitle += " / bin"
        else:
            for b in range(0, self._histo.GetNbinsX() + 1):
                bwidth = self._histo.GetBinWidth(b)
                self._histo.SetBinContent(b, self._histo.GetBinContent(b) / bwidth)
                self._histo.SetBinError(b, self._histo.GetBinError(b) / bwidth)
            ytitle = self._histo.GetYaxis().GetTitle()
            self._histo.GetYaxis().SetTitle(f"{ytitle} / bin")
        self._divide_binwidth = True


class PlotMaker(object):
    c_counter = 0

    legend_nevent = True

    glob_default_params = {"legend_nevent": True}

    PLOT_STATUS = "Internal"

    HIGHT_LIGHT_SIGNAL_COLOR = {"W+jets": 2}

    def __init__(self, configMgr=None, output_dir='./', backend="root"):
        logger.debug("Calling PlotMaker init")
        self.configMgr = configMgr
        self.backend = backend
        self.counter = 0
        self.top_room_scale = 3000.0

        self.stack = self.make_stack()
        self.save_plot = self.make_save_plot()
        self.ratio_plot = self.make_ratio_plot()
        self.multiple_ratio = self.make_multiple_ratio()

        self._output_dir = ""
        if configMgr is not None:
            self.output_dir = f"{configMgr.out_path}/{output_dir}"
        else:
            self.output_dir = f"./{output_dir}"

    @staticmethod
    def default_glob_param():
        for key, value in PlotMaker.glob_default_params.items():
            setattr(PlotMaker, key, value)

    @property
    def output_dir(self):
        return self._output_dir

    @output_dir.setter
    def output_dir(self, new_dir):
        self._output_dir = new_dir
        try:
            os.makedirs(self._output_dir)
        except FileExistsError:
            pass

    def make_stack(self):
        if self.backend == "root":
            return self._root_stack
        else:
            return None

    def make_save_plot(self):
        if self.backend == "root":
            return self._root_save_plot

    def make_ratio_plot(self):
        if self.backend.lower() == "root":
            return self._root_ratio_plot

    def make_multiple_ratio(self):
        if self.backend.lower() == "root":
            return self._root_multiple_ratio_plot

    def make_canvas(self, *args, **kwargs):
        if self.backend.lower() == "root":
            PlotMaker.c_counter += 1
            c_counter = PlotMaker.c_counter
            return RootBackend.make_canvas(str(c_counter), *args, **kwargs)
        else:
            logger.warning(f"{self.backend} dose not have make_canvas")
            return None

    def make_legend(self, *args, **kwargs):
        # x1=0.65, y1=0.7, x2=0.85, y2=0.9, text_size=0.03
        if self.backend.lower() == "root":
            return RootBackend.make_legend(*args, **kwargs)
        else:
            NotImplementedError(f"Not implemented for {self.backend=}")

    def make_text(self, *args, mathtext=False, **kwargs):
        if self.backend.lower() == "root":
            if mathtext:
                return RootBackend.make_math_text(*args, **kwargs)
            else:
                return RootBackend.make_text(*args, **kwargs)
        else:
            raise NotImplementedError(f"Not implemented for {self.backend=}")

    def _splitline(self, l1, l2):
        return f"#splitline{{{l1}}}{{{l2}}}"

    def data_style(self, ihisto):
        if self.backend.lower() == "root":
            return RootBackend.make_data_style(ihisto)

    def make_atlas_label(self, *args, **kwargs):
        if self.backend.lower() == "root":
            kwargs.setdefault("status", PlotMaker.PLOT_STATUS)
            return RootBackend.make_atlas_label(*args, **kwargs)
        else:
            raise NotImplementedError(f"Not implemented for {self.backend=}")

    def set_range(self, histo, *, xrange=None, yrange=None):
        if self.backend.lower() == "root":
            if isinstance(histo, PlotJob):
                histo = histo.histogram
            return RootBackend.set_range(histo, xrange=xrange, yrange=yrange)

    def update_legend(self, legend, legend_opt, plot_job):
        if self.backend.lower() == "root":
            return RootBackend.update_legend(legend, legend_opt, plot_job)
        else:
            logger.critical(f"{self.backend} dose not have update_legend")
            return None

    def draw_single(self, pFilter='', rFilter='ROOT'):
        for p in self.configMgr.processes:
            if pFilter and pFilter not in p.name:
                continue
            for r in p.regions:
                if rFilter and rFilter not in r.name:
                    continue
                for h in r.histograms:
                    c = self.make_canvas()
                    h.style()
                    h.root.Draw("COLZ" if isinstance(h.root, ROOT.TH2) else "")
                    c.SaveAs(
                        f"{self.output_dir}/Eff_{p.name}_{r.name}_{h.observable}.pdf"
                    )

    def draw_cmp_process(self, normalize=False, pFilter=''):
        writeDir = f"{self.output_dir}/cmp_process/"
        if not os.path.exists(writeDir):
            os.makedirs(writeDir)

        for r in self.configMgr.regions:
            for h in r.histograms:
                # Now make one histogram adding all processes
                # onto the same canvas
                canvas = self.make_canvas()
                leg = self.make_legend()
                sname = f"{r.name}_{h.name}"
                dcmd = 'hist'
                hists = []
                maxBin = 0

                for p in self.configMgr.processes:
                    # User filter of processes
                    if pFilter and pFilter not in p.name:
                        continue

                    hist = self.configMgr.get_histogram(p.name, r.name, h.name).root
                    hist.SetName(sname)
                    hist.SetLineColor(p.color)
                    hist.SetLineWidth(2)

                    if normalize:
                        hist.Scale(1.0 / hist.Integral())

                    hist.Draw(dcmd)
                    # hist.GetYaxis().SetRangeUser(0,0.25)
                    hists.append(hist)
                    if hist.GetMaximum() > maxBin:
                        maxBin = hist.GetMaximum()

                    leg.AddEntry(hist, p.name, "PL")
                    sname += f"_{p.name}"
                    dcmd = 'SAME'

                if len(hists) == 0:
                    continue

                hists[0].GetYaxis().SetRangeUser(0, 1.3 * maxBin)
                leg.Draw('same')
                canvas.Update()
                canvas.SaveAs(f"{writeDir}/{sname}.pdf")

    # ==========================================================================
    def compare_2regions(
        self,
        region1,
        region2,
        observable,
        *,
        y_lowtitle="",
        oname=None,
        leg1=None,
        leg2=None,
        norm=True,
        text=None,
        yrange=None,
        xrange=None,
    ):
        """
        Compare 2 regions. assuming using ROOT backend.

        Args:
            region1 (obj:Region) : region 1.

            region2 (obj:Region) : region 2.

            observable (str) : name of the observable to compare.

        Optional (named) Args:

            leg1 (str) : legend name for region 1.

            leg2 (str) : legend name for region 2.

            norm (bool) : normalization.
        """
        if not region1.has_observable(observable):
            logger.warning(f"{region1.name} has no obs: {observable}")
        if not region2.has_observable(observable):
            logger.warning(f"{region2.name} has no obs: {observable}")

        if norm:
            obs1 = region1.get_histogram(observable).normalize.root
            obs2 = region2.get_histogram(observable).normalize.root
            obs1.GetYaxis().SetTitle("Normalized # Events")
        else:
            obs1 = region1.get_histogram(observable).root
            obs2 = region2.get_histogram(observable).root
        obs1.SetLineColor(2)
        obs1.SetLineWidth(2)
        obs2.SetLineColor(4)
        obs2.SetLineWidth(2)

        if xrange:
            if isinstance(xrange, dict):
                if observable in xrange:
                    xmin = xrange[observable][0]
                    xmax = xrange[observable][1]
                    obs1.GetXaxis().SetRangeUser(xmin, xmax)
                    obs2.GetXaxis().SetRangeUser(xmin, xmax)
            elif isinstance(xrange, tuple):
                xmin = xrange[0]
                xmax = xrange[1]
                obs1.GetXaxis().SetRangeUser(xmin, xmax)
                obs2.GetXaxis().SetRangeUser(xmin, xmax)

        leg = self.make_legend()
        leg.SetTextFont(42)
        leg.SetTextSize(0.02)
        leg.AddEntry(obs1, leg1 if leg1 else region1.name)
        leg.AddEntry(obs2, leg2 if leg2 else region2.name)

        obs1_job = PlotJob("", obs1, "HE")
        obs2_job = PlotJob("", obs2, "HE")

        ofilename = f"{observable.replace('/','_over_')}"

        ratio = self.ratio_plot(
            obs1_job,
            obs2_job,
            y_lowtitle,
            logy=False,
            yrange=yrange,
            low_yrange=(0.7, 1.3),
        )
        self.save_plot(
            [ratio], leg, oname, ofilename, yrange=None, add_text=[text], logy=False
        )

    # ==========================================================================
    # methods that use ROOT module.
    # ==========================================================================
    def _root_color(self, value):
        root_color_dict = {
            2: ROOT.kRed,
            3: ROOT.kGreen + 1,
            4: ROOT.kBlue,
            5: ROOT.kYellow + 1,
            6: ROOT.kOrange + 1,
            7: ROOT.kViolet,
            8: ROOT.kGray,
            9: ROOT.kGreen + 5,
            10: ROOT.kBlue + 5,
            11: ROOT.kYellow + 5,
        }
        return root_color_dict.get(value, value)

    def root_set_color(self, histogram, color):
        RootBackend.apply_styles(histogram, color=self._root_color(color))

    def _root_stack(
        self,
        histo_dict,
        tag,
        legend,
        legend_opt="",
        color_dict=None,
        hide_process=False,
        reverse_order=False,
        do_sort=True,
        divide_binwidth=True,
    ):
        """
        histo_dict contains {name tag : histogram} pair

        e.g. histo_dict = {tag1:hiso1, tag2:histo2 } etc

        color_dict will use the same tag.

        hide_process : bool, hiding processes that go into the stack plots.
        """

        if color_dict is None:
            color_dict = {
                i: ROOT.TColor.GetColor(x) for i, x in enumerate(STACK_PLOT_HEX)
            }

        ROOT.gROOT.SetBatch(True)

        # legend = self.make_legend()
        legend_opt = legend_opt.replace(" ", "").split(",")

        reorder_buffer = {}

        xrange = None
        nbin = 100
        xtitle = ""
        ytitle = ""
        for color, (histo_tag, histo) in enumerate(histo_dict.items()):
            if histo is None:
                logger.critical(f"{histo_tag} is None")
                continue
            xtitle = xtitle or histo.xtitle
            ytitle = ytitle or histo.ytitle

            histo.nan_to_num()
            if isinstance(histo, Histogram2D):
                ytitle = "Number of events"
                my_histo = histo.root.ProjectionX(histo.xtitle, 0, -1, "e")
                if divide_binwidth:
                    logger.warning("2D histogram doesn't do binwidth division.")
            else:
                if divide_binwidth:
                    histo.bin_content /= histo.bin_width
                    histo.sumW2 /= histo.bin_width**2
                    histo.ytitle += " / bin"
                my_histo = histo.root
                xrange = (histo.bins[0], histo.bins[-1])
                nbin = len(histo.bins) - 1

            my_color = histo.parent.parent.color  # or color_dict.get(
            #    color, list(color_dict)[-1]
            # )

            my_histo.Draw("goff")
            my_histo.SetLineWidth(0)
            my_histo.SetLineColor(my_color)
            my_histo.SetMarkerColor(my_color)
            my_histo.SetFillColorAlpha(my_color, 0.42)
            if histo_tag in PlotMaker.HIGHT_LIGHT_SIGNAL_COLOR:
                my_histo.SetLineColor(PlotMaker.HIGHT_LIGHT_SIGNAL_COLOR[histo_tag])
                my_histo.SetLineWidth(2)
                my_histo.SetMarkerColor(0)
                my_histo.SetFillColorAlpha(my_color, 0.42)

            if xrange:
                my_histo.GetXaxis().SetLimits(*xrange)
                my_histo.GetXaxis().SetRangeUser(*xrange)

            histo_legend_tag = histo_tag
            if "int" in legend_opt and type(self).legend_nevent:
                histo_legend_tag += f", {my_histo.Integral():.0f}"
            if "prediction" in legend_opt:
                histo_legend_tag = f"prediction: {histo_legend_tag}"
            if not hide_process:
                legend.AddEntry(my_histo, histo_legend_tag, "F")

            integral_checker = my_histo.Integral()
            while integral_checker in reorder_buffer:
                integral_checker -= 0.01
            reorder_buffer[integral_checker] = my_histo

        stack_histo = ROOT.THStack()
        if xrange:
            _unique_name = f"hbin_{nbin}_{xrange}"
            existing_obj = ROOT.gROOT.FindObject(_unique_name)
            if existing_obj:
                hbin = existing_obj
            else:
                hbin = ROOT.TH1D(_unique_name, "hbin", nbin, *xrange)
                hbin.Draw("goff")
                hbin.GetXaxis().SetLimits(*xrange)
                hbin.GetXaxis().SetRangeUser(*xrange)
            stack_histo.SetHistogram(hbin.Clone())
            stack_histo.Draw("goff")
            stack_histo.GetXaxis().SetLimits(*xrange)
            stack_histo.GetXaxis().SetRangeUser(*xrange)

        sorted_area = reorder_buffer.keys()
        if do_sort:
            sorted_area = sorted(sorted_area)
        if reverse_order:
            sorted_area.reverse()
        for area in sorted_area:
            stack_histo.Add(reorder_buffer[area])

        logger.info("finalizing stack")
        # stack_histo.SetHistogram(stack_histo.GetStack().First().Clone()) # for axis
        if hide_process:
            stack_histo = stack_histo.GetStack().Last().Clone()
            legend.AddEntry(stack_histo, f"sum of MCs, {stack_histo.Integral():.0f}")
        stack_histo.Draw("goff")
        if xrange:
            stack_histo.GetXaxis().SetLimits(*xrange)
            stack_histo.GetXaxis().SetRangeUser(*xrange)
        stack_histo.GetXaxis().SetTitle(xtitle)
        stack_histo.GetYaxis().SetTitle(ytitle)

        ROOT.gROOT.SetBatch(False)

        obsname = next(iter(histo_dict.values())).name

        return PlotJob(tag, stack_histo, "HIST", obsname)

    def _root_save_plot(
        self,
        plot_job_list,
        odir,
        oname,
        *,
        legend=None,
        logy=True,
        xrange=None,
        yrange=None,
        figfmt="png,pdf",
        text=None,
        show_text=False,
        label_txt=None,
        batch=True,
        is_ratio=False,
        syst_band_histogram=None,
        show_total_syts_only=True,  # show total syst. band only.
        divide_binwidth=True,
        ytitles=None,
    ):
        ROOT.gROOT.SetBatch(batch)
        _verbose = ROOT.gErrorIgnoreLevel
        ROOT.gErrorIgnoreLevel = ROOT.kFatal

        logger.info("preparing canvas.")
        canvas = self.make_canvas()
        canvas.cd()
        if logy:
            canvas.SetLogy()

        if not isinstance(plot_job_list, list):
            plot_job_list = [plot_job_list]

        for entry, plot_job in enumerate(plot_job_list):
            histo = plot_job.histogram
            # histo.GetXaxis().SetRange(-1, 0) # why do I need this?
            if entry == 0:
                histo.Draw(f"{plot_job.draw_opt}")
                # histo.GetXaxis().SetNdivisions(310)
                if xrange:
                    histo.GetXaxis().SetRange(-1, 0)
                    histo.GetXaxis().SetRangeUser(*xrange)
                if yrange:
                    histo.GetYaxis().SetRangeUser(*yrange)
                if ytitles and plot_job.name in ytitles:
                    new_ytitle = ytitles[plot_job.name]
                    if hasattr(histo, "GetUpperRefObject"):
                        yaxis = histo.GetUpperRefObject().GetYaxis()
                    else:
                        yaxis = histo.GetYaxis()
                    yaxis.SetTitle(new_ytitle)
            else:
                histo.GetXaxis().SetRange(-1, 0)
                histo.Draw(f"{plot_job.draw_opt} same")

        canvas.Update()

        if legend:
            try:
                legend.Draw()
            except AttributeError:
                pass

        logger.info("finalizing canvas.")

        if show_text and text:
            if not isinstance(text, list):
                text = [text]
            text_x, text_y = 0.20, 0.75  # 0.73
            for t in text:
                # if len(t) > 36:
                #    t = self._splitline("".join(t[:28]), "".join(t[28:]))
                if isinstance(t, str):
                    self.make_text(t, x=text_x, y=text_y, color=1, size=30)
                    text_y -= 0.04
                elif isinstance(t, dict):
                    self.make_text(**t)
                else:
                    logger.warning(f"cannot process type {type(t)}")

        if is_ratio:
            label_pos = {"x": 0.185, "y": 0.87}
        else:
            label_pos = {"x": 0.185, "y": 0.82}

        RootBackend.make_atlas_label(label_txt, **label_pos)

        ratio_buffers = []
        buffers = []
        syst_leg_name = "Stat. #oplus syst. unc."  # Total unc.
        if syst_band_histogram and syst_band_histogram.systematic_band:
            # systematic_band is Histogram object with syst. band
            total_band = None
            syst_bands = list(syst_band_histogram.systematic_band.keys())
            syst_bands += [syst_leg_name]
            bin_width = ex_l = ex_h = np.diff(syst_band_histogram.bins) / 2.0
            x_values = syst_band_histogram.bins[:-1] + bin_width
            nominal_content = syst_band_histogram.bin_content[1:-1]
            # constructing systematic band (shaded error band)
            for bname in syst_bands:
                if bname == syst_leg_name:
                    band = total_band
                else:
                    band = syst_band_histogram.systematic_band[bname]
                    if total_band is None:
                        total_band = copy.deepcopy(band)
                    else:
                        total_band.combine(band)

                # skip through all other systematic band and keep the total.
                if show_total_syts_only and bname != syst_leg_name:
                    continue

                # scaled_band = band.scale_nominal(nominal_content)
                scaled_band = {
                    "up": band.up,
                    "down": band.down,
                }
                if is_ratio:
                    y_values = np.ones(x_values.shape, dtype="float")
                    ey_h = np.nan_to_num(band["up"][1:-1] / nominal_content)
                    ey_l = np.nan_to_num(band["down"][1:-1] / nominal_content)
                    data_pts = [
                        x.astype("float")
                        for x in [x_values, y_values, ex_l, ex_h, ey_l, ey_h]
                    ]
                    ratio_band_g = ROOT.TGraphAsymmErrors(len(x_values), *data_pts)
                    ratio_buffers.append(ratio_band_g)
                y_values = nominal_content
                ey_h = scaled_band["up"][1:-1]
                ey_l = scaled_band["down"][1:-1]
                if divide_binwidth:
                    y_values /= np.diff(syst_band_histogram.bins)
                    ey_h /= np.diff(syst_band_histogram.bins)
                    ey_l /= np.diff(syst_band_histogram.bins)
                data_pts = [
                    x.astype("float")
                    for x in [x_values, y_values, ex_l, ex_h, ey_l, ey_h]
                ]
                band_graph = ROOT.TGraphAsymmErrors(len(x_values), *data_pts)
                band_graph.SetLineColor(2)
                band_graph.SetLineWidth(2)
                buffers.append(band_graph)
                if legend:
                    band_graph.SetLineWidth(0)
                    legend.AddEntry(band_graph, f"{bname}", "F")

            # drawing the bands on the top and bottom canvas.
            colors = [922, 800, 880, 600, 861, 820, 411]
            fill_styles = [1001, 1001, 1001, 1001, 1001, 1001, 1001]
            _histo_xmin = histo.GetXaxis().GetXmin()
            _histo_xmax = histo.GetXaxis().GetXmax()
            if hasattr(plot_job_list[0].histogram, "GetLowerPad"):
                plot_job_list[0].histogram.GetLowerPad().cd()
                for i, _graph in enumerate(reversed(ratio_buffers)):
                    _graph.SetLineColor(colors[i])
                    _graph.SetFillColor(colors[i])
                    _graph.SetFillColorAlpha(colors[i], 0.35)
                    _graph.SetFillStyle(fill_styles[i])
                    _graph.GetXaxis().SetRange(-1, 0)
                    _graph.GetXaxis().SetLimits(_histo_xmin, _histo_xmax)
                    _graph.GetXaxis().SetRangeUser(_histo_xmin, _histo_xmax)
                    _graph.Draw("2 same")
            if hasattr(plot_job_list[0].histogram, "GetUpperPad"):
                plot_job_list[0].histogram.GetUpperPad().cd()
                for i, _graph in enumerate(reversed(buffers)):
                    _graph.SetLineColor(colors[i])
                    _graph.SetFillColor(colors[i])
                    _graph.SetFillColorAlpha(colors[i], 0.35)
                    _graph.SetFillStyle(fill_styles[i])
                    _graph.GetXaxis().SetRange(-1, 0)
                    _graph.GetXaxis().SetLimits(_histo_xmin, _histo_xmax)
                    _graph.GetXaxis().SetRangeUser(_histo_xmin, _histo_xmax)
                    _graph.Draw("2 same")

        canvas.Update()
        figfmt = figfmt.replace(" ", "")
        for pic_format in figfmt.split(","):
            save_path = pathlib.Path(self.output_dir).joinpath(
                odir, f"{oname}.{pic_format}"
            )
            save_path.parent.mkdir(parents=True, exist_ok=True)
            canvas.SaveAs(str(save_path.resolve()))
        ROOT.gROOT.SetBatch(False)
        ROOT.gErrorIgnoreLevel = _verbose

    def _root_save_canvas(
        self,
        canvas,
        legend,
        odir,
        oname,
        *,
        logy=False,
        figfmt="png,pdf",
        text=None,
        show_text=False,
        label_txt=None,
    ):
        ROOT.gROOT.SetBatch(True)
        _verbose = ROOT.gErrorIgnoreLevel
        ROOT.gErrorIgnoreLevel = ROOT.kFatal

        if logy:
            canvas.SetLogy()

        canvas.cd()

        try:
            legend.Draw()
        except Exception:
            pass

        if show_text:
            if not isinstance(text, list):
                text = [text]
            text_x, text_y = 0.52, 0.57
            if text:
                for t in text:
                    if len(t) > 36:
                        t = self._splitline("".join(t[:28]), "".join(t[28:]))
                    self.make_text(t, text_x, text_y, 1)
                    text_y -= 0.04

        self.make_atlas_label(label_txt)

        figfmt = figfmt.replace(" ", "")
        for pic_format in figfmt.split(","):
            save_path = pathlib.Path(self.output_dir).joinpath(
                odir, f"{oname}.{pic_format}"
            )
            save_path.parent.mkdir(parents=True, exist_ok=True)
            canvas.SaveAs(str(save_path.resolve()))

        ROOT.gROOT.SetBatch(False)
        ROOT.gErrorIgnoreLevel = _verbose

    def _root_ratio_plot(
        self,
        plot_job1,
        plot_job2,
        low_ytitle,
        *,
        xrange=None,  # tupele(xmin,xmax)
        yrange=None,  # tupele(ymin,ymax)
        logy=False,
        low_yrange=None,
        div_opt="divsym",
        invert_ratio=False,
        width=None,
    ):
        ROOT.gROOT.SetBatch(True)

        h1 = plot_job1.histogram  # .GetStack().Last().Clone()
        h2 = plot_job2.histogram
        if xrange:
            h1.Draw("goff")
            h2.Draw("goff")
            h1.GetXaxis().SetRange(-1, 0)
            h2.GetXaxis().SetRange(-1, 0)
            h1.GetXaxis().SetLimits(*xrange)
            h1.GetXaxis().SetRangeUser(*xrange)
            h2.GetXaxis().SetLimits(*xrange)
            h2.GetXaxis().SetRangeUser(*xrange)
        ratio = ROOT.TRatioPlot(h1, h2, div_opt)
        ratio.SetH1DrawOpt(plot_job1.draw_opt)
        ratio.SetH2DrawOpt(plot_job2.draw_opt)
        # ratio.SetGraphDrawOpt("h")
        ratio.Draw("goff")

        # rr = plot_job2.histogram.Clone()
        # _rr = plot_job1.histogram.GetStack().Last().Clone()
        # rr.Divide(_rr)
        # asym = ROOT.TGraphAsymmErrors()
        # asym.Divide(_rr, rr, "pois")

        logger.info("modifying axis")
        if xrange:
            logger.debug(f"setting x range to {xrange}")
            ratio.GetXaxis().SetRange(-1, 0)
            ratio.GetXaxis().SetLimits(*xrange)
            ratio.GetXaxis().SetRangeUser(*xrange)
            # ratio.GetUpperRefObject().GetXaxis().SetLimits(*xrange)
            # ratio.GetUpperRefObject().GetXaxis().SetRangeUser(*xrange)
            # ratio.GetLowerRefGraph().GetXaxis().SetLimits(*xrange)
            # ratio.GetLowerRefGraph().GetXaxis().SetRangeUser(*xrange)
        else:
            pass
            # low_xmin = ratio.GetLowerRefGraph().GetXaxis().GetXmin()
            # low_xmax = ratio.GetLowerRefGraph().GetXaxis().GetXmax()
            # ratio.GetXaxis().SetLimits(low_xmin, low_xmax)
            # ratio.GetXaxis().SetRangeUser(low_xmin, low_xmax)
            # ratio.GetUpperRefObject().GetXaxis().SetLimits(low_xmin, low_xmax)
            # ratio.GetLowerRefGraph().GetXaxis().SetLimits(low_xmin, low_xmax)
            # ratio.GetUpperRefObject().GetXaxis().SetRangeUser(low_xmin, low_xmax)
            # ratio.GetLowerRefGraph().GetXaxis().SetRangeUser(low_xmin, low_xmax)
        if yrange:
            logger.debug(f"setting y range to {yrange}")
            ratio.GetUpperRefObject().SetMaximum(yrange[1])
            ratio.GetUpperRefObject().SetMinimum(yrange[0])
            ratio.GetUpperRefObject().GetYaxis().SetLimits(yrange[0], yrange[1])
            ratio.GetUpperRefObject().GetYaxis().SetRangeUser(yrange[0], yrange[1])
        else:
            max_y = ratio.GetUpperRefObject().GetMaximum()
            if max_y == 0:
                new_max_y = 1.0
            else:
                new_max_y = max_y * self.top_room_scale
            ratio.GetUpperRefObject().SetMaximum(new_max_y)
            ratio.GetUpperRefObject().SetMinimum(1.5)
            ratio.GetUpperRefObject().GetYaxis().SetLimits(0.1, new_max_y)

        if logy:
            logger.debug("setting y to log scale")
            ratio.GetUpperPad().SetLogy()

        logger.debug("Adjusting ratio plot.")
        raw_xvalue = ratio.GetLowerRefGraph().GetX()
        raw_yvalue = ratio.GetLowerRefGraph().GetY()
        if "divsym" in div_opt:
            raw_xvalue_he = ratio.GetLowerRefGraph().GetEX()
            raw_xvalue_le = ratio.GetLowerRefGraph().GetEX()
            raw_yvalue_he = ratio.GetLowerRefGraph().GetEY()
            raw_yvalue_le = ratio.GetLowerRefGraph().GetEY()
        else:
            raw_xvalue_he = ratio.GetLowerRefGraph().GetEXhigh()
            raw_xvalue_le = ratio.GetLowerRefGraph().GetEXlow()
            raw_yvalue_he = ratio.GetLowerRefGraph().GetEYhigh()
            raw_yvalue_le = ratio.GetLowerRefGraph().GetEYlow()
        ratio.GetLowerRefGraph().SetLineColor(ROOT.kBlack)

        if raw_yvalue:
            if invert_ratio:
                yvalue = [0 if value == 0 else 1.0 / value for value in raw_yvalue]
                yvalue_he = [
                    0 if y == 0 else e / (y**2)
                    for y, e in zip(raw_yvalue, raw_yvalue_he)
                ]
                yvalue_le = [
                    0 if y == 0 else e / (y**2)
                    for y, e in zip(raw_yvalue, raw_yvalue_le)
                ]
            else:
                yvalue = raw_yvalue
                yvalue_he = raw_yvalue_he
                yvalue_le = raw_yvalue_le
        else:
            yvalue = [0]
            yvalue_he = [0]
            yvalue_le = [0]
        for point, (x, xle, xhe, y, yle, yhe) in enumerate(
            zip(
                raw_xvalue,
                raw_xvalue_le,
                raw_xvalue_he,
                yvalue,
                yvalue_le,
                yvalue_he,
            )
        ):
            # print(f"check {y} {rr.GetBinContent(point)}")
            ratio.GetLowerRefGraph().SetPoint(point, x, y)
            if width is None:
                width = h2.GetBinWidth(point + 1) * 0.5
            if width == 0:
                ratio.GetLowerRefGraph().SetMarkerStyle(8)
                ratio.GetLowerRefGraph().SetMarkerSize(1.0)
                ratio.GetLowerRefGraph().SetMarkerColor(1)
            if "divsym" in div_opt:
                ratio.GetLowerRefGraph().SetPointError(point, width, yle)
            else:
                ratio.GetLowerRefGraph().SetPointError(point, xle, xhe, yle, yhe)

        ymean = np.average(yvalue)
        ysigma = np.std(yvalue)
        logger.info(f"mean: {ymean}, sigma: {ysigma}")

        ratio.GetLowerRefYaxis().SetRangeUser(
            ymean - 2.0 * ysigma, ymean + 2.0 * ysigma
        )

        ratio.GetLowerRefYaxis().SetTitle(low_ytitle)
        ratio.GetLowerRefGraph().SetLineWidth(2)
        ratio.SetLowBottomMargin(0.40)
        ratio.SetLeftMargin(0.15)
        ratio.GetLowYaxis().SetNdivisions(10)
        ratio.GetLowerRefXaxis().SetLabelSize(0.04)
        ratio.GetLowerRefXaxis().SetTitleSize(0.04)
        ratio.GetLowerRefXaxis().SetTitleOffset(1.2)
        ratio.GetLowerRefYaxis().SetLabelSize(0.03)
        ratio.GetLowerRefYaxis().SetTitleSize(0.04)
        ratio.GetUpperRefYaxis().SetLabelSize(0.05)
        ratio.GetUpperRefYaxis().SetTitleSize(0.05)

        if low_yrange:
            ratio.GetLowerRefYaxis().SetRangeUser(*low_yrange)
            ratio.GetLowerRefGraph().SetMaximum(low_yrange[1])
            ratio.GetLowerRefGraph().SetMinimum(low_yrange[0])

        ROOT.gROOT.SetBatch(False)

        return PlotJob(
            f"{plot_job1.tag}_over_{plot_job2.tag}", ratio, "", plot_job1.name
        )

    def _root_multiple_ratio_plot(
        self,
        sub_dir,
        obs,
        base_plot_job,
        comp_plot_job,  # list(plotJob)
        low_ytitle,
        *,
        xrange=None,  # tupele(xmin,xmax)
        yrange=None,  # tupele(ymin,ymax)
        logy=False,
        low_yrange=None,
        ndiv=10,
        legend=None,
        figfmt="png",
        text=None,
        show_text=False,
        label_txt=None,
    ):
        ROOT.gROOT.SetBatch(True)

        ratio_canvas = self.make_canvas()
        ratio_canvas.cd()
        ratio_list = []
        base = base_plot_job.histogram.Clone()
        for comp in comp_plot_job:
            _ratio = ROOT.TRatioPlot(comp.histogram, base, "divsym")
            _ratio.SetH1DrawOpt(base_plot_job.draw_opt)
            _ratio.SetH2DrawOpt(comp.draw_opt)
            _ratio.Draw("goff")
            ye = _ratio.GetLowerRefGraph().GetEY()
            for i, _ye in enumerate(ye):
                _xe = base.GetBinWidth(i + 1) / 2
                # if obs == "lep1Pt_varbin_e":
                #   print(f"{i} {_xe} {_ye}")
                #    print(f"{_ratio.GetLowerRefGraph().GetPointX(i)}")
                _ratio.GetLowerRefGraph().SetPointError(i, _xe, _ye)
            # if obs == "lep1Pt_varbin_e":
            #    print(base)
            _ratio.GetLowerRefGraph().SetLineColor(comp.histogram.GetLineColor())
            ratio_list.append(_ratio)

        canvas = self.make_canvas()
        canvas.cd()
        ratio = ratio_list[0]
        ratio.Draw()
        logger.info("modifying axis")
        if xrange:
            logger.debug(f"setting x range to {xrange}")
            ratio.GetXaxis().SetLimits(xrange[0], xrange[1])
            ratio.GetXaxis().SetRangeUser(xrange[0], xrange[1])
        if yrange:
            logger.debug(f"setting y range to {yrange}")
            ratio.GetUpperRefObject().SetMaximum(yrange[1])
            ratio.GetUpperRefObject().SetMinimum(yrange[0])
            ratio.GetUpperRefObject().GetYaxis().SetLimits(*yrange)
            ratio.GetUpperRefObject().GetYaxis().SetRangeUser(*yrange)
        else:
            max_y = ratio.GetUpperRefObject().GetMaximum()
            if max_y == 0:
                new_max_y = 1.0
            else:
                new_max_y = max_y * self.top_room_scale
            ratio.GetUpperRefObject().SetMinimum(0.1)
            ratio.GetUpperRefObject().SetMaximum(new_max_y)
            ratio.GetUpperRefObject().GetYaxis().SetLimits(0.1, new_max_y)
            ratio.GetUpperRefObject().GetYaxis().SetRangeUser(0.1, new_max_y)
        if low_yrange:
            ratio.GetLowerRefYaxis().SetRangeUser(*low_yrange)
            ratio.GetLowerRefGraph().SetMaximum(low_yrange[1])
            ratio.GetLowerRefGraph().SetMinimum(low_yrange[0])

        ratio.GetLowerRefYaxis().SetTitle(low_ytitle)
        ratio.SetLowBottomMargin(0.40)
        ratio.SetLeftMargin(0.15)
        ratio.GetLowYaxis().SetNdivisions(ndiv)
        # lower axis
        ratio.GetLowerRefXaxis().SetLabelSize(0.04)
        ratio.GetLowerRefXaxis().SetTitleSize(0.04)
        ratio.GetLowerRefYaxis().SetLabelSize(0.03)
        ratio.GetLowerRefYaxis().SetTitleSize(0.05)
        ratio.SetGridlines([1])
        # upper axis
        # ratio.GetUpperRefYaxis().SetLabelSize(0.04)
        # ratio.GetUpperRefYaxis().SetTitleSize(0.05)

        ratio.GetUpperPad().Update()
        if logy:
            logger.debug("setting y to log scale")
            ratio.GetUpperPad().SetLogy()

        ratio.GetLowerPad().cd()
        for _ratio in ratio_list[1:]:
            _ratio.GetLowerRefGraph().Draw("p same")
            ratio.GetLowerPad().Update()

        ratio.GetUpperPad().cd()
        for _comp in comp_plot_job[1:]:
            _comp.histogram.Draw("h same")
            ratio.GetUpperPad().Update()

        self._root_save_canvas(
            canvas,
            legend,
            sub_dir,
            obs.replace("/", "_over_").replace(" ", ""),
            # logy=logy,
            figfmt=figfmt,
            text=text,
            show_text=show_text,
            label_txt=label_txt,
        )

        ROOT.gROOT.SetBatch(False)

        # return (canvas,ratio)#(PlotJob(obs, ratio, ""), ratio_list)

    def plot_corrections(self, configMgr, filter=None):
        canvas = RootBackend.make_canvas(name='corrections')
        leg = RootBackend.make_legend()

        dcmd = ''

        hists = []

        savetag = ''

        for corr in configMgr.corrections:
            if filter and corr not in filter:
                continue

            if corr[0] not in savetag:
                savetag += corr[0]

            if corr[2] not in savetag:
                savetag += f'_{corr[2]}'

            savetag += f'_{corr[1]}'

            process = configMgr.get_process(corr[1])
            hist = configMgr.corrections[corr].root
            hist.GetYaxis().SetTitle("Correction Factor")
            hist.GetYaxis().SetRangeUser(0.0, 1.6)
            hist.SetLineColor(process.color)
            hist.SetLineWidth(2)
            hists.append(hist)
            hist.Draw(dcmd)
            dcmd = 'same'

            leg.AddEntry(hist, corr[1], "LP")

        leg.Draw("same")
        RootBackend.make_atlas_label()
        canvas.SaveAs(f"{self.output_dir}/correction_factors_{savetag}.pdf")

    # --------------------------------------------------------------------------
    def plot_pset_systematics(
        self,
        process_set,
        output_dir="systematic_plots",
        low_yrange=(0.5, 1.5),
        rfilter=None,
        hfilter=None,
        computed_only=False,
        syst_fullname=False,
        syst_list=None,
    ):
        """
        plot systematics for given ProceeSet instance.
        Mainly use for debugging and visual inspection for a given process set.
        """
        sys_style = RootBackend.apply_systematic_style

        sys_names = [] if computed_only else process_set.list_systematic_names()
        computed_sys_names = process_set.list_computed_systematics()

        nominal = process_set.get(None)
        region_list = nominal.list_regions()
        obs_list = [obs.name for obs in nominal[0] if isinstance(obs, Histogram)]
        if syst_list is None:
            syst_list = list(sys_names) + list(computed_sys_names)
        for i, name in enumerate(syst_list):
            if i < len(sys_names):
                plist = process_set.get(name)
            else:
                plist = process_set.get_computed_systematic(name)
                name = "_".join(name)
            if plist == process_set.nominal:
                logger.warning(f"No plist for {name}")
                continue
            for r in region_list:
                for obs in obs_list:
                    canvas = RootBackend.make_canvas(name=f"{name}/{r}/{obs}")
                    canvas.SetLogy()
                    canvas.cd()

                    try:
                        nominal_hist = nominal.get_region(r).get_observable(obs)
                        nominal_hist = nominal_hist.root
                    except KeyError:
                        continue

                    self.data_style(nominal_hist)

                    histo_buffer = [
                        p.get_region(r).get_observable(obs).root for p in plist
                    ]

                    ndiv = 306
                    ratio = ROOT.TRatioPlot(nominal_hist, nominal_hist, "divsym")
                    ratio.SetH1DrawOpt("hist")
                    ratio.SetH2DrawOpt("hist")
                    ratio.Draw()
                    ratio.GetLowYaxis().SetNdivisions(ndiv)
                    ratio.GetLowerRefYaxis().SetLabelSize(0.04)
                    ratio.GetLowerRefYaxis().SetTitleSize(0.04)
                    ratio.GetLowerRefYaxis().SetLabelSize(0.03)
                    ratio.GetLowerRefYaxis().SetTitleSize(0.05)
                    ratio.GetLowerRefYaxis().SetRangeUser(*low_yrange)
                    ratio.SetGridlines([1])
                    ratios = [ratio]
                    for i, h in enumerate(histo_buffer, start=2):
                        ratio = h / nominal_hist
                        sys_style(h)
                        sys_style(ratio)
                        if len(plist) < 10:
                            h.SetLineColor(i)
                            ratio.SetLineColor(i)
                        ratios[0].GetUpperPad().cd()
                        h.Draw("hist SAME")
                        ratios[0].GetLowerPad().cd()
                        ratio.Draw("SAME E")
                        ratios.append(ratio)
                        del ratio
                    # ratios[0].GetLowerPad().cd()
                    # ratios[0].GetLowerRefGraph().Draw("SAME")

                    n_plist = len(plist)
                    canvas.cd()
                    legend = RootBackend.make_legend(x1=0.40, x2=0.60)
                    legend.AddEntry(nominal_hist, f"{process_set.name} nominal", "p")
                    if n_plist < 10:
                        for i, h in enumerate(histo_buffer):
                            if syst_fullname:
                                m_name = plist[i].systematic.full_name[1:]
                            else:
                                m_name = name
                            legend.AddEntry(h, f"{m_name}({i+1})/{n_plist}")
                    legend.Draw()

                    output = pathlib.Path(
                        f"{self.output_dir}/{output_dir}/{process_set.name}/{name}/{r}/{obs}.png"
                    )
                    output.parent.mkdir(parents=True, exist_ok=True)
                    canvas.SaveAs(f"{output.resolve()}")

    def plot_process_set(
        self,
        process_set,
        output_dir="",
        low_yrange=(0.5, 1.5),
        region_filter=None,
        systematic_list=[],
    ):
        if not output_dir:
            output_dir = f"{self.output_dir}/process_set/"

        sys_style = RootBackend.apply_systematic_style

        if not systematic_list:
            systematic_list = list(process_set.list_systematic_full_name())
            systematic_list += list(process_set.list_computed_systematics())

        nominal = process_set.get()
        regions = nominal.list_regions(region_filter)
        for region_name in regions:
            region = nominal.get_region(region_name)
            for obs in region.histograms:
                tot_buffer = {}
                for systematic in systematic_list:
                    if (syst_process := process_set.get(systematic)) is nominal:
                        continue
                    if not isinstance(obs, Histogram):
                        continue
                    syst_region = syst_process.get_region(region.name)

                    canvas = RootBackend.make_canvas(
                        name=f"{region.full_name}/{obs}/{systematic}"
                    )
                    canvas.SetLogy()
                    canvas.cd()
                    legend = RootBackend.make_legend(x1=0.45, x2=0.65)

                    # converting to ROOT histogram
                    nominal_rh = obs.root
                    syst_rh = syst_region.get_observable(obs.name).root

                    sys_style(nominal_rh)
                    sys_style(syst_rh, "component")
                    syst_rh.SetLineColor(2)

                    ndiv = 306
                    ratio = ROOT.TRatioPlot(syst_rh, nominal_rh, "divsym")
                    ratio.SetH1DrawOpt("HIST")
                    ratio.SetH2DrawOpt("HIST")
                    ratio.Draw()
                    ratio.GetLowYaxis().SetNdivisions(ndiv)
                    ratio.GetLowerRefYaxis().SetLabelSize(0.04)
                    ratio.GetLowerRefYaxis().SetTitleSize(0.04)
                    ratio.GetLowerRefYaxis().SetLabelSize(0.03)
                    ratio.GetLowerRefYaxis().SetTitleSize(0.05)
                    ratio.GetLowerRefYaxis().SetRangeUser(*low_yrange)
                    ratio.SetGridlines([1])
                    ratio.Draw("HIST")

                    legend.AddEntry(nominal_rh, f"nominal-{nominal.name}")
                    legend.AddEntry(syst_rh, f"{systematic}")
                    legend.Draw()

                    syst_name = "_".join(systematic).replace("/", "_")
                    fdir = f"{output_dir}/{nominal.name}/{region.name}/{syst_name}/"
                    output = pathlib.Path(f"{fdir}/{obs.name}.png")
                    output.parent.mkdir(parents=True, exist_ok=True)
                    canvas.SaveAs(f"{output.resolve()}")

                    tot_buffer[systematic[-1]] = syst_rh

                tot_canvas = RootBackend.make_canvas(
                    name=f"{region.full_name}/{obs}/tot"
                )
                tot_canvas.SetLogy()
                tot_canvas.cd()
                tot_legend = RootBackend.make_legend(x1=0.25, x2=0.45)
                nominal_rh = obs.root
                nominal_rh.Draw("HIST")
                color = 2
                for name, syst_rh in tot_buffer.items():
                    syst_rh.SetLineColor(color)
                    syst_rh.Draw("HIST SAME")
                    tot_legend.AddEntry(syst_rh, f"{name}")
                    color += 1
                tot_legend.Draw()
                fdir = f"{output_dir}/{nominal.name}/{region.name}/all/"
                output = pathlib.Path(f"{fdir}/{obs.name}.png")
                output.parent.mkdir(parents=True, exist_ok=True)
                tot_canvas.SaveAs(f"{output.resolve()}")

    def plot_relative_systematics(
        self,
        process,
        output_dir="relative_systematic_plots",
        yrange=(0.001, 10),
        show_components=True,
        show_details=False,
        region_filter=None,
        logy=False,
        combine_band=None,
        exclude_band=None,
        include_band=None,
        include_stats=True,
        legend_pos=None,
        display_total=True,
        symmetrize=True,
        type_map=None,
        fig_fmt="pdf",
        exp_leg=None,
        theory_leg=None,
    ):
        """
        Plotting relative systematic band for a given process object.
        The band infomation can be retrived via the nominal histogram.

        Args:
            process: core.Process
                nominal process that already filled with systematic band.

            output_dir : str, optional
                output directory for plots.

            yrange : tuple(float, float), or dict{"region/histogram" : (float, float)}
                range of the y-axis.
                If dict is passed, "default" is used as fallback.
                example dict :
                {
                    "electron_inclusive_truth/mjjTruht" : (-0.5, 0.5),
                    "default" : (-1.5, 1.5),
                }

            show_components : bool, default=True
                plotting components of the total band.

            combine_band : dict(list), default=None
                grouping for combining bands. combined bands will be removed
                during the plotting. e.g for combining 'ttbar' and 'zjets', 'jet'
                and 'lepton' bands:
                {
                    "bkgd-theory":['ttbar', 'zjets'],
                    "exp":['jet', 'lepton'],
                }
        """
        if combine_band is None:
            combine_band = {}
        if exclude_band is None:
            exclude_band = []
        if include_band is None:
            include_band = []

        if not output_dir:
            output_dir = f"{self.output_dir}/relative_systematic/"

        sys_style = RootBackend.apply_systematic_style

        # use for harmonizing common naming
        common_name = {
            "stats": "Statistical",
            "lumi": "Luminosity",
            "total": "Stat. #oplus syst. unc.",
        }

        # getting region names
        regions = process.list_regions(region_filter)

        for rname in regions:
            r = process.get_region(rname)
            for obs in r.histograms:
                # checking systematic band
                # note 2D histogram has no band variables
                try:
                    bands = copy.deepcopy(obs.systematic_band)
                    if bands is None:
                        continue
                except AttributeError:
                    continue

                # filtering by exclude
                for bname in list(bands.keys()):
                    if exclude_band and bname in exclude_band:
                        bands.pop(bname, None)

                # combining several bands into one.
                # NOTE: combined bands will be removed during plotting
                for grp_name, grp in combine_band.items():
                    if len(grp) < 1:
                        logger.warning(f"Cannot combine size {len(grp)} {grp_name}")
                        continue
                    temp_bands = [bands.pop(x, None) for x in grp]
                    temp_bands = [x for x in temp_bands if x is not None]
                    if not temp_bands:
                        continue
                    first_band = copy.deepcopy(temp_bands[0])
                    first_band.name = grp_name  # rename the band
                    for _band in temp_bands[1:]:  # combine the rest
                        first_band.combine(_band)
                    bands.update({grp_name: first_band})  # update bands dict

                # update the grouping if total band already exist
                if "total" not in bands:
                    total_band = obs.total_band(
                        include_stats=include_stats,
                        exclude_names=exclude_band,
                        ext_band=bands,
                    )
                    bands[total_band.name] = total_band
                else:
                    for grp_name, grp in combine_band.items():
                        for gname in grp:
                            try:
                                bands["total"].remove_sub_band(gname)
                            except KeyError:
                                logger.warning(f"total has no sub-band {gname} !!!")
                        bands["total"].update_sub_bands(bands[grp_name])

                logger.info(f"found systematic band {bands.keys()}")

                # plotting each band
                for bname, band in bands.items():
                    # plot only bands in the 'include'
                    if include_band and bname not in include_band:
                        continue
                    total = {"up": None, "down": None}
                    total_components = {"up": {}, "down": {}}
                    # plot up and down band
                    for stype in ["up", "down"]:
                        side_band = getattr(band, stype)
                        canvas_name = f"{r}/{bname}/{stype}/{obs}"
                        canvas = RootBackend.make_canvas(name=canvas_name)
                        canvas.cd()
                        if logy:
                            canvas.SetLogy()
                        legend = RootBackend.make_legend(x1=0.40, x2=0.60, y1=0.5)

                        # make copy of the nominal histogram object,
                        # replace the bin content with band
                        c_obs = obs.copy(shallow=True)
                        c_obs.name = f"{bname}_{stype}"
                        c_obs.ytitle = "Relative uncertainty"
                        c_obs.bin_content = side_band

                        rhisto = c_obs.root
                        sys_style(rhisto)

                        if isinstance(yrange, dict):
                            _range = yrange.get(
                                f"{r.name}/{obs.name}",
                                yrange.get("default", (0.001, 10)),
                            )
                        elif yrange is None:
                            _range = (0.001, 10)
                        else:
                            _range = yrange
                        rhisto.GetYaxis().SetRangeUser(*_range)

                        legend.AddEntry(rhisto, f"{bname}-{stype}")
                        total[stype] = rhisto

                        draw_opt = "HIST"
                        rhisto.Draw(draw_opt)

                        if show_components:
                            color_index = 0
                            draw_opt = "HIST same"
                            comp_legend = {
                                "experimental": RootBackend.make_legend(
                                    x1=0.40, x2=0.60, y1=0.4
                                ),
                                "theory": RootBackend.make_legend(
                                    x1=0.60, x2=0.80, y1=0.4
                                ),
                            }
                            for sub_name, sub_band in band.sub_bands.items():
                                try:
                                    color = ROOT.TColor.GetColor(COLOR_HEX[color_index])
                                except IndexError:
                                    color = ROOT.TColor.GetColor(COLOR_HEX[-1])
                                m_side_band = getattr(sub_band, stype)
                                # exclude zero contribution components
                                if m_side_band.sum() <= 0.0:  # this is numpy array
                                    continue
                                c_obs = obs.copy(shallow=True)
                                c_obs.name = f"{sub_name}_{stype}"
                                c_obs.ytitle = "Relative uncertainty"
                                c_obs.bin_content = m_side_band
                                c_obs.nan_to_num()
                                comp_rhisto = c_obs.root
                                sys_style(comp_rhisto, sub_band.type, color, 0.8)
                                comp_legend.get(sub_band.type, legend).AddEntry(
                                    comp_rhisto, c_obs.name
                                )
                                comp_rhisto.Draw(draw_opt)
                                color_index += 1
                                total_components[stype][sub_name] = (
                                    comp_rhisto,
                                    sub_band.type,
                                )
                                if sub_name == "stats":  # stats will not have color
                                    color_index -= 1
                            for leg in comp_legend.values():
                                leg.Draw()

                        legend.Draw()
                        fdir = f"{output_dir}/{process.name}/{r.name}/{bname}/{stype}"
                        output = pathlib.Path(
                            f"{fdir}/{obs.name.replace('/','_')}.{fig_fmt}"
                        )
                        output.parent.mkdir(parents=True, exist_ok=True)
                        canvas.SaveAs(f"{output.resolve()}")

                        if show_details:  # more sub-level components
                            for sub_name, sub_band in band.sub_bands.items():
                                sub_canvas = RootBackend.make_canvas(
                                    name=f"{canvas_name}/{sub_name}"
                                )
                                sub_canvas.cd()
                                if logy:
                                    sub_canvas.SetLogy()
                                sub_legend = RootBackend.make_legend(x1=0.40, x2=0.60)
                                c_obs = obs.copy(shallow=True)
                                c_obs.name = f"Details of : {sub_name}"
                                c_obs.ytitle = "(Syst.-Nom.)/Nom."
                                c_obs.bin_content = getattr(sub_band, stype)
                                c_obs.nan_to_num()
                                rhisto = c_obs.root
                                sys_style(rhisto)
                                rhisto.GetYaxis().SetRangeUser(*_range)
                                sub_legend.AddEntry(rhisto, c_obs.name)
                                rhisto.Draw(draw_opt)
                                color = 2
                                draw_opt = "HIST SAME"
                                components = sub_band.components[stype]
                                buffer = []
                                for comp_name, comp in components.items():
                                    c_obs = obs.copy(shallow=True)
                                    try:
                                        c_obs.name = f"{comp_name[0]}"
                                    except:
                                        c_obs.name = f"{comp_name}"
                                    c_obs.ytitle = "(Syst.-Nom.)/Nom."
                                    c_obs.bin_content = comp
                                    c_obs.nan_to_num()
                                    comp_rhisto = c_obs.root
                                    comp_rhisto.SetFillStyle(3444)
                                    comp_rhisto.SetFillColor(color)
                                    sys_style(comp_rhisto, sub_band.type)
                                    comp_rhisto.SetLineColor(color)
                                    sub_legend.AddEntry(comp_rhisto, c_obs.name)
                                    buffer.append(comp_rhisto)
                                    color += 1
                                    comp_rhisto.Draw(draw_opt)
                                sub_legend.Draw()
                                output = pathlib.Path(
                                    f"{fdir}/{obs.name.replac('/', '_')}_{sub_name}.{fig_fmt}"
                                )
                                sub_canvas.SaveAs(f"{output.resolve()}")

                    # drawing combine plots
                    canvas_name = f"{r}/{bname}/both/{obs}"
                    canvas = RootBackend.make_canvas(name=canvas_name)
                    canvas.cd()
                    x_start = 0.43
                    y1 = 0.8
                    _default_leg_pos = {
                        "x1": x_start + 0.2,
                        "x2": x_start + 0.35,
                        "y1": y1,
                        "y2": y1 + 0.05,
                        "text_size": 0.036,
                    }
                    _stats_leg_pos = {
                        "x1": x_start,
                        "x2": x_start + 0.15,
                        "y1": y1,
                        "y2": y1 + 0.05,
                        "text_size": 0.036,
                    }
                    _exp_leg_pos = exp_leg or {
                        "x1": x_start,
                        "x2": x_start + 0.15,
                        "y1": y1 - 0.20,
                        "y2": y1,
                        "text_size": 0.036,
                    }
                    _theo_leg_pos = theory_leg or {
                        "x1": x_start + 0.2,
                        "x2": x_start + 0.38,
                        "y1": y1 - 0.23,
                        "y2": y1,
                        "text_size": 0.036,
                    }
                    if legend_pos:
                        _default_leg_pos = legend_pos.get("default", _default_leg_pos)
                        _stats_leg_pos = legend_pos.get("stats", _stats_leg_pos)
                        _exp_leg_pos = legend_pos.get("experiment", _exp_leg_pos)
                        _theo_leg_pos = legend_pos.get("theory", _theo_leg_pos)
                    _default_leg = RootBackend.make_legend(**_default_leg_pos)
                    _stats_leg = RootBackend.make_legend(**_stats_leg_pos)
                    _experimental_leg = RootBackend.make_legend(**_exp_leg_pos)
                    _theory_leg = RootBackend.make_legend(**_theo_leg_pos)
                    legend = {
                        "default": _default_leg,
                        "stats": _stats_leg,
                        "experimental": _experimental_leg,
                        "theory": _theory_leg,
                        "unfold": _theory_leg,
                        "event": _experimental_leg,
                        "xsec": _experimental_leg,
                    }

                    if not yrange:
                        max_b = total["up"].GetMaximumBin()
                        max_val = total["up"].GetBinContent(max_b)
                        if max_val < 0.25:
                            _range = (-0.25, 0.25)
                        else:
                            _range = (-0.5, 0.5)

                    # do sorting base on the UP contribution
                    sort_names = []
                    sort_int = []
                    for name in total_components["up"].keys():
                        _up, _type = total_components["up"][name]
                        if type_map and name in type_map:
                            _type = type_map[name]
                        sort_int.append(np.sum(np.array(_up)))
                        sort_names.append(name)
                        legend.get(_type, legend["stats"]).AddEntry(
                            _up, common_name.get(name, name)
                        )
                    sort_names = [
                        x for _, x in sorted(zip(sort_int, sort_names), reverse=True)
                    ]

                    if symmetrize:
                        # perform symmetrization to each components
                        for _name in total_components['up']:
                            _up_histo = total_components['up'][_name]
                            try:
                                _dn_histo = total_components['down'][_name]
                            except KeyError:
                                continue
                            c_hist_up = _up_histo[0].Clone()
                            c_hist_dn = _dn_histo[0].Clone()
                            _up_histo[0].Add(c_hist_dn)
                            _up_histo[0].Scale(0.5)
                            _dn_histo[0].Add(c_hist_up)
                            _dn_histo[0].Scale(0.5)
                        c_hist_up = total["up"].Clone()
                        c_hist_dn = total["down"].Clone()
                        total["up"].Add(c_hist_dn)
                        total["up"].Scale(0.5)
                        total["down"].Add(c_hist_up)
                        total["down"].Scale(0.5)

                    # do drawing
                    draw_opt = "HIST"
                    for name in sort_names:
                        histo = total_components["up"][name]
                        histo[0].GetYaxis().SetRangeUser(*_range)
                        histo[0].GetYaxis().SetTitleOffset(1.5)
                        histo[0].SetFillStyle(0)  # overwrite exiting fill style
                        histo[0].SetFillColorAlpha(0, 0)  # overwrite exiting fill
                        histo[0].SetLineWidth(3)
                        histo[0].Draw(draw_opt)
                        draw_opt = "HIST same"
                        try:
                            histo = total_components["down"][name]
                            histo[0].Scale(-1.0)
                            histo[0].GetYaxis().SetRangeUser(*_range)
                            histo[0].SetFillStyle(0)  # overwrite exiting fill style
                            histo[0].SetFillColorAlpha(0, 0)  # overwrite exiting fill
                            histo[0].SetLineWidth(3)
                            histo[0].Draw(draw_opt)
                        except KeyError:
                            logger.warning(f"cannot find down band {name}")
                            continue
                    total["up"].GetYaxis().SetRangeUser(*_range)
                    total["down"].Scale(-1.0)
                    total["down"].GetYaxis().SetRangeUser(*_range)
                    if display_total:
                        total["up"].Draw(draw_opt)
                        if "same" not in draw_opt:
                            draw_opt = f"{draw_opt} same"
                        total["down"].Draw(draw_opt)
                        legend["default"].AddEntry(
                            total["up"], f"{common_name.get(bname, bname)}"
                        )

                    for leg in legend.values():
                        leg.SetFillStyle(0)
                        leg.Draw()

                    RootBackend.make_atlas_label()

                    fdir = f"{output_dir}/{process.name}/{r.name}/{bname}/both"
                    output = pathlib.Path(
                        f"{fdir}/{obs.name.replace('/','_')}.{fig_fmt}"
                    )
                    output.parent.mkdir(parents=True, exist_ok=True)
                    canvas.SaveAs(f"{output.resolve()}")

    def plot_processes(
        self,
        nominal,
        plist,
        output_dir="process_plots",
        low_yrange=(0.5, 1.5),
        rfilter=None,
        hfilter=None,
    ):
        """
        plot difference process comparing to the nominal.
        """
        sys_style = RootBackend.apply_systematic_style

        region_list = nominal.list_regions()
        obs_list = [obs.name for obs in nominal[0] if isinstance(obs, Histogram)]

        region_obs = [(r, h) for r in region_list for h in obs_list]

        for r, obs in region_obs:
            canvas = RootBackend.make_canvas(name=f"{r}/{obs}")
            canvas.SetLogy()
            canvas.cd()

            nominal_hist = nominal.get(r).get(obs).root
            self.data_style(nominal_hist)

            histo_buffer = [p.get(r).get(obs).root for p in plist]

            ndiv = 306
            ratio = ROOT.TRatioPlot(nominal_hist, nominal_hist, "divsym")
            ratio.SetH1DrawOpt("hist")
            ratio.SetH2DrawOpt("hist")
            ratio.Draw()
            ratio.GetLowYaxis().SetNdivisions(ndiv)
            ratio.GetLowerRefYaxis().SetLabelSize(0.04)
            ratio.GetLowerRefYaxis().SetTitleSize(0.04)
            ratio.GetLowerRefYaxis().SetLabelSize(0.03)
            ratio.GetLowerRefYaxis().SetTitleSize(0.05)
            ratio.GetLowerRefYaxis().SetRangeUser(*low_yrange)
            ratio.SetGridlines([1])
            ratios = [ratio]

            for i, h in enumerate(histo_buffer, start=2):
                ratio = h / nominal_hist
                sys_style(h)
                sys_style(ratio)
                if len(plist) < 15:
                    h.SetLineColor(i)
                    ratio.SetLineColor(i)
                ratios[0].GetUpperPad().cd()
                h.Draw("hist SAME")
                ratios[0].GetLowerPad().cd()
                ratio.Draw("SAME E")
                ratios.append(ratio)

            legend = RootBackend.make_legend(x1=0.40, x2=0.60)
            legend.AddEntry(nominal_hist, f"{nominal.title}", "p")
            n_plist = len(plist)
            if n_plist < 15:
                for i, h in enumerate(histo_buffer):
                    legend.AddEntry(h, f"{plist[i].title}")
            ratios[0].GetUpperPad().cd()
            legend.Draw()

            output = pathlib.Path(f"{output_dir}/{r}/{obs}.png")
            output.parent.mkdir(parents=True, exist_ok=True)
            canvas.SaveAs(f"{output.resolve()}")
