import warnings
import logging
import concurrent.futures
import fnmatch
import copy
import numpy as np
from tqdm import tqdm
from pathlib import Path
from matplotlib import pyplot
from cycler import cycler

from . import histManipulate
from .plotMaker import PlotMaker, PlotJob, COLOR_HEX
from .configs import ConfigMgr
from .histo import Histogram, Histogram2D
from .systematics import SystematicsBand
from .systematics import tools as syst_tools
from .backends import RootBackend

try:
    import ROOT
except ImportError:
    warnings.warn("Cannot import ROOT module!")

logging.basicConfig()
logging.getLogger().setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def run_PlotMaker(config_name):
    logger.info("Running run_PlotMaker")

    my_configMgr = ConfigMgr.open(config_name)

    plot_maker = PlotMaker(my_configMgr)
    plot_maker.draw_single()

    logger.info("after processing \n\n\n")


# ==============================================================================

# ==============================================================================


def _region_stack(
    region_name,
    my_plotMaker,
    output,
    observable_list,
    data_process,
    mc_processes,
    xrange,
    yrange,
    low_yrange,
    low_ytitle,
    text,
    logy,
    show_text,
    label_txt,
    invert_ratio=False,
    systematic=None,
    include_systematic_band=True,
    external_syst_process=None,
    hide_process=False,
    legend_opt="",
    enable_legend=True,
    legend_pos=(0.45, 0.55, 0.65, 0.85),
    legend_txt_size=None,
    do_sort=True,
    divide_binwidth=True,
    ytitles=None,
    figfmt="pdf",
):
    sub_dir = f"{output}/{region_name.replace('/', '_slash_')}/"

    try:
        data_region = data_process.get(systematic).get_region(region_name)
    except:
        data_region = None

    mc_region_dict = {}
    for mc in mc_processes:
        try:
            mc_region_dict[mc.name] = mc.get(systematic).get_region(region_name)
        except KeyError:
            logger.warning(f"{mc.name}/{systematic} has no region {region_name}")
            continue

    # loop through observables
    with tqdm(total=len(observable_list)) as pbar:
        for obs in observable_list:
            pbar.set_description(obs)
            pbar.update()

            band_histogram = None
            if include_systematic_band and external_syst_process:
                try:
                    band_region = external_syst_process.get_region(region_name)
                    band_histogram = band_region.get_observable(obs)
                    if hasattr(band_histogram, "systematic_band"):
                        m_bands = band_histogram.systematic_band
                        if m_bands:
                            for band in m_bands.values():
                                band.scale_components(band_histogram.bin_content)
                    # statistical error
                    if hasattr(band_histogram, "statistical_error"):
                        stats_error = band_histogram.statistical_error()
                        stats_band = SystematicsBand(
                            "stats", "stats", band_histogram.shape
                        )
                        stats_band.add_component("up", "stats", stats_error["up"])
                        stats_band.add_component("down", "stats", stats_error["down"])
                        band_histogram.update_systematic_band(stats_band)
                except KeyError:
                    logger.warning(f"unable to get syst band {region_name}/{obs}")
                    pass
            elif include_systematic_band:
                norm_region = mc_processes[0].get().get_region(region_name)
                nominal_tot = norm_region.get_observable(obs).copy()
                if include_systematic_band and getattr(nominal_tot, "systematic_band"):
                    stats_error = nominal_tot.statistical_error()
                    for band in nominal_tot.systematic_band.values():
                        band.scale_components(nominal_tot.bin_content)
                    for mc in mc_processes[1:]:
                        try:
                            other = mc.get().get_region(region_name).get_observable(obs)
                            nominal_tot += other
                            other_stats_error = other.statistical_error()
                            stats_error["up"] = np.sqrt(
                                stats_error["up"] ** 2 + other_stats_error["up"] ** 2
                            )
                            stats_error["down"] = np.sqrt(
                                stats_error["down"] ** 2
                                + other_stats_error["down"] ** 2
                            )
                            for band_name, band in other.systematic_band.items():
                                band_copy = copy.deepcopy(band)
                                band_copy.scale_components(other.bin_content)
                                if band_name not in nominal_tot.systematic_band:
                                    new_band = {band_name: band_copy}
                                    nominal_tot.systematic_band.update(new_band)
                                else:
                                    nominal_tot.systematic_band[band_name].combine(
                                        band_copy
                                    )
                        except KeyError:
                            continue
                    stats_band = SystematicsBand(
                        "statistical", "stats", nominal_tot.shape
                    )
                    stats_band.add_component("up", "stats", stats_error["up"])
                    stats_band.add_component("down", "stats", stats_error["down"])
                    nominal_tot.update_systematic_band(stats_band)
                    band_histogram = nominal_tot
                    # band_histogram["nominal"] = nominal_tot

            legend = my_plotMaker.make_legend(*legend_pos)
            legend.SetNColumns(2)
            if legend_txt_size:
                legend.SetTextSize(legend_txt_size)

            if data_region:
                try:
                    data_hist = data_region.get_observable(obs)
                except KeyError:
                    continue
                if isinstance(data_hist, Histogram2D):
                    continue
                if data_hist.integral() < 1e-3:
                    continue
                data_hist.remove_negative_bin()
                data_hist.nan_to_num()
                data_hist = data_hist.root
                my_plotMaker.data_style(data_hist)
                data_name = data_process.title or data_process.name
                data_plotJob = PlotJob(
                    data_name, data_hist.Clone(), "pEX0", obs, divide_binwidth
                )
                data_plotJob.legend_style = "pe"
                my_plotMaker.update_legend(legend, legend_opt, data_plotJob)
            else:
                data_hist = None
                data_plotJob = None

            mc_hist_dict = {}
            for mc, mc_region in mc_region_dict.items():
                if isinstance(mc_region.parent, str):
                    mc_name = mc
                else:
                    mc_name = mc_region.parent.name or mc
                    # try with legend title
                    mc_name = mc_region.parent.title or mc_name
                try:
                    mc_hist_dict[mc_name] = mc_region.get_observable(obs)
                    mc_hist_dict[mc_name].remove_negative_bin()
                    mc_hist_dict[mc_name].nan_to_num()
                except KeyError:
                    logger.warning(f"{mc_name}/{mc_region.name} has no {obs}")
                    continue
            if not mc_hist_dict:
                if data_region:
                    mc_hist_dict[data_process.name] = data_region.get_observable(obs)
                else:
                    logger.warning("mc hist dict is empty")
                    continue

            mc_stack_plotJob = my_plotMaker.stack(
                mc_hist_dict,
                f"mc_{obs}_stack",
                legend,
                legend_opt,  # "int",
                hide_process=hide_process,
                do_sort=do_sort,
                divide_binwidth=divide_binwidth,
            )

            if isinstance(xrange, dict):
                if obs in xrange:
                    logger.debug(f"replace {obs} xrange {xrange[obs]}")
                    m_xrange = xrange[obs]
                else:
                    m_xrange = None
            else:
                m_xrange = xrange

            if isinstance(yrange, dict):
                if obs in yrange:
                    logger.debug(f"replace {obs} yrange {yrange[obs]}")
                    m_yrange = yrange[obs]
                else:
                    m_yrange = None
            else:
                m_yrange = yrange

            if data_plotJob:
                _xmin = data_region.get_observable(obs).bins[0]
                _xmax = data_region.get_observable(obs).bins[-1]
                ratio = my_plotMaker.ratio_plot(
                    mc_stack_plotJob,
                    data_plotJob,
                    low_ytitle,  # "data/MCs",
                    yrange=m_yrange,
                    xrange=m_xrange or (_xmin, _xmax),
                    low_yrange=low_yrange,
                    invert_ratio=invert_ratio,
                    width=0,
                )
                is_ratio = True
            else:
                ratio = mc_stack_plotJob
                my_plotMaker.set_range(ratio, xrange=m_xrange, yrange=m_yrange)
                is_ratio = False

            if text is None:
                add_text = [region_name]
            else:
                add_text = text

            try:
                my_plotMaker.save_plot(
                    ratio,
                    sub_dir,
                    obs.replace("/", "_over_").replace(" ", ""),
                    legend=legend if enable_legend else None,
                    logy=logy,
                    figfmt=figfmt,
                    text=add_text,
                    show_text=show_text,
                    label_txt=label_txt,
                    is_ratio=is_ratio,
                    syst_band_histogram=band_histogram,
                    divide_binwidth=divide_binwidth,
                    ytitles=ytitles,
                )
            except Exception as e:
                logger.critical(f"Unable to save {sub_dir}/{obs}: {e}")


def run_stack(
    config,
    output,
    *,
    data="data",
    mcs=None,
    text=None,
    xrange=None,
    yrange=None,
    low_yrange=None,
    low_ytitle="data/MCs",
    label_txt=None,
    mc_name_filter=None,
    mc_type_filter=None,
    rname_filter=None,
    show_text=False,
    logy=True,
    check_region=False,
    workers=None,
    invert_ratio=True,
    systematic=None,
    include_systematic_band=False,
    compute_systematic_groups=[],
    lookup_systematic_groups={},
    external_syst_process=None,
    hide_process=False,
    legend_opt="",
    enable_legend=True,
    legend_pos=(0.45, 0.55, 0.65, 0.85),
    legend_txt_size=None,
    do_sort=True,
    divide_binwidth=True,
    ytitles=None,
    figfmt="pdf",
):
    my_config = ConfigMgr.open(config)

    my_plotMaker = PlotMaker(my_config, output)

    if isinstance(mc_name_filter, str):
        mc_name_filter = [mc_name_filter]
    if isinstance(mc_type_filter, str):
        mc_type_filter = [mc_type_filter]
    if isinstance(rname_filter, str):
        rname_filter = [rname_filter]

    # update regions that are not store in the config level.
    # e.g. tf regions are not record on the config level.
    region_list = set(my_config.region_list)
    if check_region:
        for p in config.processes:
            region_list |= set(p.list_regions())
    if rname_filter:
        region_list = [
            r
            for r in region_list
            if any(fnmatch.fnmatch(r, filter) for filter in rname_filter)
        ]

    data_process = my_config.get_process_set(data) if data else None

    # storing non data processes for comparison against data.
    mc_processes = []
    if mcs:
        for mc in mcs:
            try:
                mc_processes.append(my_config.get_process_set(mc))
            except KeyError:
                logger.warning(f"cannot find process set {mc}")
                continue
    else:
        for pname in my_config.list_processes():
            if pname == data_process.name:
                continue
            if mc_name_filter:
                if not any(fnmatch.fnmatch(pname, filter) for filter in mc_name_filter):
                    continue
            if mc_type_filter:
                if not any(fnmatch.fnmatch(pname, filter) for filter in mc_type_filter):
                    continue
            mc_processes.append(my_config.get_process_set(pname))

    # to compute systematic band, first computing the sum of MCs
    if include_systematic_band and external_syst_process is None:
        mc_sum = histManipulate.sum_process_sets(mc_processes)

        # generate dict of systematic groups
        syst_groups = {}
        # check if computation on systematics is needed?
        if compute_systematic_groups:
            for syst_group in compute_systematic_groups:
                # syst_group should be tuple (parent name, algorithm name)
                # e.g. ("wjets_2211_MUR_MUF_Scale", "min_max")
                syst_tools.compute_process_set_systematics(mc_sum, *syst_group)
        if lookup_systematic_groups:
            for syst_type, syst_lookup_list in lookup_systematic_groups.items():
                logger.info(f"found {syst_type} type systematic")
                syst_groups[syst_type] = {}
                for lookup_key in syst_lookup_list:
                    syst_groups[syst_type].update(
                        mc_sum.generate_systematic_group(*lookup_key)
                    )

        # compute quad sum for syst.
        for syst_type in syst_groups:
            for name, syst_list in syst_groups[syst_type].items():
                logger.info(f"computing {syst_type} {name}.")
                syst_tools.compute_quadrature_sum(
                    config,
                    "",
                    name,
                    syst_type,
                    syst_list,
                    external_process_set=mc_sum,
                )
        external_syst_process = mc_sum.nominal

    observable_list = [h.name for h in my_config.histograms]

    # passing all regions in config and make stack plots of Observables
    # for different processes.
    const_args = (
        my_plotMaker,
        output,
        observable_list,
        data_process,
        mc_processes,
        xrange,
        yrange,
        low_yrange,
        low_ytitle,
        text,
        logy,
        show_text,
        label_txt,
        invert_ratio,
        systematic,
        include_systematic_band,
        external_syst_process,
        hide_process,
        legend_opt,
        enable_legend,
        legend_pos,
        legend_txt_size,
        do_sort,
        divide_binwidth,
        ytitles,
        figfmt,
    )
    if workers:
        with concurrent.futures.ProcessPoolExecutor(workers) as exe:
            for region_name in region_list:
                exe.submit(_region_stack, region_name, *const_args)
    else:
        for region_name in region_list:
            _region_stack(region_name, *const_args)


# ==============================================================================


# ==============================================================================
def plot_processes(
    config,
    output,
    *,
    process=None,
    pname_filter=None,
    ptype_filter=None,
    rname_filter=None,
    xrange=None,
    yrange=None,
    logy=True,
    text=None,
    label_txt=None,
    show_text=False,
    enable_legend=True,
    enable_systematic_band=True,
    legend_args=None,  # dict of args used in the legend
    fmt="png",
):
    my_config = ConfigMgr.open(config)
    my_plotMaker = PlotMaker(my_config, output)

    if isinstance(pname_filter, str):
        pname_filter = [pname_filter]
    if isinstance(ptype_filter, str):
        ptype_filter = [ptype_filter]
    if isinstance(rname_filter, str):
        rname_filter = [rname_filter]

    if legend_args is None:
        legend_args = {}

    # storing non data processes for comparison against data.
    m_processes = []
    if process:
        for p in process:
            try:
                m_processes.append(my_config.get_process(p))
            except KeyError:
                logger.debug(f"Unable to get {p}.")
                continue
    else:
        for pname in my_config.list_processes():
            if pname_filter:
                if not any(fnmatch.fnmatch(pname, filter) for filter in pname_filter):
                    continue
            if ptype_filter:
                if not any(fnmatch.fnmatch(pname, filter) for filter in ptype_filter):
                    continue
            m_processes.append(my_config.get_process(pname))

    # update regions that are not store in the config level.
    # e.g. tf regions are not record on the config level.
    region_list = set(my_config.region_list)
    for p in m_processes:
        region_list |= set(p.list_regions())
    if rname_filter:
        region_list = [
            r
            for r in region_list
            if any(fnmatch.fnmatch(r, filter) for filter in rname_filter)
        ]

    observable_list = my_config.observable_list

    for region_name in region_list:
        sub_dir = f"{output}/{region_name.replace('/', '_slash_')}/"

        for obs in observable_list:
            if isinstance(xrange, dict):
                if obs in xrange:
                    logger.debug(f"replace {obs} xrange {xrange[obs]}")
                    m_xrange = xrange[obs]
                else:
                    m_xrange = None
            else:
                m_xrange = xrange

            if isinstance(yrange, dict):
                if obs in yrange:
                    logger.debug(f"replace {obs} yrange {yrange[obs]}")
                    m_yrange = yrange[obs]
                else:
                    m_yrange = None
            else:
                m_yrange = yrange

            process_plot_jobs = []
            legend = my_plotMaker.make_legend(**legend_args)
            draw_opt = "APE"
            alpha = 0.5
            for color, p in enumerate(m_processes):
                try:
                    obs_hist = p.get(region_name).get(obs)
                    if isinstance(obs_hist, Histogram2D):
                        continue
                except KeyError:
                    logger.debug(f"Cannot find {p.name}/{region_name}/{obs}")
                    continue
                try:
                    m_color = ROOT.TColor.GetColor(COLOR_HEX[color])
                except IndexError:
                    m_color = ROOT.TColor.GetColor(COLOR_HEX[-1])

                if not enable_systematic_band:
                    obs_hist = obs_hist.root
                    obs_hist.SetLineWidth(3)
                    obs_hist.SetLineColor(m_color)
                    plot_job = PlotJob(p.name, obs_hist, draw_opt.replace("A", "H"))
                    my_plotMaker.update_legend(legend, "", plot_job)
                    my_plotMaker.set_range(plot_job, xrange=m_xrange, yrange=m_yrange)
                    process_plot_jobs.append(plot_job)
                else:
                    obs_stat = obs_hist.root_graph("Statistical")
                    RootBackend.apply_styles(
                        obs_stat,
                        color=m_color,
                        linewidth=1,
                        linestyle=2,
                        markerstyle=8,
                        markersize=1.2,
                    )
                    stat_pj = PlotJob(f"{p.name}(stat)", obs_stat, draw_opt)
                    legend.AddEntry(obs_stat, f"{p.name}(stat.)", "PE1X0")
                    my_plotMaker.set_range(stat_pj, xrange=m_xrange, yrange=m_yrange)
                    process_plot_jobs.append(stat_pj)
                    draw_opt = "PE"
                    if obs_hist.systematic_band is None:
                        continue
                    obs_syst = obs_hist.root_graph("Systematic")
                    RootBackend.apply_styles(obs_syst, color=m_color)
                    obs_syst.SetFillColorAlpha(m_color, alpha)
                    syst_pj = PlotJob(f"{p.name}(syst.)", obs_syst, "2")
                    legend.AddEntry(obs_syst, f"{p.name}(syst.)", "F")
                    my_plotMaker.set_range(syst_pj, xrange=m_xrange, yrange=m_yrange)
                    process_plot_jobs.append(syst_pj)

            if process_plot_jobs:
                my_plotMaker.save_plot(
                    process_plot_jobs,
                    sub_dir,
                    obs.replace("/", "_over_").replace(" ", ""),
                    legend=legend if enable_legend else None,
                    logy=logy,
                    figfmt=fmt,
                    text=text,
                    show_text=show_text,
                    label_txt=label_txt,
                )


# ==============================================================================


# ==============================================================================
def plot_histogram_stack(
    config,
    histogram,
    histograms,
    output_name,
    output_dir="stack_region",
    fmt="png",
    xrange=None,
    yrange=None,
    low_yrange=None,
    low_ytitle="data/MCs",
    label_txt=None,
    show_text=False,
    logy=True,
    check_region=False,
    workers=None,
    invert_ratio=False,
    hide_process=False,
    legend_opt="",
    enable_legend=True,
    legend_pos=(0.45, 0.55, 0.65, 0.85),
):
    """
    histogram : tuple, (name, histogram)
    histograms : dict, {name : histogram}
    """
    config = ConfigMgr.open(config)
    m_plotmaker = PlotMaker(config, output_dir)

    legend = m_plotmaker.make_legend(*legend_pos)

    name, histogram = histogram
    histogram = histogram.root
    m_plotmaker.data_style(histogram)
    pj_histogram = PlotJob(name, histogram, "pE")
    m_plotmaker.update_legend(legend, legend_opt, pj_histogram)

    stack_histograms = m_plotmaker.stack(
        histograms, "stack", legend, legend_opt, hide_process=hide_process
    )

    if isinstance(xrange, dict):
        if name in xrange:
            logger.debug(f"replace {name} xrange {xrange[name]}")
            m_xrange = xrange[name]
        else:
            m_xrange = None
    else:
        m_xrange = xrange

    if isinstance(yrange, dict):
        if name in yrange:
            logger.debug(f"replace {name} yrange {yrange[name]}")
            m_yrange = yrange[name]
        else:
            m_yrange = None
    else:
        m_yrange = yrange

    ratio = m_plotmaker.ratio_plot(
        stack_histograms,
        pj_histogram,
        low_ytitle,  # "data/MCs",
        yrange=m_yrange,
        xrange=m_xrange,
        low_yrange=low_yrange,
        invert_ratio=invert_ratio,
    )

    try:
        m_plotmaker.save_plot(
            ratio,
            output_dir,
            output_name,
            legend=legend if enable_legend else None,
            logy=logy,
            figfmt=fmt,
            label_txt=label_txt,
            is_ratio=True,
        )
    except Exception as e:
        logger.critical(f"Unable to save {output_dir}/{output_name}: {e}")


# ==============================================================================


# ==============================================================================
def plot_ratio(
    config,
    output,
    base_process,
    comp_process,
    *,
    rname_filter=None,
    xrange=None,
    yrange=None,
    low_yrange=None,
    logy=True,
    text=None,
    label_txt=None,
    show_text=False,
):
    my_config = ConfigMgr.open(config)
    my_plotMaker = PlotMaker(my_config, output)

    # update regions that are not store in the config level.
    # e.g. tf regions are not record on the config level.
    region_list = set(my_config.region_list)
    region_list |= set(my_config.get_process(base_process).list_regions())
    if rname_filter:
        region_list = [
            r
            for r in region_list
            if any(fnmatch.fnmatch(r, filter) for filter in rname_filter)
        ]

    observable_list = my_config.observable_list

    base_process = my_config.get_process(base_process)

    for region_name in tqdm(region_list):
        sub_dir = f"{output}/{region_name.replace('/', '_slash_')}/"

        for obs in observable_list:
            if isinstance(xrange, dict):
                if obs in xrange:
                    logger.debug(f"replace {obs} xrange {xrange[obs]}")
                    m_xrange = xrange[obs]
                else:
                    m_xrange = None
            else:
                m_xrange = xrange

            if isinstance(yrange, dict):
                if obs in yrange:
                    logger.debug(f"replace {obs} yrange {yrange[obs]}")
                    m_yrange = yrange[obs]
                else:
                    m_yrange = None
            else:
                m_yrange = yrange

            legend = my_plotMaker.make_legend(0.45, 0.65, 0.65, 0.90)

            try:
                base = base_process.get_region(region_name).get_observable(obs)
                if isinstance(base, Histogram2D):
                    continue
                base.remove_negative_bin()
                base = base.root
                base.SetLineWidth(2)
                base.SetLineColor(ROOT.kBlack)
            except:
                continue

            base_plot_job = PlotJob(f"base:{base_process.title}", base, "hpE")
            my_plotMaker.update_legend(legend, "int", base_plot_job)

            color = 2
            comp_plot_job = []
            for color, comp_p in enumerate(comp_process):
                try:
                    m_process = my_config.get_process(comp_p)
                    title = m_process.title
                    comp_hist = m_process.get_region(region_name).get_observable(obs)
                    comp_hist.remove_negative_bin()
                    comp_hist = comp_hist.root
                except:
                    logger.debug(f"no {obs} from {comp_p} in {region_name}")
                    continue
                try:
                    m_color = ROOT.TColor.GetColor(COLOR_HEX[color])
                except IndexError:
                    m_color = ROOT.TColor.GetColor(COLOR_HEX[-1])
                comp_hist.SetLineColor(m_color)
                comp_hist.SetFillColorAlpha(m_color, 0.4)
                # comp_hist.SetLineStyle(color)
                comp_hist.SetLineWidth(2)
                comp_hist.SetFillStyle(3001 + color)
                color += 1
                _plot_job = PlotJob(title, comp_hist, "hpE")
                my_plotMaker.update_legend(legend, "int", _plot_job)
                comp_plot_job.append(_plot_job)

            if comp_plot_job:
                my_plotMaker.multiple_ratio(
                    sub_dir,
                    obs,
                    base_plot_job,
                    comp_plot_job,
                    "ratios",
                    legend=legend,
                    xrange=m_xrange,
                    yrange=m_yrange,
                    low_yrange=low_yrange,
                    logy=logy,
                    text=text,
                    show_text=show_text,
                    label_txt=label_txt,
                )


# ==============================================================================


# ==============================================================================
def plot_histograms(
    config,
    histograms,
    output="example",
    *,
    label=None,
    batch=False,
    yrange=None,
    xrange=None,
):
    """
    plot histograms

    histograms (Histogram or list(Histogram)) : histogram object for plotting.
    label (str or list(str)) : legend label for histograms.
    output (str) : name for saving image.
    batch (bool) : batch mode.
    """

    if not isinstance(histograms, list):
        histograms = [histograms]
    if label and not isinstance(label, list):
        label = [label]

    plotmaker = PlotMaker(config, "plots")
    leg = plotmaker.make_legend()
    if not label:
        label = [histo.name for histo in histograms]

    histograms = zip(histograms, label)
    plot_jobs = []
    color = 2
    for histo, tag in histograms:
        job = PlotJob(tag, histo, "h")
        plotmaker.root_set_color(job.histogram, color)
        plotmaker.update_legend(leg, "", job)
        plot_jobs.append(job)
        color += 1

    plotmaker.save_plot(plot_jobs, leg, "", output, yrange=yrange, xrange=xrange)


# ==============================================================================

# ==============================================================================


def run_ABCD_ABCompare(
    config_name, process, text, yrange, *, xrange=None, otag="compare_AB"
):
    """
    Compare shape of observable from region A and B from the ABCD method.
    """

    my_configMgr = ConfigMgr.open(config_name)

    process = my_configMgr.get_process(process)
    abcd_meta = my_configMgr.meta_data["abcd"]
    tag_list = abcd_meta.keys()
    for tag in tag_list:
        for case in abcd_meta[tag].cases:
            output_tag = f"{otag}/{process.name}/{case.parent}/{tag}/"
            my_plotMaker = PlotMaker(my_configMgr, output_tag)
            rA_name, _ = case["A"]
            rB_name, _ = case["B"]
            region_A = process.get_region(rA_name)
            region_B = process.get_region(rB_name)
            # name_tag = f"{process.name}-{tag}"
            for obs in region_A.histograms:
                my_plotMaker.compare_2regions(
                    region_A,
                    region_B,
                    obs.name,
                    leg1="region-A",
                    leg2="region-B",
                    text=f"{tag}",
                    yrange=yrange,
                    oname="",
                    y_lowtitle="Region A/B",
                    xrange=xrange,
                )


# ==============================================================================


# ==============================================================================
def plot_pset_systematics(config, use_mp=True, process_filter=None, *args, **kwargs):
    plotmaker = PlotMaker(config)

    if process_filter:
        for proc in process_filter:
            config.remove_process_set(proc)

    if use_mp:
        with concurrent.futures.ProcessPoolExecutor() as pool:
            for pset in config.process_sets:
                pool.submit(plotmaker.plot_pset_systematics, pset, *args, **kwargs)
    else:
        for pset in config.process_sets:
            plotmaker.plot_pset_systematics(pset, *args, **kwargs)


# ==============================================================================

# ==============================================================================


def plot_pset_relative_systematics(
    config,
    process_list=[],
    show_sum=True,
    use_mp=False,
    *args,
    **kwargs,
):
    plotmaker = PlotMaker(config)

    if use_mp:
        with concurrent.futures.ProcessPoolExecutor() as pool:
            for pset in config.process_sets:
                pool.submit(plotmaker.plot_relative_systematics, pset, *args, **kwargs)
    else:
        if not process_list:
            process_list = config.list_processes()
        processes = [config.get_process(pname) for pname in process_list]
        if show_sum and len(processes) > 1:
            total_process = processes[0].copy()
            total_process.name = "Sum of MCs"
            for region in total_process.regions:
                for obs in region.histograms:
                    if isinstance(obs, Histogram2D):
                        continue
                    # scale with nominal content to get actual difference.
                    for band in obs.systematic_band.values():
                        band.scale_components(obs.bin_content)
                    for proc in processes[1:]:
                        try:
                            other = proc.get_region(region.name).get_observable(
                                obs.name
                            )
                        except KeyError:
                            continue
                        obs += other
                        for band_name, band in other.systematic_band.items():
                            band_copy = copy.deepcopy(band)
                            band_copy.scale_components(other.bin_content)
                            if band_name not in obs.systematic_band:
                                new_band = {band_name: band_copy}
                                obs.systematic_band.update(new_band)
                            else:
                                obs.systematic_band[band_name].combine(band_copy)
                    # rescale the difference with respect to the nominal
                    for band in obs.systematic_band.values():
                        band.scale_components(1.0 / obs.bin_content)
                    # breakpoint()
            processes.append(total_process)
        for process in processes:
            plotmaker.plot_relative_systematics(process, *args, **kwargs)


# ==============================================================================


# ==============================================================================
def plot_reco_relative_systematics(
    config,
    sum_process,
    comp_process_list,
    output="reco_relative_syst",
):
    sum_process = config.get_process(sum_process)
    comp_process_list = [config.get_process(x) for x in comp_process_list]
    for region in sum_process:
        if region.type != "reco":
            continue
        for hist in region:
            if hist.hist_type != "1d":
                continue
            sum_band = hist.total_band()
            if sum_band is None:
                continue
            sum_avg = (sum_band.up + sum_band.down) * 0.5
            sum_avg_hist = hist.copy()
            sum_avg_hist.bin_content *= sum_avg

            comp_list = []
            for comp in comp_process_list:
                comp_hist = comp.get(region.name).get(hist.name)
                comp_band = comp_hist.total_band()
                if comp_band is None:
                    continue
                comp_band_avg = (comp_band.up + comp_band.down) * 0.5
                comp_band_hist = comp_hist.copy()
                comp_band_hist.bin_content *= comp_band_avg
                comp_list.append((comp, comp_band_hist))

            canvas = RootBackend.make_canvas(name=f"{region.name}/{hist.name}")
            legend = RootBackend.make_legend(x1=0.40, x2=0.60, y1=0.5)
            canvas.cd()

            draw_opt = "H"
            cache = []
            for proc, h in comp_list:
                h = h / sum_avg_hist
                h.nan_to_num()
                h.ytitle = "Rel. uncert. fraction"
                rhist = h.root
                cache.append(rhist)
                rhist.SetLineColor(ROOT.TColor.GetColor(proc.color))
                rhist.SetLineWidth(2)
                rhist.GetYaxis().SetRangeUser(0, 1.2)
                rhist.Draw(draw_opt)
                legend.AddEntry(rhist, proc.title)
                draw_opt = "H same"

            legend.Draw()

            opath = Path(f"{output}/{region.name}/{hist.name}.png")
            opath.parent.mkdir(parents=True, exist_ok=True)
            canvas.SaveAs(f"{opath.resolve()}")


# ==============================================================================


# ==============================================================================
def plot_purity(
    config,
    signal_list=["wjets_2211"],
    output="purity",
    plot_response=True,
    name_map={},
    axis=0,
):
    """
    Plot purity distirbution for reco-truth matched response matrix within
    the instance of ConfigMgr.
    """
    config = ConfigMgr.open(config)

    signals = []
    for s in signal_list:
        signals.append(config.get_process(s))

    # open unfolding metadata
    regions = config.meta_data["unfold"]["regions"]
    observables = config.meta_data["unfold"]["observables"]

    for unfold_name, region in regions.items():
        for observable, histogram in observables.items():
            region_name = region["reco_match"]
            hist_name = histogram["response"]

            canvas = RootBackend.make_canvas(f"{observable}_purity", width=1000)
            canvas.cd()

            cmd = 'E'
            histos = []

            legend = RootBackend.make_legend(x1=0.46, x2=0.78, y1=0.65, y2=0.87)

            for signal in signals:
                hResponse = signal.get_region(region_name).get_histogram(hist_name)

                purity = hResponse.purity(axis).root
                RootBackend.apply_process_styles(purity, signal)
                purity.GetYaxis().SetRangeUser(0, 1.5)
                purity.SetLineColor(signal.color)
                purity.SetMarkerColor(signal.color)
                purity.SetLineWidth(3)
                purity.Draw(cmd)
                histos.append(purity)
                cmd = 'E SAME'

                label_name = signal.name
                if name_map.get(signal.name):
                    label_name = name_map[signal.name]

                legend.AddEntry(purity, label_name)

            # atlas label
            RootBackend.make_atlas_label()

            #
            legend.Draw()

            # unity line
            l_xmin = purity.GetXaxis().GetXmin()
            l_xmax = purity.GetXaxis().GetXmax()
            unit_line = RootBackend.make_line(l_xmin, 1, l_xmax, 1)
            unit_line.Draw()

            base_output_path = (
                Path(config.out_path)
                .joinpath("plots", output, region_name, unfold_name, observable)
                .resolve()
            )
            output_path = base_output_path.with_suffix('.pdf')
            output_path.parent.mkdir(parents=True, exist_ok=True)
            canvas.SaveAs(str(output_path))


# ==============================================================================


# ==============================================================================
def plot_bootstrap(bhists, output, figsize=None, ref_hist=None, show_error=True):
    """
    Plotting bootstrap histogram bin by bin.

    Args:
        bhists: list(strategies.bootstrap.npbackend.HistogramBootstrap)
            list of bootstrap histograms

        output: str
            output name of the plot.

        figsize: tuple, default=None
            figure size.

        ref_hist: core.Histogram, default=None
            histogram object for getting the reference values.
            e.g. the mean values.

        show_error: bool, default=True
            show error bound for each of bootstrap histogram bins.
    """
    nbins = len(bhists[0].bin_content[1:-1])  # exclude overflow
    ncols = min(5, nbins)
    if nbins > ncols:
        if nbins % ncols == 0:
            nrows = nbins // ncols
        else:
            nrows = nbins // ncols + 1
    else:
        nrows = 1
    jobs = ((h, i) for h in bhists for i in range(1, nbins + 1))

    if figsize is None:
        figsize = (15, 4 * nrows)

    with pyplot.rc_context(rc={'axes.prop_cycle': cycler(color=['b', 'r', 'g', 'y'])}):
        fig, axes = pyplot.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        fig.tight_layout()
        axes = axes.ravel() if ncols != 1 else [axes]
        for bhist, i in jobs:
            hist = Histogram.array_to_hist(bhist.replica[:, i])
            _h = axes[i - 1].stairs(hist.bin_content[1:-1], hist.bins)
            axes[i - 1].axvline(bhist.bin_content[i], color=_h.get_ec())
            # better to cache the sysetmatic band to avoid repeat computation
            if show_error and bhist.systematic_band is not None:
                _up = bhist.bin_content[i] * (1 + bhist.total_band().up[i])
                _dn = bhist.bin_content[i] * (1 - bhist.total_band().down[i])
                axes[i - 1].axvline(_up, linestyle="--", color=_h.get_ec())
                axes[i - 1].axvline(_dn, linestyle="--", color=_h.get_ec())

        if ref_hist is not None:
            opt = {"linestyle": "--", "color": "g"}
            _syst_up = ref_hist.systematic_band["total"].up
            _syst_dn = ref_hist.systematic_band["total"].down
            for i in range(1, nbins + 1):
                _center = ref_hist.bin_content[i]
                axes[i - 1].axvline(_center, color=opt['color'])
                axes[i - 1].axvline(_center * (1 + _syst_up[i]), **opt)
                axes[i - 1].axvline(_center * (1 - _syst_dn[i]), **opt)

    fig.savefig(output)


# ==============================================================================


# ==============================================================================
def plot_reco_bkg_frac(
    config,
    signal="wjets_2211",
    regions=None,
    output="bkg_frac",
    bkg_processes=None,
    fmt="png",
):
    config = ConfigMgr.open(config)
    if bkg_processes is None:
        bkg_processes = [x for x in config.processes if x.process_type == "bkg"]
    else:
        bkg_processes = [config.get_process(x) for x in bkg_processes]
    signal_process = config.get(signal).get()

    # adding all backgrounds with signal process
    total_process = signal_process.copy()
    for bkg in bkg_processes:
        total_process.add(bkg)

    apply_styles = RootBackend.apply_styles

    yrange = (0, 0.5)
    if regions is None:
        regions = total_process.regions
    else:
        regions = (total_process.get(r) for r in regions)
    regions_hists = ((r, h) for r in regions for h in r.histograms)
    for r, h in regions_hists:
        ratio_cache = {}
        for i, bkg in enumerate(bkg_processes):
            ratio = bkg.get(r.name).get(h.name) / h
            ratio.ytitle = "MC. bkg / (bkg + signal)"
            ratio.nan_to_num()
            ratio_rh = ratio.root
            color = ROOT.TColor.GetColor(COLOR_HEX[i % len(COLOR_HEX)])
            apply_styles(ratio_rh, color=color, linewidth=3)
            ratio_rh.SetLineColor(color)
            ratio_rh.GetYaxis().SetRangeUser(*yrange)
            ratio_rh.GetYaxis().SetTitleOffset(1.5)
            ratio_cache[bkg.title] = ratio_rh

        legend = RootBackend.make_legend(x1=0.52, x2=0.60, y1=0.5)
        canvas = RootBackend.make_canvas(name=f"{signal}/{r.name}/{h.name}")
        canvas.cd()

        draw_opt = "H"
        for label, ratio in ratio_cache.items():
            ratio.Draw(draw_opt)
            legend.AddEntry(ratio, label)
            draw_opt = "H same"

        legend.Draw()
        RootBackend.make_atlas_label()
        canvas.Update()

        hname = h.name.replace("/", "_")
        opath = Path(f"{output}/{r.name}/{hname}.{fmt}")
        opath.parent.mkdir(parents=True, exist_ok=True)
        canvas.SaveAs(str(opath))
