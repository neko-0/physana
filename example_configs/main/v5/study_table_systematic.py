import collinearw
from collinearw.strategies import unfolding

import numpy as np

from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich import box as Box

import tabulate

console = Console()


def _filtering(name, include=None, exclude=None):
    if include is None and exclude is None:
        return False
    if exclude and name in exclude:
        return True
    if include and name not in include:
        return True
    return False


def compuate_integrated_systematics(
    config,
    histname="nTruthBJet30",
    *,
    include_processes=None,
    exclude_processes=None,
    include_regions=None,
    exclude_regions=None,
    exclude_syst=None,
    show_details=False,
    filter_low=0,
    filter_high=1e9,
    integrate=True,
):
    """
    Computing integrated systematic for a given histogram.

    param:
        config : str, collinearw.core.ConfigMgr
            config file

        histname : str, default='nTruthJet30'
            name of histogram being used for integrated systematics

        show_details: bool, default=False
            decompose further for each of the systematics

        filter_low: float, default=0
            bin filtering on low values

        filter_high: float, default=1e9
            bin filtering on high values
    """

    config = collinearw.ConfigMgr.open(config)

    include_processes = [] if include_processes is None else include_processes
    exclude_processes = [] if exclude_processes is None else exclude_processes
    include_regions = [] if include_regions is None else include_regions
    exclude_regions = [] if exclude_regions is None else exclude_regions
    exclude_syst = [] if exclude_syst is None else exclude_syst

    syst_sum_filter = 1e-5
    lumi = 138.861
    scale_to_xsec = unfolding.plot.scale_to_xsec
    integral_opt = "width all"
    output = {}
    for process in config:
        # filtering processes
        if _filtering(process.name, include_processes, exclude_processes):
            continue
        region_ouptut = {}
        for region in process:
            # filtering processes
            if _filtering(region.name, include_regions, exclude_regions):
                continue
            try:
                hist = region.get_histogram(histname)
            except KeyError:
                continue
            if not hist.systematic_band:
                continue
            # filtering bin content
            filter_low_mask = hist.bin_content <= filter_low
            filter_high_mask = hist.bin_content >= filter_high
            hist.bin_content[filter_low_mask | filter_high_mask] = 0
            hist.bin_content[filter_low_mask | filter_high_mask] = 0
            hist.sumW2[filter_low_mask | filter_high_mask] = 0
            hist.sumW2[filter_low_mask | filter_high_mask] = 0
            # scale histogram to xsec
            if integrate:
                nominal_sum = scale_to_xsec(hist, lumi).integral(integral_opt)
            else:
                nominal_sum = scale_to_xsec(hist, lumi).bin_content[1:]
            syst_band = hist.systematic_band
            syst_decomp_up = {}
            syst_decomp_down = {}
            """
            syst_decomp_* has format as
            {
                'syst band name' : {
                    'component name' : (nominal_sum, syst_sum, frac),
                }
            }
            """
            for syst_name, band in syst_band.items():
                if syst_name in exclude_syst:
                    continue
                components = band.components
                up = {}
                down = {}
                # loop through all up band
                for name, comp in components["up"].items():
                    c_hist = hist.copy()
                    c_hist.bin_content = hist.bin_content * comp
                    if integrate:
                        syst_sum = scale_to_xsec(c_hist, lumi).integral(integral_opt)
                    else:
                        syst_sum = scale_to_xsec(c_hist, lumi).bin_content[1:]
                    if (syst_sum < syst_sum_filter).all():
                        continue
                    if not isinstance(name, tuple):
                        m_name = name
                    else:
                        m_name = name if show_details else name[0]
                    frac = np.nan_to_num(syst_sum / nominal_sum)
                    up[m_name] = (nominal_sum, syst_sum, frac)
                # loop through all down band
                for name, comp in components["down"].items():
                    c_hist = hist.copy()
                    c_hist.bin_content = hist.bin_content * comp
                    if integrate:
                        syst_sum = scale_to_xsec(c_hist, lumi).integral(integral_opt)
                    else:
                        syst_sum = scale_to_xsec(c_hist, lumi).bin_content[1:]
                    if (syst_sum < syst_sum_filter).all():
                        continue
                    if not isinstance(name, tuple):
                        m_name = name
                    else:
                        m_name = name if show_details else name[0]
                    frac = np.nan_to_num(syst_sum / nominal_sum)
                    down[m_name] = (nominal_sum, syst_sum, frac)
                if up:
                    syst_decomp_up[syst_name] = up
                if down:
                    syst_decomp_down[syst_name] = down
            # compute statistical band. re-asign bin_content and use scale_to_xsec
            stats = hist.statistical_error(ratio=True)
            c_hist = hist.copy()
            c_hist.bin_content *= stats["up"]
            if integrate:
                syst_sum = scale_to_xsec(c_hist, lumi).integral(integral_opt)
            else:
                syst_sum = scale_to_xsec(c_hist, lumi).bin_content[1:]
            frac = np.nan_to_num(syst_sum / nominal_sum)
            syst_decomp_up["stats"] = {"stats": (nominal_sum, syst_sum, frac)}
            c_hist = hist.copy()
            c_hist.bin_content *= stats["down"]
            if integrate:
                syst_sum = scale_to_xsec(c_hist, lumi).integral(integral_opt)
            else:
                syst_sum = scale_to_xsec(c_hist, lumi).bin_content[1:]
            frac = syst_sum / nominal_sum
            syst_decomp_down["stats"] = {"stats": (nominal_sum, syst_sum, frac)}
            region_ouptut[region.name] = {
                "up": syst_decomp_up,
                "down": syst_decomp_down,
            }
        if region_ouptut:
            output[process.name] = region_ouptut
    return output


def generate_integrated_table(
    syst_dict, bin_edges=None, show_details=True, show_percentage=True
):

    tables = []
    for process_name, region_dict in syst_dict.items():
        for region_name, syst in region_dict.items():
            table = Table(
                title=f"Systematic uncertainty decomposition for region {region_name} in {process_name}",
                show_footer=True,
                box=Box.SIMPLE_HEAVY,
                footer_style="bold white",
            )
            table.add_column("Uncert. Name", justify="right", style="green")
            if bin_edges is None:
                table.add_column("Uncert. Up (fb)", justify="right", style="green")
                table.add_column("Uncert. Down (fb)", justify="right", style="green")
            else:
                for x in bin_edges:
                    table.add_column(f"[{x:.1f}]", justify="right", style="green")
            up_syst = syst["up"]
            down_syst = syst["down"]
            total_xe = None
            grand_total_up = 0
            grand_total_down = 0
            stats_total_up = None
            stats_total_down = None
            syst_table = {}
            for syst_name in up_syst:
                total_up = 0
                total_down = 0
                sub_syst_dict = {}
                for sub_syst in up_syst[syst_name].keys() | down_syst[syst_name].keys():
                    try:
                        xe, up_v, up_r = up_syst[syst_name][sub_syst]
                    except KeyError:
                        if bin_edges is not None:
                            up_v = np.zeros(len(bin_edges))
                            up_r = np.zeros(len(bin_edges))
                        else:
                            up_v, up_r = 0, 0
                    try:
                        xe, down_v, down_r = down_syst[syst_name][sub_syst]
                    except KeyError:
                        if bin_edges is not None:
                            down_v = np.zeros(len(bin_edges))
                            down_r = np.zeros(len(bin_edges))
                        else:
                            down_v, down_r = 0, 0
                    sub_syst_dict[sub_syst] = (up_v, down_v)
                    total_up += up_v**2
                    total_down += down_v**2
                    if total_xe is None:
                        total_xe = xe
                    assert (total_xe == xe).all()
                    if syst_name == "stats":
                        stats_total_up = up_v
                        stats_total_down = down_v
                syst_table[syst_name] = (
                    np.sqrt(total_up),
                    np.sqrt(total_down),
                    sub_syst_dict,
                )
                grand_total_up += total_up
                grand_total_down += total_down
            syst_total_up = np.sqrt(grand_total_up - stats_total_up**2)
            syst_total_down = np.sqrt(grand_total_down - stats_total_down**2)
            grand_total_up = np.sqrt(grand_total_up)
            grand_total_down = np.sqrt(grand_total_down)
            # formating table
            percent = "({0:.2%})" if show_percentage else ""
            if bin_edges is None:
                table.add_row(
                    f"Total XE: {total_xe:.2f}",
                    "",
                    "",
                    end_section=True,
                )
                table.add_row(
                    "Total Uncertainty",
                    f"{grand_total_up:.2f}",
                    f"{grand_total_down:.2f}",
                    end_section=True,
                )
                table.add_row(
                    "Total Stats.",
                    f"{stats_total_up:.2f}",
                    f"{stats_total_down:.2f}",
                    end_section=True,
                )
                table.add_row(
                    "Total Syst.",
                    f"{syst_total_up:.2f}",
                    f"{syst_total_down:.2f}",
                    end_section=True,
                )
                for syst_name, syst_comps in syst_table.items():
                    total_up, total_down, comp_dict = syst_comps
                    table.add_row(
                        f"[yellow]{syst_name}",
                        f"[yellow]{total_up:.2f} {percent.format((total_up/grand_total_up)**2)}",
                        f"[yellow]{total_down:.2f} {percent.format((total_down/grand_total_down)**2)}",
                        end_section=True,
                    )
                    if not show_details:
                        continue
                    try:
                        sorted_dict = sorted(comp_dict.items())
                    except TypeError:
                        sorted_dict = comp_dict.items()
                    for comp, up_down in sorted_dict:
                        table.add_row(
                            f"[green]\\qquad {comp}",
                            f"[green]{up_down[0]:.2f} {percent.format((up_down[0]/total_up)**2)}",
                            f"[green]{up_down[1]:.2f} {percent.format((up_down[1]/total_down)**2)}",
                        )
            else:
                table.add_row(
                    "Total XE",
                    *[f"{x:.2f}" for x in total_xe],
                    end_section=True,
                )
                table.add_row(
                    "Total Uncertainty",
                    *[
                        f"{x:.2f}, {y:.2f}"
                        for x, y in zip(grand_total_up, grand_total_down)
                    ],
                    end_section=True,
                )
                table.add_row(
                    "Total Stats",
                    *[
                        f"{x:.2f}, {y:.2f}"
                        for x, y in zip(stats_total_up, stats_total_down)
                    ],
                    end_section=True,
                )
                table.add_row(
                    "Total Syts",
                    *[
                        f"{x:.2f}, {y:.2f}"
                        for x, y in zip(syst_total_up, syst_total_down)
                    ],
                    end_section=True,
                )
                for syst_name, syst_comps in syst_table.items():
                    total_up, total_down, comp_dict = syst_comps
                    table.add_row(
                        f"[yellow]{syst_name}",
                        *[
                            f"{x:.2f}{percent.format((y/tot_y)**2)}, {y:.2f}{percent.format((x/tot_x)**2)}"
                            for x, y, tot_x, tot_y in zip(
                                total_up, total_down, grand_total_up, grand_total_down
                            )
                        ],
                        end_section=True,
                    )
                    if not show_details:
                        continue
                    try:
                        sorted_dict = sorted(comp_dict.items())
                    except TypeError:
                        sorted_dict = comp_dict.items()
                    for comp, up_down in sorted_dict:
                        table.add_row(
                            f"[green]\\qquad {comp}",
                            *[
                                f"{x:.2f}{percent.format((y/tot_u)**2)}, {y:.2f}{percent.format((x/tot_d)**2)}"
                                for x, y, tot_d, tot_u in zip(
                                    up_down[0], up_down[1], total_up, total_down
                                )
                            ],
                        )

            console.print(table)
            tables.append(table)

    return tables


def run(
    config,
    include_processes,
    include_regions,
    exclude_syst,
    obs,
    integrate,
    write_to_file=False,
    output_path="syst_table.tex",
):

    output = compuate_integrated_systematics(
        config,
        histname=obs,
        include_processes=include_processes,
        include_regions=include_regions,
        exclude_syst=exclude_syst,
        show_details=False,
        integrate=integrate,
    )
    if integrate:
        tables = generate_integrated_table(output, show_details=True)
    else:
        edges = config[0][0].get_histogram(obs).bins
        tables = generate_integrated_table(
            output, bin_edges=edges, show_details=True, show_percentage=False
        )

    for table in tables:
        header = list(map(lambda x: x.header, table.columns))
        cells = list(zip(*map(lambda x: list(x.cells), table.columns)))
        footer = list(map(lambda x: x.footer, table.columns))
        # clean up format
        clean_cells = []
        for i, cell in enumerate(cells):
            if integrate:
                clean_cells.append(
                    [
                        x.replace("[green]", "")
                        .replace("[yellow]", "")
                        .replace("%", "\%")
                        .replace("_", "\_")
                        for x in cell
                    ]
                )
            else:
                # need to reformat row
                _cache = []
                for j, x in enumerate(cell):
                    new_x = (
                        x.replace("[green]", "")
                        .replace("[yellow]", "")
                        .replace("%", "\%")
                        .replace("_", "\_")
                    )
                    if i > 0 and j > 0:
                        new_x = new_x.split(",")
                        base_name = "$\\substack{{+{0} \\\ -{1} }}$"
                        new_x = base_name.format(new_x[1], new_x[0])
                    _cache.append(new_x)
                clean_cells.append(_cache)

        if not write_to_file:
            print(table.title)
            print(
                tabulate.tabulate(
                    [*clean_cells, footer], headers=header, tablefmt="latex_raw"
                )
            )
        else:
            with open(output_path, "w+") as output_file:
                print(table.title, output_file)
                print(
                    tabulate.tabulate(
                        [*clean_cells, footer], headers=header, tablefmt="latex_raw"
                    ),
                    file=output_file,
                )


if __name__ == "__main__":

    exclude_syst = [
        "diboson_powheg",
        "diboson",
        "unfold_wjets",
        "unfold_wjets_2211_ASSEW",
        "wjets-fxfx",
    ]

    observables = {
        "nTruthJet30": "njet",
        "wTruthPt": "wpt",
        "HtTruth30": "ht",
        "jet1TruthPt": "jetpt",
    }

    settings = []
    # differential
    for obs, tag in observables.items():
        for phasespace in ["collinear"]:
            el = {
                "include_processes": ["unfold_realthang_fake-EL"],
                "include_regions": [f"electron_{phasespace}_truth"],
                "exclude_syst": exclude_syst,
                "obs": obs,
                "integrate": False,
                "write_to_file": True,
                "output_path": f"measured_{phasespace}_el_{tag}.tex",
            }
            mu = {
                "include_processes": ["unfold_realthang_fake-MU"],
                "include_regions": [f"muon_{phasespace}_truth"],
                "exclude_syst": exclude_syst,
                "obs": obs,
                "integrate": False,
                "write_to_file": True,
                "output_path": f"measured_{phasespace}_mu_{tag}.tex",
            }
            settings.append(el)
            settings.append(mu)

    # dR only
    for phasespace in ["inclusive"]:
        el = {
            "include_processes": ["unfold_realthang_fake-EL"],
            "include_regions": [f"electron_{phasespace}_truth"],
            "exclude_syst": exclude_syst,
            "obs": "DeltaRTruthLepJetClosest100",
            "integrate": False,
            "write_to_file": True,
            "output_path": f"measured_{phasespace}_el_dR.tex",
        }
        mu = {
            "include_processes": ["unfold_realthang_fake-MU"],
            "include_regions": [f"muon_{phasespace}_truth"],
            "exclude_syst": exclude_syst,
            "obs": "DeltaRTruthLepJetClosest100",
            "integrate": False,
            "write_to_file": True,
            "output_path": f"measured_{phasespace}_mu_dR.tex",
        }
        settings.append(el)
        settings.append(mu)

    # mjj only
    for phasespace in ["inclusive_2j"]:
        el = {
            "include_processes": ["unfold_realthang_fake-EL"],
            "include_regions": [f"electron_{phasespace}_truth"],
            "exclude_syst": exclude_syst,
            "obs": "mjjTruth",
            "integrate": False,
            "write_to_file": True,
            "output_path": f"measured_{phasespace}_el_mjj.tex",
        }
        mu = {
            "include_processes": ["unfold_realthang_fake-MU"],
            "include_regions": [f"muon_{phasespace}_truth"],
            "exclude_syst": exclude_syst,
            "obs": "mjjTruth",
            "integrate": False,
            "write_to_file": True,
            "output_path": f"measured_{phasespace}_mu_mjj.tex",
        }
        settings.append(el)
        settings.append(mu)

    # integrated
    for phasespace in ["inclusive", "collinear"]:
        el = {
            "include_processes": ["unfold_realthang_fake-EL"],
            "include_regions": [f"electron_{phasespace}_truth"],
            "exclude_syst": exclude_syst,
            "obs": "nTruthBJet30",
            "integrate": True,
            "write_to_file": True,
            "output_path": f"measured_{phasespace}_el_syst.tex",
        }
        mu = {
            "include_processes": ["unfold_realthang_fake-MU"],
            "include_regions": [f"muon_{phasespace}_truth"],
            "exclude_syst": exclude_syst,
            "obs": "nTruthBJet30",
            "integrate": True,
            "write_to_file": True,
            "output_path": f"measured_{phasespace}_mu_syst.tex",
        }
        settings.append(el)
        settings.append(mu)

    config = collinearw.ConfigMgr.open("band_unfold_syst_added.pkl")

    for setting in settings:
        run(config, **setting)
