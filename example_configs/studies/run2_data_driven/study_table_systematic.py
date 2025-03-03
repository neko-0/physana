"""
Generate integrated relative systematic with components
"""

import collinearw
from collinearw.strategies import unfolding

import numpy as np

from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich import box as Box

import tabulate

console = Console()

def compuate_integrated_systematics(config, histname="nTruthBJet30", *, show_details=False, filter_large=False):

    config = collinearw.ConfigMgr.open(config)

    lumi = 138.861
    scale_to_xsec = unfolding.plot.scale_to_xsec
    integral_opt = "width all"
    output = {}
    for process in config:
        region_ouptut = {}
        for region in process:
            try:
                hist = region.get_histogram(histname)
            except KeyError:
                continue
            if not hist.systematic_band:
                continue
            if filter_large:
                hist.bin_content[hist.bin_content > 1e4] = 0
            nominal_sum = scale_to_xsec(hist, lumi).integral(integral_opt)
            syst_g = hist.systematic_band
            syst_decomp_up = {}
            syst_decomp_down = {}
            for syst, band in syst_g.items():
                components = band.components
                up = {}
                down = {}
                for name, comp in components["up"].items():
                    if filter_large:
                        comp[comp > 1] = 0
                    c_hist = hist.copy()
                    c_hist.bin_content = hist.bin_content * comp
                    syst_sum = scale_to_xsec(c_hist, lumi).integral(integral_opt)
                    if syst_sum < 1e-5:
                        continue
                    if not isinstance(name, tuple):
                        m_name = name
                    else:
                        m_name = name if show_details else name[0]
                    up[m_name] = (nominal_sum, syst_sum, syst_sum/nominal_sum)
                for name, comp in components["down"].items():
                    if filter_large:
                        comp[comp > 1] = 0
                    c_hist = hist.copy()
                    c_hist.bin_content = hist.bin_content * comp
                    syst_sum = scale_to_xsec(c_hist, lumi).integral(integral_opt)
                    if syst_sum < 1e-5:
                        continue
                    if not isinstance(name, tuple):
                        m_name = name
                    else:
                        m_name = name if show_details else name[0]
                    down[m_name] = (nominal_sum, syst_sum, syst_sum/nominal_sum)
                if up:
                    syst_decomp_up[syst] = up
                if down:
                    syst_decomp_down[syst] = down
            stats = hist.statistical_error(ratio=True)
            if filter_large:
                stats["up"][stats["up"] > 10 ] = 0
                stats["down"][stats["down"] > 10 ] = 0
            c_hist = hist.copy()
            c_hist.bin_content = stats["up"]
            syst_sum = scale_to_xsec(c_hist, lumi).integral(integral_opt)
            syst_decomp_up["stats"] = {"stats": (nominal_sum, syst_sum, syst_sum/nominal_sum)}
            c_hist = hist.copy()
            c_hist.bin_content = stats["down"]
            syst_sum = scale_to_xsec(c_hist, lumi).integral(integral_opt)
            syst_decomp_down["stats"] = {"stats": (nominal_sum, syst_sum, syst_sum/nominal_sum)}
            if syst_decomp_up or syst_decomp_down:
                region_ouptut[region.name] = {"up":syst_decomp_up, "down":syst_decomp_down}
        if region_ouptut:
            output[process.name] = region_ouptut
    return output

def generate_table(syst_dict, show_details=True):

    process_list = [
        "unfold_realthang_fake-EL",
        "unfold_realthang_fake-MU",
        #"wjets_2211",
        #"wjets_FxFx",
        #"wjets", # 2.2.1
        "unfold_realthang_average_comp",
    ]

    region_selector = ["inclusive", "inclusive_2j", "collinear"]
    tables = []
    for process_name, region_dict in syst_dict.items():
        if process_name not in process_list:
            continue
        for region_name, syst in region_dict.items():
            if all([x not in region_name for x in region_selector]):
                continue
            table = Table(
                title=f"Systematic uncertainty decomposition for region {region_name} in {process_name}",
                show_footer=True,
                box=Box.SIMPLE_HEAVY,
                footer_style="bold white",
            )
            table.add_column("Uncert. Name", justify="right", style="green")
            table.add_column("Uncert. Up (fb)", justify="right", style="green")
            table.add_column("Uncert. Down (fb)", justify="right",style="green")
            up_syst = syst["up"]
            down_syst = syst["down"]
            grand_total_up = 0
            grand_total_down = 0
            syst_table = {}
            for syst_name in up_syst:
                total_up = 0
                total_down = 0
                sub_syst_dict = {}
                for sub_syst in (up_syst[syst_name].keys() | down_syst[syst_name].keys()):
                    try:
                        _, up_v, up_r = up_syst[syst_name][sub_syst]
                    except KeyError:
                        up_v, up_r = 0, 0
                    try:
                        _, down_v, down_r = down_syst[syst_name][sub_syst]
                    except KeyError:
                        down_v, down_r = 0, 0
                    sub_syst_dict[sub_syst] = (up_v, down_v)
                    total_up += up_v**2
                    total_down += down_v**2
                syst_table[syst_name] = (np.sqrt(total_up), np.sqrt(total_down), sub_syst_dict)
                grand_total_up += total_up
                grand_total_down += total_down
            grand_total_up = np.sqrt(grand_total_up)
            grand_total_down = np.sqrt(grand_total_down)
            # formating table
            table.add_row(
                "Total (quadrature summed)",
                f"{grand_total_up:.2f}",
                f"{grand_total_down:.2f}",
                end_section=True,
            )
            for syst_name, syst_comps in syst_table.items():
                total_up, total_down, comp_dict = syst_comps
                table.add_row(
                    f"[yellow]{syst_name}",
                    f"[yellow]{total_up:.2f} ({(total_up/grand_total_up)**2:.2%})",
                    f"[yellow]{total_down:.2f} ({(total_down/grand_total_down)**2:.2%})",
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
                        f"[green]{comp}",
                        f"[green]{up_down[0]:.2f} ({(up_down[0]/total_up)**2:.2%})",
                        f"[green]{up_down[1]:.2f} ({(up_down[1]/total_down)**2:.2%})",
                    )
            console.print(table)
            tables.append(table)

    return tables


if __name__ == "__main__":

    output = compuate_integrated_systematics(
        "band_unfold.pkl",
        histname="nTruthJet30",
        show_details=False,
        filter_large=False,
    )
    tables = generate_table(output, show_details=True)

    # with open(output_path, 'w+') as output_file:
    #     print(
    #         tabulate.tabulate(
    #             [*cells, footer], headers=header, tablefmt="latex_raw"
    #         ),
    #         file=output_file,
    #     )
    for table in tables:
        header = list(map(lambda x: x.header, table.columns))
        cells = list(zip(*map(lambda x: list(x.cells), table.columns)))
        footer = list(map(lambda x: x.footer, table.columns))
        print(table.title)
        print(
            tabulate.tabulate(
                [*cells, footer], headers=header, tablefmt="latex_raw"
            )
        )
