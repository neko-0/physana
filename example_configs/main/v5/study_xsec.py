from collinearw import ConfigMgr, core
from collinearw.strategies import unfolding
import numpy as np

from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich import box as Box

import tabulate

import pathlib
import logging

logging.basicConfig()
logging.getLogger().setLevel(logging.CRITICAL)

configMgr = ConfigMgr.open(
    #    '/gpfs/slac/atlas/fs1/d/yuzhan/configs_bk/slac_configs/V4_production_ver7/sherpa2211_run2_tight/band_unfoled_v2.pkl'
    #'output/unfolding_v4_Aug2021Talk_nominal/unfold.pkl'
    #'output/unfolding_v3_crackTest/unfold.pkl'
    '/gpfs/slac/atlas/fs1/d/yuzhan/configs_bk/slac_configs/Giordon_Aug2021_Talks/V5_production/track_calo_with_more_sys_2211_run2/dR_truth_selection/debug_unfold.pkl',
)

# lumi = 36.1  # ifb
lumi = 138.861
# configMgr.out_path = pathlib.Path('/sdf/home/g/gstark/collinearw/output/unfolding_v4')

signals = [
    procset
    for procset in configMgr.process_sets
    if procset.process_type in ["signal", "signal_alt"]
]

unfoldeds = [
    procset
    for procset in configMgr.process_sets
    if procset.process_type in ["unfolded"]
]

inclusiveXSec = "nTruthBJet30"

integral_opts = 'width all'

console = Console()

for unfolded in unfoldeds:
    for region in unfolded.nominal:

        base_output_path = (
            pathlib.Path(configMgr.out_path)
            .joinpath(
                "tables",
                "study_xsec",
                unfolded.name,
                region.name,
            )
            .resolve()
        )
        output_path = base_output_path.with_suffix('.tex')
        output_path.parent.mkdir(parents=True, exist_ok=True)

        table = Table(
            title=f"Total Cross-Section for region {region.name} in {unfolded.name}",
            show_footer=True,
            box=Box.SIMPLE_HEAVY,
            footer_style="bold white",
        )
        table.add_column(
            "Observable",
            Text.from_markup("Inclusive", justify="right"),
            justify="left",
            style="cyan",
            no_wrap=True,
        )
        table.add_column(
            "Cross-section (fb)",
            f"{unfolding.plot.scale_to_xsec(region.get_histogram(inclusiveXSec), lumi).integral(integral_opts):.2f}",
            justify="right",
            style="bright_red",
            header_style="bright_red",
        )
        for signal in signals:
            _inclusive_hist = signal.nominal.get_region(region.name).get_histogram(
                inclusiveXSec
            )
            table.add_column(
                f"{signal.nominal.title} XS (fb)",
                f"{unfolding.plot.scale_to_xsec(_inclusive_hist, lumi).integral(integral_opts):.2f}",
                justify="right",
                style="magenta",
                header_style="magenta",
            )
        table.add_column("Uncert. Up (fb)", justify="right", style="green")
        table.add_column(
            "Uncert. Down (fb)",
            justify="right",
            style="red",
            header_style="red",
        )

        for observable in region:
            if observable.name == inclusiveXSec:
                continue

            xsec, stat_error, syst_error = unfolding.utils.get_xsec_uncert(
                observable, lumi
            )
            if not xsec:
                continue

            signal_xsecs = [
                unfolding.utils.get_xsec_uncert(
                    signal.nominal.get_region(region.name).get_histogram(
                        observable.name
                    ),
                    lumi,
                    integral_opts=integral_opts,
                )
                for signal in signals
            ]

            xsecs = list(map(lambda x: f"{x:.2f}", [xsec, *signal_xsecs]))

            if syst_error:
                table.add_row(
                    observable.name,
                    *xsecs,
                    f"{syst_error[0]:.2f}",
                    f"{syst_error[1]:.2f}",
                )
            else:
                table.add_row(observable.name, *xsecs, "", "")

        console.print(table)

        header = list(map(lambda x: x.header, table.columns))
        cells = list(zip(*map(lambda x: list(x.cells), table.columns)))
        footer = list(map(lambda x: x.footer, table.columns))

        with open(output_path, 'w+') as output_file:
            print(
                tabulate.tabulate(
                    [*cells, footer], headers=header, tablefmt="latex_raw"
                ),
                file=output_file,
            )
