from collinearw import ConfigMgr, core
from collinearw.strategies import unfolding
import numpy as np

from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich import box as Box

import pathlib
import logging

logging.basicConfig()
logging.getLogger().setLevel(logging.CRITICAL)

configMgr = ConfigMgr.open(
    #    '/gpfs/slac/atlas/fs1/d/yuzhan/configs_bk/slac_configs/V4_production_ver7/sherpa2211_run2_tight/band_unfoled_v2.pkl'
    'output/unfolding_v4_Aug2021Talk_nominal/unfold.pkl'
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

console = Console()


def get_xsec_uncert(observable, lumi):
    if not isinstance(observable, core.Histogram):
        return (None, None)

    observable = unfolding.plot.scale_to_xsec(observable, lumi)

    errors = None
    if observable.systematic_band is not None:
        error_dn = np.sqrt(
            observable.scale_band('experimental')['down'][1:-1] ** 2
            + observable.scale_band('theory')['down'][1:-1] ** 2
        )
        error_up = np.sqrt(
            observable.scale_band('experimental')['up'][1:-1] ** 2
            + observable.scale_band('theory')['up'][1:-1] ** 2
        )
        errors = (np.sum(error_dn), np.sum(error_up))
    return (observable.integral('width'), errors)


for unfolded in unfoldeds:
    for region in unfolded.nominal:

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
            f"{unfolding.plot.scale_to_xsec(region.get_histogram(inclusiveXSec), lumi).integral('width'):.2f}",
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
                f"{unfolding.plot.scale_to_xsec(_inclusive_hist, lumi).integral('width'):.2f}",
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

            xsec, uncert = get_xsec_uncert(observable, lumi)
            if not xsec:
                continue

            signal_xsecs = [
                unfolding.plot.scale_to_xsec(
                    signal.nominal.get_region(region.name).get_histogram(
                        observable.name
                    ),
                    lumi,
                ).integral('width')
                for signal in signals
            ]

            xsecs = list(map(lambda x: f"{x:.2f}", [xsec, *signal_xsecs]))

            if uncert:
                table.add_row(
                    observable.name,
                    *xsecs,
                    f"{uncert[0]:.2f}",
                    f"{uncert[1]:.2f}",
                )
            else:
                table.add_row(observable.name, *xsecs, "", "")
        console.print(table)
