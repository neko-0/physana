from collinearw import ConfigMgr
from collinearw.strategies import unfolding
import numpy as np

import pathlib
import logging

from rich.tree import Tree
from rich import print

logging.basicConfig()
logging.getLogger().setLevel(logging.CRITICAL)

configMgr = ConfigMgr.open(
    '/gpfs/slac/atlas/fs1/d/yuzhan/configs_bk/ColWJet_2022/v5i/run2_MC_driven/ave_band_unfold.pkl'
)

lumi = 138.861
configMgr.out_path = pathlib.Path(
    './output/v5i_run2_MC_driven_average'
)


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

inclusive_observable_name = "nTruthBJet30"

for unfolded in unfoldeds:
    tree = Tree(unfolded.name, style="bold bright_red", guide_style="bold bright_blue")

    for region in unfolded.nominal:
        branch_region = tree.add(
            region.name, style="bold bright_blue", guide_style="bold bright_green"
        )

        base_output_path = (
            pathlib.Path(configMgr.out_path)
            .joinpath(
                "tables",
                "study_xsec_total",
                unfolded.name,
                region.name,
            )
            .resolve()
        )
        output_path = base_output_path.with_suffix('.tex')
        output_path.parent.mkdir(parents=True, exist_ok=True)

        metadata = {}

        for process, name in zip(
            [unfolded] + signals, ['data'] + [signal.name for signal in signals]
        ):
            branch_process = branch_region.add(
                name, style="bold bright_green", guide_style="white"
            )
            metadata[name] = {'inclusive': None, 'observables': []}

            try:
                process_region = process.nominal.get_region(region.name)
            except KeyError:
                try:
                    process_region = process.nominal.get_region(region.name.replace('electron_','').replace('muon_',''))
                except KeyError:
                    continue

            for observable in process_region:
                if observable.name.endswith('_DOWN'):
                    continue

                xsec, stat_error, syst_error = unfolding.utils.get_xsec_uncert(
                    observable, lumi
                )
                syst_error = syst_error or (0.0, 0.0)

                if inclusive_observable_name in observable.name:
                    metadata[name]['inclusive'] = (xsec, stat_error, *syst_error)
                    continue
                metadata[name]['observables'].append((xsec, stat_error, *syst_error))

            inclusive = np.array(metadata[name]['inclusive'])
            observables = np.array(metadata[name]['observables'])

            if metadata[name]['inclusive'] is None:
                breakpoint()
            branch_process.add(
                f'    [white]{inclusive[0]:0.4f}[/white] ± [bright yellow]{inclusive[1]:0.4f}[/bright yellow] ± [yellow]{inclusive[2]:0.4f},{inclusive[3]:0.4f}[/yellow] [grey84 dim](inclusive, inclusive uncrt)[/grey84 dim]'
            )
            branch_process.add(
                f'    [white]{inclusive[0]:0.4f}[/white] ± [bright yellow]{inclusive[1]:0.4f}[/bright yellow] ± [yellow]{np.std(observables[:, 0]):0.4f}[/yellow] [grey84 dim](inclusive, observables std. dev.)[/grey84 dim]'
            )
            branch_process.add(
                f'    [white]{np.mean(observables[:, 0]):0.4f}[/white] ± [bright yellow]{np.mean(observables[:, 1]):0.4f}[/bright yellow] ± [yellow]{np.sqrt(np.sum(np.square(observables[:, 2]))):0.4f},{np.sqrt(np.sum(np.square(observables[:, 3]))):0.4f}[/yellow] [grey84 dim](observables average, observables quadrature)[/grey84 dim]'
            )
            branch_process.add(
                f'    [white]{inclusive[0]:0.4f}[/white] ± [bright yellow]{inclusive[1]:0.4f}[/bright yellow] + [yellow]{np.max(observables[:,0])-inclusive[0]:0.4f}[/yellow] - [yellow]{inclusive[0] - np.min(observables[:,0]):0.4f}[/yellow] [grey84 dim](inclusive bin, observables min/max)[/grey84 dim]'
            )
            branch_process.add(
                f'    [white]{np.mean(observables[:,0]):0.4f}[/white] ± [bright yellow]{np.mean(observables[:, 1]):0.4f}[/bright yellow] + [yellow]{np.max(observables[:,0])-np.mean(observables[:,0]):0.4f}[/yellow] - [yellow]{np.mean(observables[:,0]) - np.min(observables[:,0]):0.4f}[/yellow] [grey84 dim](observables average, observables min/max)[/grey84 dim]'
            )

    print(tree)
