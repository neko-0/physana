from collinearw import ConfigMgr, backends, utils
from collinearw.strategies import unfolding
from collinearw.backends import RootBackend
from collinearw.strategies import systematics as sys_handler
import numpy as np

import ROOT
import pathlib
import logging
import json

logging.basicConfig()
logging.getLogger().setLevel(logging.CRITICAL)

configMgr = ConfigMgr.open(
    '/gpfs/slac/atlas/fs1/d/yuzhan/configs_bk/ColWJet_2022/v5i/run2_MC_driven/band_unfold_syst_added_BLUE_added.pkl'
)

lumi = 138.861
configMgr.out_path = pathlib.Path("./figures")

signals = [
    procset
    for procset in configMgr.process_sets
    if procset.name.endswith("-EL") or procset.name.endswith("-MU")
]

print(signals)

title_map = {
    "W+jets (Sh 2.2.11)": "Sherpa 2.2.11 NLO QCD",
    "W+jets (FxFx)": "MG_aMC@NLO+Pythia8 FxFx",
    "W+jets (Sh 2.2.1)": "Sherpa 2.2.1 NLO QCD",
    "W+jets 2.2.11 (ASSEW)": "Sherpa 2.2.11 NLO QCD+EW_{virt}",
}

data_processes = [
    procset for procset in configMgr.process_sets if 'average_BLUE' in procset.name
]


region_cats = ['inclusive', 'inclusive_2j', 'collinear', 'backtoback']

style = json.load(open('style_average.json'))

exclude_names = ["diboson", "diboson_powheg", "unfold_wjets", "unfold_wjets_2211_ASSEW"]

for data in data_processes:
    for region_cat in region_cats:

        region_name = f'{region_cat}_truth'

        for observable in data.nominal.get_region(f'{region_cat}_truth'):

            region = data.nominal.get_region(f'{region_cat}_truth')

            base_output_path = (
                pathlib.Path(configMgr.out_path)
                .joinpath(
                    "unfolding_updated_plots",
                    "unfolded",
                    data.name,
                    f'averaged_{region_cat}_truth',
                    observable.name,
                )
                .resolve()
            )
            output_path = base_output_path.with_suffix('.pdf')

            # Electron and muon channels
            others = []
            for signal in signals:
                if 'EL' in signal.name:
                    signal.nominal.title = "Electron channel"
                    signal.nominal.color = 4
                    signal.nominal.markerstyle = 25
                    others.append(
                        signal.nominal.get_region(f'electron_{region_cat}_truth')
                    )
                if 'MU' in signal.name:
                    signal.nominal.title = "Muon channel"
                    signal.nominal.color = 8
                    signal.nominal.markerstyle = 4
                    others.append(signal.nominal.get_region(f'muon_{region_cat}_truth'))

            unfolding.plot.unpack_results(
                f'{region_cat}_truth',
                observable.name,
                region,  # Main process
                others,
                output_path,
                138.861,
                0.02,
                layout=[[0, 1]],
                style=style,
                title_map=title_map,
                exclude_names=exclude_names,
            )
