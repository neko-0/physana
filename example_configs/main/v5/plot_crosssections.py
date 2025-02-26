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

def run(flavor, process_id, signal_filter=[]):

    print(f"Running plotting for flavor {flavor} and process {process_id}")

    configMgr = ConfigMgr.open(
        '/gpfs/slac/atlas/fs1/d/yuzhan/configs_bk/ColWJet_2022/v5i/run2_MC_driven/band_unfold_syst_added_BLUE_added.pkl'
    )

    lumi = 138.861
    configMgr.out_path = pathlib.Path("./figures")

    # Additional theory predictions to be overlaid
    signals = [
        procset
        for procset in configMgr.process_sets
        if procset.process_type in ["signal", "signal_alt"] and procset.name in signal_filter
    ]

    # Main data measurement
    unfoldeds = [
        procset
        for procset in configMgr.process_sets
        if procset.process_type == 'unfolded' and  procset.name.endswith(process_id)
    ]

    print("Including the following data names...")
    for unfolded in unfoldeds:
      print(f" ----> {unfolded.name} ")

    print("Including the following signal names...")
    for signal in signals:
      print(f" ----> {signal.name} ")

    region_cats = ['inclusive', 'inclusive_2j', 'collinear' ] # 'backtoback']

    style = json.load(open(f'{flavor}style.json'))

    title_map = {  "W+jets (Sh 2.2.11)": "Sherpa 2.2.11 NLO QCD",
                   "W+jets (FxFx)":"MG_aMC@NLO+Pythia8 FxFx",
                   "W+jets (Sh 2.2.1)":"Sherpa 2.2.1 NLO QCD",
                   "W+jets 2.2.11 (ASSEW)":"Sherpa 2.2.11 NLO QCD+EW_{virt}",
    }

    for data in unfoldeds:
        for region_cat in region_cats:

            region_name = f'{flavor}{region_cat}_truth'

            region = data.nominal.get_region(region_name)

            for observable in data.nominal.get_region(
                f'{flavor}{region_cat}_truth'
            ):
                base_output_path = (
                    pathlib.Path(configMgr.out_path)
                    .joinpath(
                        "unfolding_updated_plots",
                        "unfolded",
                        data.name,
                        region_name,
                        observable.name,
                    )
                    .resolve()
                )

                output_path = base_output_path.with_suffix('.pdf')

                others = []
                for signal in signals:
                  if signal.title=='W+jets (Sh 2.2.1)':
                    continue 

                  if signal.title=='W+jets (Sh 2.2.11)':
                    signal.nominal.title = "Sherpa 2.2.11 NLO QCD"
                    signal.nominal.markerstyle = 25
                    signal.nominal.color = 9
                  if signal.title=='W+jets (FxFx)':
                    signal.nominal.title = "MG_aMC@NLO+Pythia8 FxFx"
                    signal.nominal.markerstyle = 24
                    signal.nominal.color = 30
                  if signal.title=='W+jets (Sh 2.2.1)':
                    signal.nominal.title = "Sherpa 2.2.1 NLO QCD"
                    signal.nominal.color = 46
                    signal.nominal.markerstyle = 27


                  others.append(signal.nominal.get_region(region_name))


                # template = {"log":True,"minMain":1e-3,"maxMain":1e3,"minRatio":0,"maxRatio":2}
                # style[f'{region_name}_{observable.name}'] = template

                unfolding.plot.unpack_results(
                  f'{flavor}{region_cat}_truth',
                  observable.name,
                  region,
                  others,
                  output_path,
                  138.861,
                  0.02,
                  layout=[[0,1]],
                  style=style,
                  title_map=title_map,
                )


"""
2211_average_BLUE
 ----> wjets_FxFx_average_BLUE
 ----> wjets_average_BLUE
"""

#run("muon_", "MU",signal_filter=['wjets_2211','wjets_FxFx','wjets','wjets_2211_ASSEW'])
#run("electron_", "EL",signal_filter=['wjets_2211','wjets_FxFx','wjets','wjets_2211_ASSEW'])
run("", "average_BLUE",signal_filter=['wjets_2211_average_BLUE','wjets_FxFx_average_BLUE','wjets_average_BLUE'])
