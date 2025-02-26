from collinearw import ConfigMgr, core, utils
from collinearw.strategies import unfolding
from collinearw.backends import RootBackend

import ROOT
import pathlib
import logging

logging.basicConfig()
logging.getLogger().setLevel(logging.CRITICAL)

log = logging.getLogger(__name__)

configMgr = ConfigMgr.open(
    '/gpfs/slac/atlas/fs1/d/yuzhan/configs_bk/ColWJet_2022/v5/run2_nominal_noVgamma_tauBkg_updateMuon/unfold_nominal.pkl'
)

lumi = 138.861
configMgr.out_path = pathlib.Path(
    '/sdf/home/g/gstark/collinearw/output/unfolding_v5_nominal_noVgamma_tauBkg_updateMuon'
)

signal = configMgr.get_process_set("wjets_2211")

regions = unfolding.metadata.regions(configMgr)
observables = unfolding.metadata.observables(configMgr)

for unfold_name, region in regions.items():
    for observable, histogram in observables.items():
        base_output_path = (
            pathlib.Path(configMgr.out_path)
            .joinpath(
                "plots",
                "response",
                region["reco_match"],
                unfold_name,
                observable,
            )
            .resolve()
        )
        output_path = base_output_path.with_suffix('.pdf')

        hResponse = (
            signal.nominal.get_region(region["reco_match"])
            .get_histogram(histogram['response'])
            .root
        )
        RootBackend.apply_process_styles(hResponse, signal.nominal)

        unfolding.plot.plot_response(hResponse, output_path=output_path)

# ratio
for unfold_name, region in regions.items():
    if not 'electron' in unfold_name:
        continue

    for observable, histogram in observables.items():
        base_output_path = (
            pathlib.Path(configMgr.out_path)
            .joinpath(
                "plots",
                "response_ratio",
                region["reco_match"].replace('electron', 'lepton'),
                unfold_name.replace('electron', 'lepton'),
                observable,
            )
            .resolve()
        )
        output_path = base_output_path.with_suffix('.pdf')

        hResponse_el = (
            signal.nominal.get_region(region["reco_match"])
            .get_histogram(histogram['response'])
            .root
        )
        hResponse_mu = (
            signal.nominal.get_region(region["reco_match"].replace('electron', 'muon'))
            .get_histogram(histogram['response'])
            .root
        )

        hResponse_el.Divide(hResponse_mu)

        RootBackend.apply_process_styles(hResponse_el, signal.nominal)

        unfolding.plot.plot_response(
            hResponse_el, output_path=output_path, logz=False, add_blue=True
        )
