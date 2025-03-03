import collinearw
import pathlib
from collinearw.strategies import unfolding
import numpy as np
import json
import pyhf

config = collinearw.ConfigMgr.open(
    '/gpfs/slac/atlas/fs1/d/yuzhan/configs_bk/ColWJet_2022/v5i/run2_MC_driven/band_unfold_syst_added_ew_added_BLUE_added_May17.pkl'
)
config.out_path = pathlib.Path(
    '/sdf/home/g/gstark/collinearw/output/v5i_ewAdded_BLUEadded_May17'
)

lumi = 138.861
lumiunc = 0.02

processes = [
    ("unfold_realthang", "unfold_realthang_fake-EL", "unfold_realthang_fake-MU"),
    ("wjets_2211", "wjets_2211", "wjets_2211"),
    ("wjets_FxFx", "wjets_FxFx", "wjets_FxFx"),
    ("wjets", "wjets", "wjets"),
    ("wjets_2211_ASSEW", "wjets_2211_ASSEW", "wjets_2211_ASSEW"),
]


phasespaces = [
    "inclusive",
    "collinear",
    "inclusive_2j",
    "backtoback",
    "inclusive_pt650",
    "inclusive_pt800",
    "inclusive_pt1000",
    "inclusive_2j_pt650",
    "inclusive_2j_pt800",
    "inclusive_2j_pt1000",
]
observables = [
    "nTruthBJet30",
    "nTruthJet30",
    "DeltaRTruthLepJetClosest100",
    "wTruthPt",
    "jet1TruthPt",
    "HtTruth30",
    "mjjTruth",
]


def build_channel(name, rate):
    return {
        "name": name,
        "samples": [
            {
                "name": "main",
                "data": rate,
                "modifiers": [{"name": "mu", "type": "normfactor", "data": None}],
            }
        ],
    }


def build_workspace(el_h, mu_h):
    # get the actual rates
    el_rate = el_h.bin_content * el_h.bin_width
    mu_rate = mu_h.bin_content * mu_h.bin_width

    # get stat uncertainties
    el_stat = np.sqrt(np.nan_to_num(el_h.sumW2))
    mu_stat = np.sqrt(np.nan_to_num(mu_h.sumW2))

    # set expected rate at the naive average
    expected_rate = np.average([el_rate, mu_rate], axis=0)

    workspace = {
        "channels": [
            build_channel("electron", expected_rate.tolist()[1:-1]),
            build_channel("muon", expected_rate.tolist()[1:-1]),
        ],
        "observations": [
            {"name": "electron", "data": el_rate.tolist()[1:-1]},
            {"name": "muon", "data": mu_rate.tolist()[1:-1]},
        ],
        "measurements": [
            {
                "name": "Measurement",
                "config": {
                    "poi": "mu",
                    "parameters": [{"name": "mu", "bounds": [[-10, 10]]}],
                },
            }
        ],
        "version": "1.0.0",
    }

    return workspace, expected_rate, el_rate, mu_rate


def build_histosys(name, rate, up, down):
    return {
        "name": name,
        "type": "histosys",
        "data": {
            "hi_data": ((1.0 + up) * rate).tolist()[1:-1],
            "lo_data": ((1.0 - down) * rate).tolist()[1:-1],
        },
    }


hist_serial = collinearw.serialization.Serialization("histogram")
extra_info = ["systematic", "style", "structure", "systematic-extra"]
for proc_name, el_proc, mu_proc in processes:
    for phsp in phasespaces:
        for obs in observables:
            el_h = config.get(f"{el_proc}//nominal//electron_{phsp}_truth//{obs}")
            mu_h = config.get(f"{mu_proc}//nominal//muon_{phsp}_truth//{obs}")
            # scale to xsec units
            el_h = unfolding.plot.scale_to_xsec(el_h, lumi, lumiunc=lumiunc)
            mu_h = unfolding.plot.scale_to_xsec(mu_h, lumi, lumiunc=lumiunc)

            spec, expected_rate, el_rate, mu_rate = build_workspace(el_h, mu_h)

            bands = list(el_h.systematic_band.keys())
            for band in bands:
                systs = list(el_h.systematic_band[band].components['up'].keys())
                processed = set([])
                for syst in systs:
                    syst_name = syst
                    syst_up = syst
                    syst_dn = syst
                    if isinstance(syst, tuple):
                        syst_name = syst[0]
                        syst_up = tuple(
                            [syst[0], syst[1], syst[2].replace('down', 'up')]
                        )
                        syst_dn = tuple(
                            [syst[0], syst[1], syst[2].replace('up', 'down')]
                        )

                    if syst_name in processed:
                        continue
                    processed.add(syst_name)

                    el_syst_up = el_h.systematic_band[band].components['up'][syst_up]
                    el_syst_dn = el_h.systematic_band[band].components['down'][syst_dn]
                    spec['channels'][0]['samples'][0]['modifiers'].append(
                        build_histosys(
                            syst_name.replace('lepton', 'electron'),
                            el_rate,
                            el_syst_up,
                            el_syst_dn,
                        )
                    )

                    mu_syst_up = mu_h.systematic_band[band].components['up'][syst_up]
                    mu_syst_dn = mu_h.systematic_band[band].components['down'][syst_dn]
                    spec['channels'][1]['samples'][0]['modifiers'].append(
                        build_histosys(
                            syst_name.replace('lepton', 'muon'),
                            mu_rate,
                            el_syst_up,
                            el_syst_dn,
                        )
                    )

            workspace = pyhf.Workspace(spec)
            model = workspace.model()
            data = workspace.data(model)

            pyhf.set_backend('numpy', 'minuit')
            parameters, correlations = pyhf.infer.mle.fit(data, model, return_correlations=True)
            averages = model.expected_actualdata(parameters)
            mu = parameters[model.config.poi_index]
