from collinearw import ConfigMgr
from collinearw.strategies import unfolding

import pathlib
import logging

logging.basicConfig()
logging.getLogger().setLevel(logging.CRITICAL)

configMgr = ConfigMgr.open(f'/gpfs/slac/atlas/fs1/d/yuzhan/configs_bk/ColWJet_2022/v6b/run2_MC_driven_Sep14/average_Oct12_full_syst_with_fxfx.pkl')

syst_config = ConfigMgr.open("/gpfs/slac/atlas/fs1/d/yuzhan/configs_bk/ColWJet_2022/v6b/run2_MC_driven_Sep14/band_unfolded_Oct_unfold_uncertainty_with_FxFx.pkl")
# syst_config.remove_process("wjets")
# syst_config.remove_process("wjets_2211_ASSEW")

configMgr = ConfigMgr.intersection([configMgr, syst_config], copy=False)
configMgr.update_children_parent()

lumi = 138.861
#lumiunc = 0.02
lumiunc = 0.0
configMgr.out_path = pathlib.Path("/sdf/home/g/gstark/collinearw/output/v6b")

observable_name = 'nTruthBJet30'

region_name_mapping = {
    'inclusive_truth': 'Inclusive',
    'inclusive_2j_truth': 'Inc. (njet#geq2)',
    'collinear_truth': 'Collinear',
    #'backtoback_truth': 'Backtoback',
    'muon_inclusive_truth': 'Inclusive',
    'muon_collinear_truth': 'Collinear',
    'muon_inclusive_2j_truth': 'Inc. (njet#geq2)',
    #'muon_backtoback_truth': 'Backtoback',
    # 'muon_backtoback_truth': 'backtoback',
    # 'muon_inclusive_2j_truth': 'inclusive_2j',
    # 'electron_inclusive_truth': 'inclusive',
    # 'electron_collinear_truth': 'collinear',
    # 'electron_backtoback_truth': 'backtoback',
    # 'electron_inclusive_2j_truth': 'inclusive_2j',
    # 'inclusive_pt650_truth': 'Inclusive (jet_{1}>650 GeV)',
    # 'inclusive_2j_pt650_truth': 'inclusive (njet#gq,jet_{1}>650 GeV)',
    # 'inclusive_pt800_truth': 'Inclusive (jet_{1}>800 GeV)',
    # 'inclusive_2j_pt800_truth': 'inclusive (njet#gq, jet_{1}>1000 GeV)',
    # 'inclusive_pt1000_truth': 'Inclusive (jet_{1}>1000 GeV)',
    # 'inclusive_2j_pt1000_truth': 'inclusive (njet#gq, jet_{1}>1000 GeV)',
}

region_names = ["Inclusive", "Inc. (njet#geq2)", "Collinear"]

avg_suffix = ""

process_name_mapping = {
    # 'wjets_MU': 'wjets',
    # 'wjets_EL': 'wjets',
    # 'wjets_2211_MU': 'wjets_2211',
    # 'wjets_2211_EL': 'wjets_2211',
    # 'wjets_2211_ASSEW_MU': 'wjets_2211_ASSEW',
    # 'wjets_2211_ASSEW_EL': 'wjets_2211_ASSEW',
    # 'wjets_FxFx_MU': 'wjets_FxFx',
    # 'wjets_FxFx_EL': 'wjets_FxFx',
    # 'unfold_realthang_fake-MU': 'unfold_realthang_fake-MU',
    # 'unfold_realthang_fake-EL': 'unfold_realthang_fake-EL',
    # # 'unfold_realthang_average_comp': 'unfold_realthang_average_comp',
    # 'unfold_realthang_average_BLUE': 'unfold_realthang_average_BLUE',
    # 'wjets_average_BLUE': 'wjets_average_BLUE',
    # 'wjets_2211_average_BLUE': 'wjets_2211_average_BLUE',
    # 'wjets_FxFx_average_BLUE': 'wjets_FxFx_average_BLUE',
    "wjets_2211": "Sherpa 2.2.11 NLO QCD",
    "wjets_2211_ASSEW": "Sherpa 2.2.11 NLO QCD+EW_{virt}",
    "unfold_realthang_average": "Data",
    "wjets_FxFx": "MG_aMC@NLO+Pythia8 FxFx",
    #"Sherpa 2.2.1 NLO QCD" : "wjets",
}

exclude_names = ["diboson", "unfold_wjets", "unfold_wjets_2211_ASSEW"]

processes = [
    configMgr.get_process(process_name).copy()
    for process_name in process_name_mapping
]

for process in processes:
  process.title = process_name_mapping[process.name]

legend_order = [2, 0, 1, 3]
legend_styles = ["p", "p", "pe", "p"]

base_output_path = (
    pathlib.Path(configMgr.out_path)
    .joinpath(
        "cross-section",
        observable_name,
    )
    .resolve()
)
output_path = base_output_path.with_suffix('.pdf')
output_path.parent.mkdir(parents=True, exist_ok=True)

data = {
    region_name: {name: (0.0, 0.0, 0.0) for name in process_name_mapping}
    for region_name in region_names
}
for process in processes:
    if process.name == "wjets_2211":
        process.markerstyle = 25
        process.color = 9#"#e93524"
        # process.fillstyle=1001
        # process.alpha = 0.5
    elif process.name == "wjets_2211_ASSEW":
        process.markerstyle = 27
        process.color = 46#"#1a73e8"
        # process.fillstyle=1001
        # process.alpha = 0.5
    elif process.name == "wjets":
        process.markerstyle = 27
        process.color = 46#"#F39C12"
        # process.fillstyle=1001
        # process.alpha = 0.5
    elif process.name == "wjets_FxFx":
        process.markerstyle = 24
        process.color = 30#"#9B59B6"
        # process.fillstyle=1001
        # process.alpha = 0.5

    if process.name.endswith('MU'):
        regions = [
            'muon_inclusive_truth',
            'muon_inclusive_2j_truth',
            'muon_collinear_truth',
            #'muon_backtoback_truth',
        ]
    elif process.name.endswith('EL'):
        regions = [
            'electron_inclusive_truth',
            'electron_inclusive_2j_truth',
            'electron_collinear_truth',
            #'electron_backtoback_truth',
        ]
    elif "average" not in process.name:
        regions = [
            "muon_inclusive_truth",
            "muon_collinear_truth",
            "muon_inclusive_2j_truth",
            #"muon_backtoback_truth",
        ]
    else:
        regions = [
            'inclusive_truth',
            'inclusive_2j_truth',
            'collinear_truth',
            #'backtoback_truth',
            # 'inclusive_pt650_truth',
            # 'inclusive_pt800_truth',
            # 'inclusive_pt1000_truth',
            # 'inclusive_2j_pt650_truth',
            # 'inclusive_2j_pt800_truth',
            # 'inclusive_2j_pt1000_truth',
        ]

    for source_region_name in regions:
        target_region_name = region_name_mapping[source_region_name]

        try:
            observable = process.get_region(source_region_name).get_observable(
                observable_name
            )
        except KeyError:
            observable = process.get_region(source_region_name).get_observable(
                f'{observable_name}{avg_suffix}'
            )
        # extra_exc = ["PDF"] if process.name == "wjets_2211" else []
        extra_exc = []
        # xsec_data == xsec, stat_error, syst_error
        xsec, stat_error, syst_errors = unfolding.utils.get_xsec_uncert(
            observable,
            lumi,
            lumiunc,
            exclude_names=exclude_names + extra_exc,
        )

        data[target_region_name][process.name] = (
            xsec,
            stat_error,
            stat_error,
            *syst_errors,
        )

unfolding.plot.plot_xsec_integral(
    configMgr,
    region_names,
    list(process_name_mapping.keys()),
    processes,
    output_path,
    legend_order=legend_order,
    legend_styles=legend_styles,
    yrange=(1.5e2, 1.2e4),
    yrange_ratio=(0.6, 1.6),
    ratio_base_idx=list(process_name_mapping.keys()).index('unfold_realthang_average'),
    legend_args={"y1":0.5, "x1":0.5, "text_size":0.05},
    **data,
)
