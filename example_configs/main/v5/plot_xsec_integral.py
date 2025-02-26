from collinearw import ConfigMgr
from collinearw.strategies import unfolding

import pathlib
import logging

logging.basicConfig()
logging.getLogger().setLevel(logging.CRITICAL)

configMgr = ConfigMgr.open(
    '/gpfs/slac/atlas/fs1/d/yuzhan/configs_bk/ColWJet_2022/v5i/run2_MC_driven/band_unfold_syst_added_ew_added_BLUE_added_May17.pkl'
)

lumi = 138.861
lumiunc = 0.02
configMgr.out_path = pathlib.Path(
    '/sdf/home/g/gstark/collinearw/output/v5i_ewAdded_BLUEadded_May17'
)

include_lepton = False

observable_name = 'nTruthBJet30'

region_name_mapping = {
    'inclusive_truth': 'inclusive',
    'inclusive_2j_truth': 'inclusive_2j',
    'collinear_truth': 'collinear',
    'backtoback_truth': 'backtoback',
    'muon_inclusive_truth': 'inclusive',
    'muon_collinear_truth': 'collinear',
    'muon_backtoback_truth': 'backtoback',
    'muon_inclusive_2j_truth': 'inclusive_2j',
    'electron_inclusive_truth': 'inclusive',
    'electron_collinear_truth': 'collinear',
    'electron_backtoback_truth': 'backtoback',
    'electron_inclusive_2j_truth': 'inclusive_2j',
}

for k,v in list(region_name_mapping.items()):
    if 'inclusive_2j' in k:
        region_name_mapping[k.replace('inclusive_2j', 'inclusive_2j_pt650')] = 'inclusive_2j_650'
        continue
    if 'inclusive' in k:
        for pt in ['pt650', 'pt800', 'pt1000']:
            region_name_mapping[k.replace('inclusive', f'inclusive_{pt}')] = v.replace('inclusive', f'inclusive_{pt}')

region_names = sorted(set(region_name_mapping.values()))

process_name_mapping = {
    'wjets_MU': 'wjets',
    'wjets_EL': 'wjets',
    'wjets_2211_MU': 'wjets_2211',
    'wjets_2211_EL': 'wjets_2211',
    'wjets_2211_ASSEW_MU': 'wjets_2211_ASSEW',
    'wjets_2211_ASSEW_EL': 'wjets_2211_ASSEW',
    'wjets_FxFx_MU': 'wjets_FxFx',
    'wjets_FxFx_EL': 'wjets_FxFx',
    'unfold_realthang_fake-MU': 'unfold_realthang_fake-MU',
    'unfold_realthang_fake-EL': 'unfold_realthang_fake-EL',
    'unfold_realthang_average_BLUE': 'unfold_realthang_average_BLUE',
    'wjets_average_BLUE': 'wjets_average_BLUE',
    'wjets_2211_average_BLUE': 'wjets_2211_average_BLUE',
    'wjets_2211_ASSEW_average_BLUE': 'wjets_2211_ASSEW_average_BLUE',
    'wjets_FxFx_average_BLUE': 'wjets_FxFx_average_BLUE',
}
legend_order_names = [
  'unfold_realthang_average_BLUE',
  'unfold_realthang_fake-EL',
  'unfold_realthang_fake-MU',
  'wjets_2211_average_BLUE',
  'wjets_2211_MU',
  'wjets_2211_EL',
  'wjets_2211_ASSEW_average_BLUE',
  'wjets_2211_ASSEW_MU',
  'wjets_2211_ASSEW_EL',
  'wjets_FxFx_average_BLUE',
  'wjets_FxFx_MU',
  'wjets_FxFx_EL',
  'wjets_average_BLUE',
  'wjets_MU',
  'wjets_EL',
]

if not include_lepton:
    legend_order_names = [name for name in legend_order_names if not name.endswith('EL') and not name.endswith('MU')]
    process_name_mapping = dict((k,v) for k,v in process_name_mapping.items() if not k.endswith('EL') and not k.endswith('MU'))

processes = [
    configMgr.get_process(process_name).copy()
    for process_name in process_name_mapping.values()
]
names = list(process_name_mapping.keys())
legend_order = [list(process_name_mapping.keys()).index(name) for name in legend_order_names]
assert len(names) == len(processes)

for idx, process_name in enumerate(process_name_mapping.keys()):
    if process_name.startswith('unfold'):
        processes[idx].title = 'Data (ave.)'

    if process_name.endswith('BLUE'):
        continue
    elif process_name.endswith('MU'):
        processes[idx].title = processes[idx].title + ' (mu.)'
        processes[idx].markerstyle = 23
    else:
        processes[idx].title = processes[idx].title + ' (el.)'
        processes[idx].markerstyle = 22

base_output_path = (
    pathlib.Path(configMgr.out_path)
    .joinpath(
        "tables",
        "study_xsec_total",
        observable_name,
    )
    .resolve()
)
output_path = base_output_path.with_suffix('.pdf')
output_path.parent.mkdir(parents=True, exist_ok=True)

data = {
    region_name: {name: (0.0, 0.0, 0.0) for name in names}
    for region_name in region_names
}
for name, process in zip(names, processes):
    if name.endswith('MU'):
        regions = [
            'muon_inclusive_truth',
            'muon_inclusive_2j_truth',
            'muon_collinear_truth',
            'muon_backtoback_truth',
        ]
    elif name.endswith('EL'):
        regions = [
            'electron_inclusive_truth',
            'electron_inclusive_2j_truth',
            'electron_collinear_truth',
            'electron_backtoback_truth',
        ]
    else:
        regions = [
            'inclusive_truth',
            'inclusive_2j_truth',
            'collinear_truth',
            'backtoback_truth',
        ]

    for k in list(regions):
        if 'inclusive_2j' in k:
            regions.append(k.replace('inclusive_2j', 'inclusive_2j_pt650'))
            continue
        if 'inclusive' in k:
            for pt in ['pt650', 'pt800', 'pt1000']:
                regions.append(k.replace('inclusive', f'inclusive_{pt}'))

    for source_region_name in regions:
        target_region_name = region_name_mapping[source_region_name]

        try:
            observable = process.get_region(source_region_name).get_observable(
                observable_name
            )
        except KeyError:
            observable = process.get_region(source_region_name).get_observable(
                f'{observable_name}_UP'
            )

        # hardcode exclude names
        exclude_names = None
        if process.name.startswith("wjets"):
            if process.name.startswith("wjets_FxFx") or process.name.startswith("wjets_2211"):
                pass
            else:
                exclude_names = ["PDF"]
        elif process.name.startswith("unfold"):
            exclude_names = ["diboson", "diboson_powheg", "unfold_wjets", "unfold_wjets_2211_ASSEW"]


        # xsec_data == xsec, stat_error, syst_error
        xsec, stat_error, syst_errors = unfolding.utils.get_xsec_uncert(
            observable,
            lumi,
            lumiunc,
            exclude_names=exclude_names
        )

        data[target_region_name][name] = (
            xsec,
            stat_error,
            stat_error,
            *syst_errors,
        )

unfolding.plot.plot_xsec_integral(
    configMgr,
    region_names,
    names,
    processes,
    output_path,
    legend_order=legend_order,
    yrange=(1e1, 5e3),
    yrange_ratio=(0.8, 1.3),
    ratio_base_idx=names.index('unfold_realthang_average_BLUE'),
    **data,
)
