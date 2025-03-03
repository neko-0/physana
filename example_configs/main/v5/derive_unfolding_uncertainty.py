from collinearw import ConfigMgr, core
import numpy as np

import pathlib
import logging

logging.basicConfig()
logging.getLogger().setLevel(logging.CRITICAL)

log = logging.getLogger(__name__)

configMgr = ConfigMgr.open(
    '/gpfs/slac/atlas/fs1/d/yuzhan/configs_bk/ColWJet_2022/v5i/run2_MC_driven/ave_band_unfold.pkl'
)

lumi = 138.861
lumiunc = 0.02
configMgr.out_path = pathlib.Path(
    '/sdf/home/g/gstark/collinearw/output/v5i_run2_MC_driven_average'
)

unfoldeds = [
    procset
    for procset in configMgr.process_sets
    if procset.process_type in ["unfolded"] and 'FailPID' not in procset.name
]

unfolded_names = [u.name for u in unfoldeds]

for base in ['closure', 'realthang_fake-MU', 'realthang_fake-EL']:
    unfolded_name = f'unfold_{base}'
    if unfolded_name not in unfolded_names:
        log.warning(f'{unfolded_name} not in file')
        continue

    '''
  unfold_closure ['unfold_alt_closure_wjets_FxFx', 'unfold_alt_closure_wjets', 'unfold_alt_closure_wjets_2211_ASSEW']
  unfold_realthang_fake-MU ['unfold_alt_realthang_fake-MU_wjets_FxFx', 'unfold_alt_realthang_fake-MU_wjets', 'unfold_alt_realthang_fake-MU_wjets_2211_ASSEW']
  unfold_realthang_fake-EL ['unfold_alt_realthang_fake-EL_wjets_FxFx', 'unfold_alt_realthang_fake-EL_wjets', 'unfold_alt_realthang_fake-EL_wjets_2211_ASSEW']
  '''
    unfolded_systs = [
        proc for proc in unfoldeds if base in proc.name and proc.name != unfolded_name
    ]
    unfolded = configMgr.get_process_set(unfolded_name)

    for unfolded_syst in unfolded_systs:
        # unfold_alt_realthang_fake-MU_wjets_FxFx
        suffix = unfolded_syst.name[unfolded_syst.name.index(base) + len(base) :]
        base_process = unfolded.nominal
        comp_process = (base_process - unfolded_syst.nominal) / base_process

        for region in base_process:
            for observable in region:
                shape = observable.shape
                band = core.SystematicBand(f"unfold{suffix}", "theory", shape)
                component = np.nan_to_num(
                    np.abs(
                        comp_process.get_region(region.name)
                        .get_observable(observable.name)
                        .bin_content
                    )
                )
                band.add_component("up", "unfold", component)
                band.add_component("down", "unfold", component)
                observable.update_systematic_band(band)

configMgr.save('band_unfold_syst_added.pkl')
