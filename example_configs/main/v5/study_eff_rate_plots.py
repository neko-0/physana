from collinearw import ConfigMgr
from collinearw.strategies import unfolding

import pathlib

import logging

logging.basicConfig()
logging.getLogger().setLevel(logging.CRITICAL)

run2_with_fakes = ConfigMgr.open(
    "/gpfs/slac/atlas/fs1/d/yuzhan/configs_bk/slac_configs/Giordon_Aug2021_Talks/V5_production/track_calo_with_more_sys_2211_run2/fakes_2211_nominal_only.pkl"
)
run2_with_fakes.out_path = pathlib.Path(
    '/sdf/home/g/gstark/collinearw/output/unfolding_v5'
)

unfolding.plot.make_eff_plots(run2_with_fakes)
