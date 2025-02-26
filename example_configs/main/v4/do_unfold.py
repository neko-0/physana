from collinearw import ConfigMgr
from collinearw.strategies import unfolding

import pathlib

import logging

logging.basicConfig()
logging.getLogger().setLevel(logging.CRITICAL)

# run2_with_fakes = ConfigMgr.open(f"./output/unfoldTest_v3_fullrun2_sys/fakes.pkl")
run2_with_fakes = ConfigMgr.open(
    #    "/gpfs/slac/atlas/fs1/d/yuzhan/configs_bk/slac_configs/V4_production_ver2/sherpa2211_mc16a/fakes_2211.pkl"
    "/gpfs/slac/atlas/fs1/d/yuzhan/configs_bk/slac_configs/Giordon_Aug2021_Talks/V4_production/track_calo_k_factor_2211/fakes_2211.pkl"
)
run2_with_fakes.out_path = pathlib.Path(
    '/sdf/home/g/gstark/collinearw/output/unfolding_v4_Aug2021Talk_nominal'
)
make_debug_plots = False
lumi = 138.861
# lumi = 36.1
unfold = unfolding.run_auto(
    run2_with_fakes, lumi=lumi, output_folder="unfolding", debug=make_debug_plots
)
# unfold = unfolding.run(run2_with_fakes, 'wjets', 'wjets', 'wjets', 'closure')
unfold.save('unfold.pkl')
