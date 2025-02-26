from collinearw import ConfigMgr
from collinearw.strategies import unfolding

import logging

logging.basicConfig()
logging.getLogger().setLevel(logging.CRITICAL)

run2_with_fakes = ConfigMgr.open(f"./output/unfoldTest_v3_fullrun2_sys/fakes.pkl")
unfolding.make_eff_plots(run2_with_fakes)
# NB: need to pass in common list of systematics=[....] (coordinate w/ Yuzhan's fakes)
unfold = unfolding.run_auto(
    run2_with_fakes, lumi=138.861 * 1000, output_folder=f"unfolding"
)
# unfold = unfolding.run(run2_with_fakes, 'wjets', 'wjets', 'wjets', 'closure')
unfold.save(f'unfold.pkl')
# unfolding.make_debugging_plots(unfold, output_folder=f"unfolding")
