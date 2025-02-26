from collinearw import ConfigMgr
from collinearw.strategies import unfolding

import logging

logging.basicConfig()
logging.getLogger().setLevel(logging.CRITICAL)

run2_with_fakes = ConfigMgr.open("./output/unfoldTest_v2_fullrun2/fakes.pkl")
unfolding.make_eff_plots(run2_with_fakes)
unfold = unfolding.run_auto(run2_with_fakes, lumi=138.861 * 1000)
# unfold = unfolding.run(run2_with_fakes, 'wjets', 'wjets', 'wjets', 'closure')
unfold.save('unfold.pkl')
unfolding.make_debugging_plots(unfold)
