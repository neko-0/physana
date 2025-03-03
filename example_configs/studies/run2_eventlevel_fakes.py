from collinearw import ConfigMgr, HistManipulate
from collinearw import run_HistMaker
from collinearw import run_HistManipulate
from collinearw import run_PlotMaker

import logging

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
log = logging.getLogger(__name__)


dijets = ConfigMgr.open(
    "/nfs/slac/atlas/fs1/d/yuzhan/collinearw_ana_2020/Sep_Production_Wj_AB212108_v2/ABCD_study/dijets_closure_standalone_v1/abcd_closure.pkl"
)
# dijets = ConfigMgr.open("/nfs/slac/atlas/fs1/d/yuzhan/collinearw_ana_2020/Feb_Production_Wj_AB212108_v1/ABCD_study/dijets_closure_standalone_v10/abcd_closure.pkl")
dijets_tf_config = run_HistManipulate.run_ABCD_TF(dijets, "dijets", oname=None)

run2 = ConfigMgr.open(
    "/nfs/slac/atlas/fs1/d/yuzhan/collinearw_ana_2020/June_Production_Wj_AB212108_v2/ABCD_study/run2_v13/run2.pkl"
)
subed_run2 = HistManipulate.Subtract_MC(run2, "data", "subtracted_data")
self_tf_configMgr = run_HistManipulate.run_ABCD_TF(
    subed_run2, "subtracted_data", oname="self_tf_config"
)
# self_tf_configMgr = ConfigMgr.open("/nfs/slac/atlas/fs1/d/yuzhan/collinearw_ana_2020/June_Production_Wj_AB212108_v2/ABCD_study/run2_v12/self_tf_config.pkl")

# run_HistManipulate.run_ABCD_Fakes_EventLevel(run2, dijets_tf_config, "dijets", {"muon*":(("lep1Eta","lep1Pt"),"eta_vs_lepPt_good"), "electron*":(("lep1Eta","lep1Pt"),"eta_vs_lepPt")}, "run2_fake_event_level_dijets_lPtEta_good_check_mc16a")
# run_HistManipulate.run_ABCD_Fakes_EventLevel(run2, dijets_tf_config, "dijets", {"muon*":(("abs(lep1Eta)","lep1Pt"), "abs(eta)_vs_lepPt_good"), "electron*":(("lep1Eta","lep1Pt"),"eta_vs_lepPt")}, "run2_fake_event_level_dijets_abslPtEta_good_check_mc16a")
# run_HistManipulate.run_ABCD_Fakes_EventLevel(run2, self_tf_configMgr, "dijets", {"muon*":(("lep1Eta","lep1Pt"),"eta_vs_lepPt_good"), "electron*":(("lep1Eta","lep1Pt"),"eta_vs_lepPt")}, "run2_fake_event_level_dijets_lPtEta_good_check_mc16a")
# run_HistManipulate.run_ABCD_Fakes_EventLevel(run2, self_tf_configMgr, "dijets", {"muon*":(("abs(lep1Eta)","lep1Pt"), "abs(eta)_vs_lepPt_good"), "electron*":(("lep1Eta","lep1Pt"),"eta_vs_lepPt")}, "run2_fake_event_level_dijets_abslPtEta_good_check_mc16a")

run_HistManipulate.run_ABCD_Fakes_EventLevel(
    run2,
    dijets_tf_config,
    "dijets",
    (("lep1Eta", "lep1Pt"), "eta_vs_lepPt_good"),
    "run2_fake_event_level_dijets_lPtEta_good_run2_check",
)
# run_HistManipulate.run_ABCD_Fakes_EventLevel(run2, dijets_tf_config, "dijets", (("abs(lep1Eta)","lep1Pt"), "abs(eta)_vs_lepPt_good"), "run2_fake_event_level_dijets_abslPtEta_good_run2")
run_HistManipulate.run_ABCD_Fakes_EventLevel(
    run2,
    self_tf_configMgr,
    "subtracted_data",
    (("lep1Eta", "lep1Pt"), "eta_vs_lepPt_good"),
    "run2_fake_event_level_self_lPtEta_good_run2",
)
# run_HistManipulate.run_ABCD_Fakes_EventLevel(run2, self_tf_configMgr, "subtracted_data", (("abs(lep1Eta)","lep1Pt"), "abs(eta)_vs_lepPt_good"), "run2_fake_event_level_self_abslPtEta_good_run2")
