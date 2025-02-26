from collinearw import ConfigMgr
from collinearw import TableMaker

configMgr = ConfigMgr.open("/gpfs/slac/atlas/fs1/d/yuzhan/configs_bk/ColWJet_2022/v6b/run2_MC_driven_Aug2_ABCD/fakes_2211_nominal_corr.pkl")
tm = TableMaker("Tables")

processNameMap = {
    "wjets_2211": "W+jets (Sh 2.2.11)",
    "wjets_2211_tau": "$\\tau\\nu$+jets",
    "wjets_tau": "$\\tau\\nu$+jets",
    "wjets_EW_2211" :"EW V+jets",
    "zjets_2211": "Z+jets",
    "diboson": "Diboson",
    "diboson_powheg" : "Diboson",
    "vgamma" : "V+$\\gamma$+jets",
    "ttbar": "\\ttbar{}",
    "singletop": "Single top",
    "dijets" : "Multi-jet (MC based)",
    "fakes" : "Multi-jet (Data-driven)"
}

regionNameMap = {
    "Inclusive": "Inclusive selection",
    "BVeto": "B-jet veto",
    "ZjetsCR_Mu" : "Z+jets CR",
    "ttbarCR_Mu" : "\\ttbar{} CR",
    "ZjetsCR_Ele" : "Z+jets CR",
    "ttbarCR_Ele" : "\\ttbar{} CR",
    "dijetsCR_Ele" : "Electron",
    "dijetsCR_Mu" : "Muon",
    "muon_inclusive_reco_ABCD-fake-MU-rA_" : "Inclusive",
    "muon_inclusive_2j_reco_ABCD-fake-MU-rA_" : "Inclusive 2-jet",
    "muon_collinear_reco_ABCD-fake-MU-rA_" : "Collinear",
    "electron_inclusive_reco_ABCD-fake-EL-rA_" : "Inclusive",
    "electron_inclusive_2j_reco_ABCD-fake-EL-rA_" : "Inclusive 2-jet",
    "electron_collinear_reco_ABCD-fake-EL-rA_" : "Collinear",
}

# B-jet veto effect
#regions = {"ttbarCR_Ele", "ZjetsCR_Ele", "dijetsCR_Ele"}
#regions = {"ttbarCR_Ele", "ZjetsCR_Ele"}
#regions = {"ttbarCR_Mu", "ZjetsCR_Mu"}
#regions = {"dijetsCR_Ele", "dijetsCR_Mu"}
#regions = {"electron_ABCD-met-pid-met-ELTrack-et-cone-rA_", "muon_ABCD-muon-met-iso-rA_"}
regions = {"muon_inclusive_reco_ABCD-fake-MU-rA_","muon_inclusive_2j_reco_ABCD-fake-MU-rA_","muon_collinear_reco_ABCD-fake-MU-rA_"}
#regions = {"electron_inclusive_reco_ABCD-fake-EL-rA_","electron_inclusive_2j_reco_ABCD-fake-EL-rA_","electron_collinear_reco_ABCD-fake-EL-rA_"}

tm.makeTables(
    configMgr,
    regions=regions,
    regionNameMap=regionNameMap,
    processNameMap=processNameMap,
    excludeProcesses=["wjets_2211","dijets","singletop_Wt","singletop_Wt_DS","singletop_stchan"]
)
