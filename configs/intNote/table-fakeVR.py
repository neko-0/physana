from collinearw import ConfigMgr
from collinearw import TableMaker

configMgr = ConfigMgr.open("/gpfs/slac/atlas/fs1/d/yuzhan/configs_bk/ColWJet_2022/v6b/run2_MC_driven_Aug2_ABCD/fakes_2211_nominal_corr.pkl")
tm = TableMaker("Tables")

processNameMap = {
    "wjets_2211": "W+jets (Sh 2.2.11)",
    "wjets_2211_tau": "$\\tau\\nu$+jets (Sh 2.2.11)",
    "zjets_2211": "Z+jets (Sh 2.2.11)",
    "diboson": "Diboson",
    "diboson_powheg" : "Diboson (Powheg)",
    "ttbar": "\\ttbar{}",
    "singletop_Wt_DS": "Single top",
    "singletop_stchan":"Single top",
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
    "electron_ABCD-met-pid-met-ELTrack-et-cone-rA_" : "W+jets SR Electron",
    "muon_ABCD-muon-met-iso-rA_" : "W+jets SR Muon",
    #"electron_ABCD-PID-fake-electron-et-cone-rA_" : "W+jets CR",
    #"muon_ABCD-fake-muon-rA_" : "W+jets CR",
}

# B-jet veto effect
#regions = {"ttbarCR_Ele", "ZjetsCR_Ele", "dijetsCR_Ele"}
#regions = {"ttbarCR_Ele", "ZjetsCR_Ele"}
#regions = {"ttbarCR_Mu", "ZjetsCR_Mu"}
#regions = {"dijetsCR_Ele", "dijetsCR_Mu"}
#regions = {"electron_ABCD-met-pid-met-ELTrack-et-cone-rA_", "muon_ABCD-muon-met-iso-rA_"}
regions = {"electron_backtoback_lowMET_reco_ABCD-fake-EL-rA_"}

tm.makeTables(
    configMgr,
    regions=regions,
    regionNameMap=regionNameMap,
    processNameMap=processNameMap,
    excludeProcesses=["dijets","singletop_Wt","singletop_Wt_DS","singletop_stchan"]
)
