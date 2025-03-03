from collinearw import ConfigMgr
from collinearw import TableMaker

configMgr = ConfigMgr.open("iterative/run2.pkl")
tm = TableMaker("Tables")

processNameMap = {
    "wjets_2211": "W+jets (Sherpa 2.2.11)",
    "zjets_2211": "Z+jets (Sherpa 2.2.11)",
    "diboson": "Diboson",
    "ttbar": "$t\\bar{t}$",
    "singletop": "Single top",
}

regionNameMap = {
    "Inclusive": "Inclusive selection",
    "BVeto": "B-jet veto",
    "ZjetsCR_Mu" : "Z+jets CR",
    "ttbarCR_Mu" : "t\\bar{t}\\. CR",
    "ZjetsCR_Ele" : "Z+jets CR",
    "ttbarCR_Ele" : "t\\bar{t}\\. CR",
    "electron_inclusive_reco_ABCD-fake-EL-rA_" : "W+jets CR",
    "muon_inclusive_reco_ABCD-fake-MU-rA_" : "W+jets CR",
    "electron_ABCD-PID-fake-electron-et-cone-rA_" : "W+jets CR",
    "muon_ABCD-fake-muon-rA_" : "W+jets CR",
}

# B-jet veto effect
regions = {"ttbarCR_Ele", "ZjetsCR_Ele", "electron_ABCD-PID-fake-electron-et-cone-rA_"}

tm.makeTables(
    configMgr,
    regions=regions,
    regionNameMap=regionNameMap,
    processNameMap=processNameMap,
)
