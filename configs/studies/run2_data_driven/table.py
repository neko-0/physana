from collinearw import ConfigMgr
from collinearw import TableMaker

configMgr = ConfigMgr.open("reco_band.pkl")
tm = TableMaker("Tables")

include_process = ["data", "wjets_2211", "zjets_2211", "ttbar", "dijets", "singletop", "diboson_powheg", "vgamma", "wjets_2211_tau"]

for p in configMgr.list_processes():
    if p not in include_process:
        configMgr.remove_process_set(p)

processNameMap = {
    "wjets_2211": "W+jets (Sh 2.2.11)",
    "wjets_2211_tau": "$\\tau\\nu$+jets (Sh 2.2.11)",
    "zjets_2211": "Z+jets (Sh 2.2.11)",
    "diboson": "Diboson",
    "diboson_powheg" : "Diboson", #(Powheg+Pythia8)
    "ttbar": "\\ttbar{}",
    "singletop": "Single top",
    "dijets" : "Dijets",
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
    "electron_collinear_reco_ABCD-fake-EL-rA_" : "W+jets SR(collinear) Electron",
    "muon_collinear_reco_ABCD-fake-MU-rA_" : "W+jets SR(collinear) Muon",
    #"electron_ABCD-PID-fake-electron-et-cone-rA_" : "W+jets CR",
    #"muon_ABCD-fake-muon-rA_" : "W+jets CR",
}

# B-jet veto effect
#regions = {"ttbarCR_Ele", "ZjetsCR_Ele", "dijetsCR_Ele"}
#regions = {"ttbarCR_Ele", "ZjetsCR_Ele"}
#regions = {"ttbarCR_Mu", "ZjetsCR_Mu"}
#regions = {"dijetsCR_Ele", "dijetsCR_Mu"}
regions = {"electron_collinear_reco_ABCD-fake-EL-rA_", "muon_collinear_reco_ABCD-fake-MU-rA_"}

tm.makeTables(
    configMgr,
    regions=regions,
    regionNameMap=regionNameMap,
    processNameMap=processNameMap,
    signal="wjets_2211",
)
