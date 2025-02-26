from collinearw import ConfigMgr
from collinearw import run_HistMaker
from collinearw.strategies import abcd, unfolding

import copy
import os

import logging

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
log = logging.getLogger(__name__)

#
setting = ConfigMgr(
    src_path="/nfs/slac/atlas/fs1/d/yuzhan/collinearw_files/June2020_Production/merged_files/",
    out_path="./unfoldTest_v2_fullrun2/",  # output
    description="unfolding",
    io_backend="uproot",  # io backend
    histogram_backend="numpy",  # histograms backend
)

# Phase-space corrections
basedir = os.path.dirname(os.path.realpath(__file__))
corr_configMgr = ConfigMgr.open(
    f"{basedir}/../../../data/Wj_AB212108_v3/ControlRegions/corrections.pkl"
)
setting.corrections = corr_configMgr.corrections

#
setting.set_singlefile(
    [
        "Wj_AB212108_v2_mc16a.root",
        "Wj_AB212108_v2_mc16d.root",
        "Wj_AB212108_v2_mc16e.root",
    ]
)

# name, treename, selection, process_type
setting.add_process(
    "data", "data", process_type="data", color=1, legendname="Data", binerror=1
)  # black, kPoisson error
setting.add_process(
    "wjets",
    "wjets_NoSys",
    process_type="signal",
    color=600,
    markerstyle=8,
    legendname="Sherpa 2.2.1",
)  # blue
setting.add_process(
    "wjets_mg",
    "wjets_mg_NoSys",
    process_type="signal_alt",
    color=632,
    markerstyle=22,
    legendname="MadGraph+Pythia8 CKKW-L",
)  # red
"""
setting.add_process(
    "wjets_FxFx",
    "wjets_FxFx_NoSys",
    process_type="signal_alt",
    color=418,
    lumi=1.07,
    legendname="Sherpa 2.2.10 FxFx",
    markerstyle=28,
)  # green+2
"""
setting.add_process("ttbar", "ttbar_NoSys", process_type="bkg", color=859)  # azure-1
setting.add_process(
    "zjets", "zjets_NoSys", process_type="bkg", color=861, legendname="Z+jets"
)  # azure+1
setting.add_process(
    "singletop", "singletop_NoSys", process_type="bkg", color=800
)  # orange
setting.add_process(
    "diboson", "diboson_NoSys", process_type="bkg", color=411
)  # green-5

# selections

lepton_selections = {"electron": ["AnalysisType==1"], "muon": ["AnalysisType==2"]}

phasespace_selections = {
    "inclusive": [],
    "inclusive-closestmatching": [],
    # "collinear": ["DeltaPhiWJetClosest25<=1.5"],
    # "backtoback": ["DeltaPhiWJetClosest25>1.5"],
}
truth_selection = [
    "isTruth",
    "nTruthBJet25==0",
    "lep1TruthPt>30.0",
    "jet1TruthPt>500.0",
    "metTruth>25.0",
]
reco_selection = [
    "isReco",
    "nBJet25==0",
    "trigMatch_singleLepTrig",
    "lep1Pt>30.0",
    "jet1Pt>=500",
]
# Any time you have an observable that requires that closeby jet, you need to
# match it.  There is always a leading jet in the event, so that has to be
# matched all the time.  But to define collinear / BTB, you search for a jet
# close to the W. this could be the leading jet, but could also not be.
truth_matching = {
    "inclusive": ["TM_lepDR<0.4", "TM_leadJetDR<0.4"],
    "inclusive-closestmatching": [
        "TM_lepDR<0.4",
        "TM_leadJetDR<0.4",
        "TM_closestJet30DR<0.4",
    ],
    "collinear": ["TM_lepDR<0.4", "TM_leadJetDR<0.4", "TM_closestJet30DR<0.4"],
    "backtoback": ["TM_lepDR<0.4", "TM_leadJetDR<0.4", "TM_closestJet30DR<0.4"],
}
truth_notmatching = {k: [f"!({' && '.join(v)})"] for k, v in truth_matching.items()}

# needed for the matching inefficiency (for unfolding)
# NB: make sure this matches what we use for ABCD
isolation_selection = [
    "met > 25",
    "lep1Signal==1",
    "((ptcone20_TightTTVALooseCone_pt500/lep1Pt < 0.06 && AnalysisType==1) || (IsoFCLoose_FixedRad==1 && AnalysisType==2))",
]

# weights we need for each
weight_reco = "genWeight*eventWeight*bTagWeight*pileupWeight*leptonWeight*jvtWeight"
weight_truth = "genWeight*eventWeight"

for lepton_flavor, lepton_selection in lepton_selections.items():
    for phasespace, phasespace_selection in phasespace_selections.items():

        tSelection = copy.copy(truth_selection)
        rSelection = copy.copy(reco_selection)
        rSelection += lepton_selection

        # We should improve this, maybe by having the truth selection
        # bundled with the lepton selection? Works for now...
        if lepton_flavor == 'electron':
            tSelection.append('TMath::Abs(lep1TruthPdgId)==11')
            corr_type = "electron"
        if lepton_flavor == "muon":
            tSelection.append('TMath::Abs(lep1TruthPdgId)==13')
            corr_type = "muon"

        unfolding.add_regions(
            setting,
            f"{lepton_flavor}_{phasespace}",
            truth_selection=tSelection,
            reco_selection=rSelection,
            matching_selection=truth_matching[phasespace],
            notmatching_selection=truth_notmatching[phasespace] + isolation_selection,
            weight_truth=weight_truth,
            weight_reco=weight_reco,
            corr_type=corr_type,
        )

abcd.reserve_abcd_regions(setting, "PID", ("met", 25), ("lep1Signal", 1))
abcd.reserve_abcd_regions(setting, "fake-muon", ("met", 25), ("IsoFCLoose_FixedRad", 1))
abcd.reserve_abcd_regions(
    setting,
    "fake-electron",
    ("met", 25),
    ("ptcone20_TightTTVALooseCone_pt500/lep1Pt", 0.06),
    reverse_y=True,
)
abcd.create_abcd_regions(
    setting,
    ["PID", "fake-muon"],
    base_region_names=[
        region.name for region in setting.regions if 'muon' in region.name
    ],
)
abcd.create_abcd_regions(
    setting,
    ["PID", "fake-electron"],
    base_region_names=[
        region.name for region in setting.regions if 'electron' in region.name
    ],
)

unfolding.add_observables(
    setting, "lep1Pt", "lep1TruthPt", 10, 30, 1030, "Leading lepton p_{T} [GeV]"
)
unfolding.add_observables(
    setting, "jet1Pt", "jet1TruthPt", 15, 500, 2000, "Leading jet p_{T} [GeV]"
)
unfolding.add_observables(
    setting, "wPt", "wTruthPt", 15, 0, 1500, "Leading W p_{T} [GeV]"
)
unfolding.add_observables(setting, "Ht", "HtTruth", 25, 500, 3000, "H_{T} [GeV]")

unfolding.add_observables(
    setting, "mjj", "mjjTruth", 20, 0, 2000, "Leading and sub-leading jet mass [GeV]"
)
unfolding.add_observables(
    setting, "nJet25", "nTruthJet25", 9, 1, 10, "Number of jets (p_{T} > 25 GeV)"
)
unfolding.add_observables(
    setting, "nJet100", "nTruthJet100", 9, 1, 10, "Number of jets (p_{T} > 100 GeV)"
)
unfolding.add_observables(
    setting,
    "DeltaPhiWJetClosest25",
    "DeltaPhiTruthWJetClosest25",
    16,
    0,
    3.2,
    "min(#Delta#phi(W,jet_{i}^{25}))",
)
unfolding.add_observables(
    setting,
    "DeltaRLepJetClosest100",
    "DeltaRTruthLepJetClosest100",
    25,
    0,
    5,
    "min(#DeltaR(lepton,jet_{i}^{100}))",
)

setting.add_histogram2D(
    "eta_vs_lepPt_electron",
    "TMath::Abs(lep1Eta)",
    "lep1Pt",
    xbin=[0.0, 1.05, 1.37, 1.52, 2.0, 2.5],
    ybin=[
        30.0,
        50.0,
        100.0,
        150.0,
        200.0,
        300.0,
        400.0,
        500.0,
        600,
        800,
        1000,
        1500,
        2000,
    ],
    xtitle="eta (2d)",
    ytitle="Lepton pT (2d)",
)

setting.add_histogram2D(
    "eta_vs_lepPt_muon",
    "TMath::Abs(lep1Eta)",
    "lep1Pt",
    xbin=[0, 0.5, 1.0, 1.5, 2.0, 2.5],
    ybin=[30.0, 50.0, 100.0, 150.0, 200.0, 500.0, 2000.0],
    xtitle="eta (2d)",
    ytitle="Lepton pT (2d)",
)
setting.prepare()

setting.save("run2.pkl")
mc16a_setting = run_HistMaker.run_HistMaker(setting, "run2.pkl")  # , rsplit=True)
