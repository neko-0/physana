from collinearw import ConfigMgr
from collinearw import run_HistMaker
from collinearw.strategies import abcd, unfolding
from collinearw.serialization import Serialization

import copy
import os

import logging

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
log = logging.getLogger(__name__)

#
setting = ConfigMgr(
    src_path="/gpfs/slac/atlas/fs1/d/yuzhan/collinearw_files/Wj_AB212108_v3p3/merged/",
    out_path="./unfoldTest_v3_fullrun2_sys/",  # output
    description="unfolding",
    io_backend="uproot",  # io backend
    histogram_backend="numpy",  # histograms backend
)

# Phase-space corrections, but only load the non signal processes (e.g zjets, ttbar)
# the correction format is a python dict with key as the form (correction type, process name, observable)
# unfortunatly the correction are derived separetely for each generator, if you want to have single
# correction file for different wjets generator, you might need to merge and rename the key yourself.
basedir = os.path.dirname(os.path.realpath(__file__))
m_serial = Serialization()
bkgd_correction = m_serial.from_pickle(
    f"{basedir}/../../../data/Wj_AB212108_v3/ControlRegions/wjets_Sherpa_bkgd_correction.pkl"
)
setting.corrections = bkgd_correction

#
setting.set_singlefile(
    [
        "mc16a.root",
        # "mc16d.root",
        # "mc16e.root",
    ]
)


# name, treename, selection, process_type
setting.add_process(
    "data",
    # "data",
    process_type="data",
    color=1,
    legendname="Data",
    binerror=1,
)  # black, kPoisson error
setting.add_process(
    "wjets",
    # "wjets_NoSys",
    process_type="signal",
    color=600,
    markerstyle=8,
    legendname="Sherpa 2.2.1",
)  # blue
setting.add_process(
    "wjets_mg",
    # "wjets_mg_NoSys",
    process_type="signal_alt",
    color=632,
    markerstyle=22,
    legendname="MadGraph+Pythia8 CKKW-L",
)  # red
"""
setting.add_process(
    "wjets_FxFx",
    #"wjets_FxFx_NoSys",
    process_type="signal_alt",
    color=418,
    lumi=1.07,
    legendname="Sherpa 2.2.10 FxFx",
    markerstyle=28,
)  # green+2
"""
setting.add_process(
    "ttbar",
    # "ttbar_NoSys",
    process_type="bkg",
    color=859,
)  # azure-1
setting.add_process(
    "zjets",
    # "zjets_NoSys",
    process_type="bkg",
    color=861,
    legendname="Z+jets",
)  # azure+1
setting.add_process(
    "singletop",
    # "singletop_NoSys",
    process_type="bkg",
    color=800,
)  # orange
setting.add_process(
    "diboson",
    # "diboson_NoSys",
    process_type="bkg",
    color=411,
)  # green-5


"""
# define tree systematics
setting.define_tree_systematics(
    "jet_insitu", # this is just a name you define for this type of systematics
    ["JET_JER_DataVsMC_MC16__1down", "MET_SoftTrk_ScaleDown", "MET_SoftTrk_ScaleUp"], # this is tree name in the ROOT file
    sys_type="stdev", # latter for dealing with systematics, might need improvements as we go
    normalize=True,
    symmetrize=False,
)
"""

"""TODO: consider loading from a file
[
  {
    "name": "purw_sys",
    "sys_type": "stdev",
    "values": ["pileupWeightDown", "pileupWeightUp"],
    "kind": "weight",
    "processes": ["wjets", "ttbar", "zjets", "singletop", "diboson"]
  }
]
"""
# define weight systematics
# NB: generalize this better and normalize out the pileupWeight
setting.define_weight_systematics(
    "purw_sys",  # just a name
    ["pileupWeightDown", "pileupWeightUp"],  # this is the branch name in the ttree
    sys_type="stdev",  #
)

for process in ["wjets", "ttbar", "zjets", "singletop", "diboson"]:
    setting.set_systematics(process, ["purw_sys"])

# selections
# NB: ABCD plane for muons needs to be in the met > 25 GeV phase-space
lepton_selections = {
    "electron": ["AnalysisType==1"],
    "muon": ["AnalysisType==2 && met > 25"],
}

phasespace_selections = {
    "inclusive": [],
    # "collinear": ["DeltaPhiWJetClosest30<=1.5"],
    # "backtoback": ["DeltaPhiWJetClosest30>1.5"],
}
truth_selection = [
    "isTruth",
    "nTruthLeptons==1",
    "nTruthBJet25==0",
    "lep1TruthPt>30.0",
    "jet1TruthPt>500.0",
    "metTruth>25.0",
]
reco_selection = [
    "isReco",
    "nLeptons==1",
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
            # note there're several muon type in the correction file
            # we can clean up the correction file to have only one correction type
            corr_type = "muon_IpIso"

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
abcd.reserve_abcd_regions(
    setting, "fake-muon", ("lep1Signal", 1), ("IsoFCLoose_FixedRad", 1)
)
abcd.reserve_abcd_regions(
    setting,
    "fake-electron",
    ("met", 25),
    ("ptcone20_TightTTVALooseCone_pt500/lep1Pt", 0.06),
    reverse_y=True,
)

abcd.create_abcd_regions(
    setting,
    ["fake-muon"],
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
    setting, "lep1Pt", "lep1TruthPt", 10, 30, 530, "Leading lepton p_{T} [GeV]"
)
unfolding.add_observables(setting, "met", "metTruth", 20, 25, 525, "E_{T}^{miss} [GeV]")
unfolding.add_observables(
    setting, "jet1Pt", "jet1TruthPt", 15, 0, 1500, "Leading jet p_{T} [GeV]"
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
    setting, "nJet30", "nTruthJet30", 9, 1, 10, "Number of jets (p_{T} > 30 GeV)"
)
unfolding.add_observables(
    setting, "nJet30", "nTruthJet30", 9, 1, 10, "Number of jets (p_{T} > 30 GeV)"
)
unfolding.add_observables(
    setting,
    "nBJet30",
    "nTruthBJet30",
    1,
    0,
    20,
    "Inclusive observable [number of b-jets (p_{T} > 30 GeV)]",
)

unfolding.add_observables(
    setting,
    "DeltaPhiWJetClosest30",
    "DeltaPhiTruthWJetClosest30",
    16,
    0,
    3.2,
    "min(#Delta#phi(W,jet_{i}^{30}))",
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

setting.save("before_histmaker_run2.pkl")
mc16a_setting = run_HistMaker.run_HistMaker(setting)  # , rsplit=True)
mc16a_setting.save("run2.pkl")
