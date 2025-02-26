from collinearw import ConfigMgr
from collinearw import run_HistMaker
from collinearw.strategies import abcd, unfolding
from dask_jobqueue import SLURMCluster
from dask.distributed import Client
from dask.distributed import as_completed as dask_as_completed

import copy
import os
import pathlib
import logging

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
log = logging.getLogger(__name__)


#
setting = ConfigMgr(
    src_path="/gpfs/slac/atlas/fs1/d/yuzhan/collinearw_files/Wj_AB212164_v4/merged_files/",
    # out_path="./unfoldTest_v3_fullrun2/",  # output
    description="unfolding",
    io_backend="uproot",  # io backend
    histogram_backend="numpy",  # histograms backend
)

output_path = pathlib.Path(os.path.realpath(__file__)).parent
setting.set_output_location(output_path)


# Phase-space corrections, but only load the non signal processes (e.g zjets, ttbar)
# the correction format is a python dict with key as the form (correction type, process name, observable)
# unfortunatly the correction are derived separetely for each generator, if you want to have single
# correction file for different wjets generator, you might need to merge and rename the key yourself.
basedir = os.path.dirname(os.path.realpath(__file__))
bkgd_correction = "./iterative/run2_wjets_2211_bkgd_correction.pkl"
setting.corrections.add_correction_file(bkgd_correction)

#
setting.set_singlefile(
    [
        "mc16a_v4.root",
        # "mc16d_v4.root",
        # "mc16e_v4.root",
    ]
)

mc16_path = (
    "/gpfs/slac/atlas/fs1/d/yuzhan/collinearw_files/Wj_AB212164_v4/merged_files/"
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
    "wjets_2211",
    # "wjets_NoSys",
    process_type="signal",
    color=600,
    markerstyle=8,
    legendname="W+jets Sherpa 2.2.11",
    # filename=f"{mc16_path}/wjets_merged_processed.root",
)  # blue
setting.add_process(
    "wjets_mg",
    # "wjets_mg_NoSys",
    process_type="signal_alt",
    color=632,
    markerstyle=22,
    legendname="MadGraph+Pythia8 CKKW-L",
)  # red

setting.add_process(
    "ttbar",
    # "ttbar_NoSys",
    process_type="bkg",
    color=859,
    # filename=f"{mc16_path}/ttbar_merged_processed.root",
)  # azure-1
setting.add_process(
    "zjets_2211",
    # "zjets_NoSys",
    process_type="bkg",
    color=861,
    legendname="Z+jets 2.2.11",
    # filename=f"{mc16_path}/zjets_merged_processed.root",
)  # azure+1
setting.add_process(
    "singletop",
    # "singletop_NoSys",
    process_type="bkg",
    color=800,
    # filename=f"{mc16_path}/singletop_merged_processed.root",
)  # orange
setting.add_process(
    "diboson",
    # "diboson_NoSys",
    process_type="bkg",
    color=411,
    # filename=f"{mc16_path}/diboson_merged_processed.root",
)  # green-5

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
    "nTruthBJet30==0",
    "lep1TruthPt>30.0",
    "jet1TruthPt>500.0",
    "metTruth>25.0",
]
reco_selection = [
    "isReco",
    "nLeptons==1",
    "nBJet30==0",
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
    "((IsoTightTrackOnly_VarRad==1 && AnalysisType==1) || (IsoLoose_FixedRad==1 && AnalysisType==2))",
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
abcd.reserve_abcd_regions(
    setting, "fake-muon", ("lep1Signal", 1), ("IsoLoose_FixedRad", 1)
)
abcd.reserve_abcd_regions(
    setting,
    "fake-electron",
    ("met", 25),
    ("IsoTightTrackOnly_VarRad", 1),
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
    setting, "jet2Pt", "jet2TruthPt", 15, 0, 1500, "Sub-leading jet p_{T} [GeV]"
)
unfolding.add_observables(
    setting, "wPt", "wTruthPt", 15, 0, 1500, "Leading W p_{T} [GeV]"
)
unfolding.add_observables(setting, "Ht30", "HtTruth30", 25, 500, 3000, "H_{T} [GeV]")
unfolding.add_observables(
    setting, "mjj", "mjjTruth", 20, 0, 2000, "Leading and sub-leading jet mass [GeV]"
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
unfolding.add_observables(
    setting,
    "DeltaRLepJetClosest30",
    "DeltaRTruthLepJetClosest30",
    25,
    0,
    5,
    "min(#DeltaR(lepton,jet_{i}^{30}))",
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

setting.load_systematics(
    "minimum_sys.jsonnet", create_dummy=["wjets_2211", "zjets_2211", "ttbar"]
)

setting.save("before_histmaker_run2_2211.pkl")

workers = 40
account = "shared"
cluster = SLURMCluster(
    queue=account,
    walltime='01:00:00',
    project="collinear_Wjets",
    cores=16,
    memory="200 GB",
    job_extra=[f'--account={account}', f'--partition={account}'],
    local_directory="dask_output",
    log_directory="dask_logs",
    death_timeout=36000,
)
cluster.scale(jobs=workers)
client = Client(cluster)
client.get_versions(check=True)
print(cluster.job_script())

mc16a_setting = run_HistMaker.run_HistMaker(
    setting, split_type="region", executor=client, as_completed=dask_as_completed
)
mc16a_setting.corrections.clear_buffer()
mc16a_setting.save("run2_2211.pkl")
