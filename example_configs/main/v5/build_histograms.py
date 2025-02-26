from collinearw import ConfigMgr
from collinearw import run_HistMaker
from collinearw.strategies import abcd, unfolding
from collinearw.serialization import Serialization
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


def main():
    setting = ConfigMgr(
        src_path="/gpfs/slac/atlas/fs1/d/yuzhan/collinearw_files/Wj_AB212164_v5/merged_files/",
        # out_path="./unfoldTest_v3_fullrun2/",  # output
        description="unfolding",
        io_backend="uproot",  # io backend
        histogram_backend="numpy",  # histograms backend
    )

    output_path = pathlib.Path(os.path.realpath(__file__)).parent
    setting.set_output_location(output_path)
    #
    setting.set_singlefile(
        [
            "mc16a_weights.root",
            "mc16a_kinematics_1_EG.root",
            "mc16a_kinematics_1_MUON.root",
            "mc16a_kinematics_2_JET.root",
            "mc16a_kinematics_2_MET.root",
            "mc16a_kinematics_3_JET_GroupedNP.root",
            "mc16a_kinematics_3_JET_JER_down.root",
            "mc16a_kinematics_3_JET_JER_up.root",
            "mc16d_weights.root",
            "mc16d_kinematics_1_EG.root",
            "mc16d_kinematics_1_MUON.root",
            "mc16d_kinematics_2_JET.root",
            "mc16d_kinematics_2_MET.root",
            "mc16d_kinematics_3_JET_GroupedNP.root",
            "mc16d_kinematics_3_JET_JER_down.root",
            "mc16d_kinematics_3_JET_JER_up.root",
            # "mc16e_weights.root",
            # "mc16e_kinematics_1_EG.root",
            # "mc16e_kinematics_1_MUON_up.root",
            # "mc16e_kinematics_1_MUON_down.root",
            # "mc16e_kinematics_2_JET.root",
            # "mc16e_kinematics_2_MET.root",
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
        "wjets_2211",
        # "wjets_NoSys",
        process_type="signal",
        color=600,
        markerstyle=8,
        legendname="W+jets Sherpa 2.2.11",
        # filename=f"{mc16_path}/wjets_merged_processed.root",
    )  # blue
    setting.add_process(
        "wjets",
        # "wjets_mg_NoSys",
        process_type="signal_alt",
        color=632,
        markerstyle=22,
        legendname="W+jets Sherpa 2.2.1",
    )  # red
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
    )  # green-5

    setting.add_process(
        "ttbar",
        # "ttbar_NoSys",
        process_type="bkg",
        color=859,
        # filename=f"{mc16_path}/ttbar_merged_processed.root",
    )  # azure-1

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
        # "AbsEtaLess2p47": ["TMath::Abs(lep1Eta) < 2.47"],
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
        "((ptcone20_TightTTVALooseCone_pt1000/lep1Pt<0.06 && lep1Topoetcone20/lep1Pt < 0.06 && AnalysisType==1) || (IsoLoose_FixedRad==1 && AnalysisType==2))",
    ]

    # weights we need for each
    weight_reco = "genWeight*eventWeight*bTagWeight*pileupWeight*leptonWeight*jvtWeight*triggerWeight"
    weight_truth = "genWeight*eventWeight"

    filter_region_hist_types = {
        "truth": {"reco", "response", "tf"},
        "reco": {"truth", "response"},
        "reco_match": {"tf"},
        "reco_match_not": {"truth", "response", "tf"},
        "abcd": {"truth", "response"},
    }

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
                matching_selection=truth_matching[phasespace] + isolation_selection,
                notmatching_selection=truth_notmatching[phasespace]
                + isolation_selection,
                weight_truth=weight_truth,
                weight_reco=weight_reco,
                corr_type=corr_type,
                filter_region_hist_types=filter_region_hist_types,
            )

    abcd.reserve_abcd_regions(setting, "PID", ("met", 25), ("lep1Signal", 1))
    abcd.reserve_abcd_regions(
        setting, "fake-muon", ("lep1Signal", 1), ("IsoLoose_FixedRad", 1)
    )
    abcd.reserve_abcd_regions(
        setting,
        "fake-electron",
        ("met", 25),
        # ("IsoTight_VarRad", 1),
        ("ptcone20_TightTTVALooseCone_pt1000/lep1Pt", 0.06),
        reverse_y=True,
    )

    abcd.reserve_abcd_regions(
        setting,
        "et-cone",
        ("met", 25),
        ("lep1Topoetcone20/lep1Pt", 0.06),
        reverse_y=True,
    )

    abcd.create_abcd_regions(
        setting,
        ["fake-muon"],
        tag_name="fake-MU",
        base_region_names=[
            region.name
            for region in setting.regions
            if 'muon' in region.name
            and region.type not in ["truth", "reco_match", "reco_match_not"]
        ],
    )
    abcd.create_abcd_regions(
        setting,
        ["PID", "fake-electron", "et-cone"],
        tag_name="fake-EL",
        base_region_names=[
            region.name
            for region in setting.regions
            if 'electron' in region.name
            and region.type not in ["truth", "reco_match", "reco_match_not"]
        ],
    )

    setting.add_observable("lep1Phi", 15, -3, 3, "lepton #phi")
    unfolding.add_observables(
        setting, "jet2Pt", "jet2TruthPt", 10, 0, 500, "Sub-leading jet p_{T} [GeV]"
    )
    unfolding.add_observables(
        setting, "lep1Pt", "lep1TruthPt", 10, 30, 530, "Leading lepton p_{T} [GeV]"
    )
    # unfolding.add_observables(setting, "TMath::Abs(lep1Eta)", "TMath::Abs(lep1TruthEta)", 25, 0, 2.5, "Abs(lepton p_{eta})")
    unfolding.add_observables(
        setting, "lep1Eta", "lep1TruthEta", 50, -2.5, 2.5, "lepton #eta"
    )
    # .add_observables(setting, "lep1Phi", "lep1TruthPhi", 15, -3, 3, "lepton phi")
    unfolding.add_observables(
        setting, "nJet30", "nTruthJet30", 9, 1, 10, "Number of jets (p_{T} > 30 GeV)"
    )
    unfolding.add_observables(
        setting, "Ht30", "HtTruth30", 25, 500, 3000, "H_{T} [GeV]"
    )
    unfolding.add_observables(
        setting, "wPt", "wTruthPt", 15, 0, 1500, "Leading W p_{T} [GeV]"
    )
    unfolding.add_observables(
        setting,
        "mjj",
        "mjjTruth",
        20,
        0,
        2000,
        "Leading and sub-leading jet mass [GeV]",
    )
    unfolding.add_observables(
        setting, "met", "metTruth", 10, 25, 550, "E_{T}^{miss} [GeV]"
    )
    jetPt_var_bin = [500, 600, 700, 800, 900, 1000, 1200, 1500, 2000, 2500]
    unfolding.add_observables(
        setting, "jet1Pt", "jet1TruthPt", jetPt_var_bin, 1, 1, "Leading jet p_{T} [GeV]"
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
        xbin=[0.0, 1.05, 2.5],
        ybin=[30, 100, 150, 200, 300, 400, 600, 800, 1000, 1500, 2000, 2500],
        xtitle="eta (2d)",
        ytitle="Lepton pT (2d)",
        type="reco",
    )

    setting.add_histogram2D(
        "eta_vs_lepPt_muon",
        "TMath::Abs(lep1Eta)",
        "lep1Pt",
        xbin=[0.0, 1.05, 2.5],
        ybin=[30, 100, 200, 300, 400, 500, 800, 2500],
        xtitle="eta (2d)",
        ytitle="Lepton pT (2d)",
        type="reco",
    )

    # Phase-space corrections, but only load the non signal processes (e.g zjets, ttbar)
    # the correction format is a python dict with key as the form (correction type, process name, observable)
    # unfortunatly the correction are derived separetely for each generator, if you want to have single
    # correction file for different wjets generator, you might need to merge and rename the key yourself.
    bkgd_correction = "./iterative/run2_wjets_2211_bkgd_correction.shelf"
    setting.corrections.add_correction_file(bkgd_correction)

    # Note: no need to create dummy and fill if nominal phase-space correction is applied for all
    # dummy can be created later after everything is filled. see create_dummy_systematics()
    setting.load_systematics(
        "minimum_sys.jsonnet",
        # create_dummy=["wjets_2211", "zjets_2211", "ttbar", "diboson", "singletop"],
    )
    setting.prepare(True)

    setting.save("before_histmaker_run2_2211.pkl")

    workers = 200
    workers = 8 if workers < 8 else workers
    cores = 8
    account = "shared"
    cluster = SLURMCluster(
        queue=account,
        walltime='05:00:00',
        project="collinear_Wjets",
        cores=1,
        processes=1,
        memory="32GB",
        job_extra=[
            f'--account={account}',
            f'--partition={account}',
            f'--cpus-per-task={cores}',
        ],
        local_directory="dask_output",
        log_directory="dask_logs",
        # death_timeout=36000,
        n_workers=workers,
        # nanny=False,
    )
    print(cluster.job_script())
    client = Client(cluster)
    client.get_versions(check=True)
    # client.wait_for_workers(workers)
    # print(f"{workers=}")

    setting.RAISE_TREENAME_ERROR = False

    mc16a_setting = run_HistMaker.run_HistMaker(
        setting,
        split_type="region",
        merge_buffer_size=50,
        executor=client,
        as_completed=dask_as_completed,
    )
    mc16a_setting.corrections.clear_buffer()

    # create dummy systematics
    mc16a_setting.create_dummy_systematics(
        ["wjets_2211", "zjets_2211", "ttbar", "diboson", "singletop"]
    )

    mc16a_setting.save("run2_2211.pkl")


if __name__ == "__main__":
    main()
