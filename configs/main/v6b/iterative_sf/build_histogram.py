"""
Histogram ConfigMgr builder for Production Wj_AB212164_v4.
Use for deriving phase-space correction factors.
"""

from collinearw.serialization import Serialization
from collinearw import ConfigMgr
from collinearw import run_HistMaker
from collinearw.strategies import abcd
import pathlib
import ROOT
import logging
import os
import glob
import numpy as np
from dask_jobqueue import SLURMCluster, LSFCluster
from dask.distributed import Client
from dask.distributed import as_completed as dask_as_completed

logging.basicConfig()
logging.getLogger().setLevel(logging.WARNING)
log = logging.getLogger(__name__)

cluster_type = "lll" # "slurm"

workers = 25
workers = 8 if workers < 8 else workers
cores = 32
account = "shared"
if cluster_type == "slurm":
    cluster = SLURMCluster(
        queue=account,
        walltime='02:00:00',
        account="collinear_Wjets",
        cores=cores,
        # cores=cores,
        processes=1,
        memory="64GB",
        # job_extra_directives=[f'--account={account}', f'--partition={account}', f'--cpus-per-task={cores}'],
        job_extra_directives=[f'--account={account}', f'--partition={account}'],
        # local_directory="dask_output",
        log_directory="dask_logs",
        #death_timeout=36000,
        # n_workers=workers,
        nanny=False,
    )
    cluster.scale(jobs=workers)
    print(cluster.job_script())
    client = Client(cluster)
    client.get_versions(check=True)
elif cluster_type == "lsf":
    cluster = LSFCluster(
        # queue="centos7",
        project="collinear_Wjets",
        cores=cores,
        memory="64GB",
        processes=1,
        #nanny=False,
        walltime="1:30",
        n_workers=workers,
        #job_extra=[f"-R centos7"],
        local_directory="lsf_output",
        log_directory="lsf_logs",
        death_timeout=300,
    )
    # cluster.scale(workers)
    print(cluster.job_script())
    #breakpoint()
    client = Client(cluster)
    client.get_versions(check=True)
else:
    client = None

def main():
    setting = ConfigMgr()
    output_path = pathlib.Path(os.path.realpath(__file__)).parent
    setting.set_output_location(output_path)

    src_path = "/gpfs/slac/atlas/fs1/d/yuzhan/collinearw_files/Wj_AB212221_v6b/merged_files/"
    alt_src_path = "/gpfs/slac/atlas/fs1/d/yuzhan/collinearw_files/Wj_AB212221_v6b_Wt_DS_wjet_EW_v2/merged_files/"


    # Common phase-space cuts
    common_cuts = "isReco && lep1Pt>30.0 && TMath::Abs(lep1Eta)<2.4 && trigMatch_singleLepTrig && jet1Pt>400.0"

    # name, treename, selection, process_type
    filename = glob.glob(f"{src_path}/mc16*/data*merged_processed*.root")
    setting.add_process(
        "data",
        process_type="data",
        color=1,
        legendname="Data",
        binerror=1,
        filename=filename,
    )  # black, kPoisson error

    filename = glob.glob(f"{src_path}/mc16*/*wjets_2211*merged_processed*.root")
    setting.add_process(
        "wjets_2211",
        treename="wjets_2211_NoSys",
        process_type="signal",
        color=600,
        markerstyle=8,
        legendname="W+jets (Sh 2.2.11)",
        #selection="isVgammaOverlap == 0",
        selection="(DatasetNumber < 700344 || DatasetNumber > 700349) && isVgammaOverlap == 0",
        filename=filename,
    )  # blue

    filename = glob.glob(f"{src_path}/mc16*/*wjets_FxFx*merged_processed*.root")
    setting.add_process(
        "wjets_tau",
        treename="wjets_FxFx_NoSys",
        process_type="bkg",
        color=632,
        markerstyle=8,
        legendname="#tau#nu+jets (FxFx)",
        selection="(DatasetNumber >= 509751 && DatasetNumber <= 509756) && isVgammaOverlap == 0",
        filename=filename,
    )  # blue

    filename = glob.glob(f"{src_path}/mc16*/*vgamma_sherpa2211*merged_processed*.root")
    setting.add_process(
        "vgamma",
        treename="vgamma_sherpa2211_NoSys",
        process_type="bkg",
        color=632,
        markerstyle=8,
        legendname="W+#gamma+jets (Sh 2.2.11)",
        #selection="isVgammaOverlap == 0",
        filename=filename,
    )  # blue


    filename = glob.glob(f"{src_path}/mc16*/*dijets*merged_processed*.root")
    setting.add_process(
        "dijets",
        treename="dijets_NoSys",
        process_type="bkg",
        color=859,
        legendname="Dijets",
        selection="eventWeight < 1",
        filename=filename,
    )  # azure-1

    filename = glob.glob(f"{src_path}/mc16*/*zjets_2211*merged_processed*.root")
    setting.add_process(
        "zjets_2211",
        process_type="bkg",
        color=861,
        legendname="Z+jets (Sh 2.2.11)",
        filename=filename,
        selection="isVgammaOverlap == 0",
    )  # azure+1

    filename = glob.glob(f"{src_path}/mc16*/*singletop*merged_processed*.root")
    setting.add_process(
        "singletop",
        process_type="bkg",
        color=800,
        legendname="Singletop (Powheg+Py8)",
        filename=filename,
    )  # orange

    filename = glob.glob(f"{alt_src_path}/mc16*/*wjets_EW_2211*merged_processed*.root")
    setting.add_process(
        "wjets_EW_2211",
        treename="wjets_EW_2211_NoSys",
        process_type="bkg",
        color=600,
        markerstyle=8,
        legendname="W+2jets EW(Sh 2.2.11)",
        selection="isVgammaOverlap == 0",
        #selection="(DatasetNumber < 700344 || DatasetNumber > 700349) && isVgammaOverlap==0",
        filename=filename,
    )  # blue

    # filename = glob.glob(f"{src_path}/mc16*/*singletop*merged_processed*.root")
    # setting.add_process(
    #     "singletop_Wt",
    #     treename="singletop_NoSys",
    #     process_type="bkg",
    #     color=800,
    #     legendname="Singletop, Wt(Powheg+Py8)",
    #     filename=filename,
    #     selection="(DatasetNumber == 410646 || DatasetNumber == 410647)",
    # )  # orange
    #
    # filename = glob.glob(f"{src_path}/mc16*/*singletop*merged_processed*.root")
    # setting.add_process(
    #     "singletop_stchan",
    #     treename="singletop_NoSys",
    #     process_type="bkg",
    #     color=800,
    #     legendname="Singletop, s+t ch.(Powheg+Py8)",
    #     filename=filename,
    #     selection="(DatasetNumber != 410646 && DatasetNumber != 410647)",
    # )  # orange

    filename = glob.glob(f"{src_path}/mc16*/*diboson_powheg*merged_processed*.root")
    setting.add_process(
        "diboson_powheg",
        process_type="bkg",
        color=411,
        legendname="Diboson (Powheg+Py8)",
        filename=filename,
    )  # green-5

    filename = glob.glob(f"{src_path}/mc16*/*ttbar*merged_processed*.root")
    setting.add_process(
        "ttbar",
        treename="ttbar_Sherpa2212_NoSys",
        process_type="bkg",
        color=859,
        legendname="ttbar (Sh 2.2.11)",
        filename=filename,
        # weights="LHE3Weight_MUR1_MUF1_PDF303200_ASSEW",
    )  # azure-1


    # Electron selections
    electron_1lepton = [
        "met>30.0",
        # "ptcone20_TightTTVALooseCone_pt1000/lep1Pt < 0.06",
        # "lep1Topoetcone20/lep1Pt < 0.06",
        "lep1Signal==1",
    ]
    electron_2lepton = [
        # "ptcone20_TightTTVALooseCone_pt1000/lep1Pt < 0.06",
        # "lep1Topoetcone20/lep1Pt < 0.06",
        # "lep2_ptcone20_TightTTVALooseCone_pt1000/lep2Pt < 0.06",
        # "lep2Topoetcone20/lep1Pt < 0.06",
        "lep1Signal==1",
        "lep2Signal==1",
        "TMath::Abs(lep2Eta)<2.4",
    ]

    # Muon selections
    muon_1lepton = [
        "met>30.0",
        # "IsoLoose_FixedRad==1",
        "lep1Signal==1",
    ]
    muon_2lepton = [
        # "IsoLoose_FixedRad==1",
        "lep1Signal==1",
        # "lep2_IsoLoose_FixedRad==1",
        "lep2Signal==1",
        "TMath::Abs(lep2Eta)<2.4",
    ]

    # Regions
    setting.add_region(
        "ttbarCR_Ele",
        f"{common_cuts} && nBJet30>=2 && nLeptons==1 && AnalysisType==1 && {'&&'.join(electron_1lepton)}",
        corr_type="electron",
    )
    setting.add_region(
        "ZjetsCR_Ele",
        f"{common_cuts} && nBJet30==0 && nLeptons==2 && AnalysisType==1 && diLeptonMass>60.0 && diLeptonMass<120.0 && {'&&'.join(electron_2lepton)}",
        corr_type="electron",
    )

    # dijetsCR_Ele_Sel = "lep1Topoetcone20/lep1Pt<0.06 && ptcone20_TightTTVALooseCone_pt1000/lep1Pt<0.06 && lep1Signal==0"
    setting.add_region(
        "dijetsCR_Ele",
        # f"{common_cuts} && nBJet30==0 && nLeptons==1 && AnalysisType==1 && {dijetsCR_Ele_Sel}",
        f"{common_cuts} && nBJet30==0 && nLeptons==1 && AnalysisType==1 && met>30.0 && lep1Signal==0",
        corr_type="electron",
    )
    setting.add_region(
        "dijetsCR_Ele_noISO",
        # f"{common_cuts} && nBJet30==0 && nLeptons==1 && AnalysisType==1 && {dijetsCR_Ele_Sel}",
        f"{common_cuts} && nBJet30==0 && nLeptons==1 && AnalysisType==1 && met>30.0",
        corr_type="electron",
    )
    setting.add_region(
        "dijetsCR_Ele_vali",
        f"{common_cuts} && nBJet30==0 && nLeptons==1 && AnalysisType==1 && lep1Signal==1 && DeltaRLepJetClosest100>2.6 && met<100",
        corr_type="electron",
    )

    setting.add_region(
        "electron",
        f"{common_cuts} && nBJet30==0 && nLeptons==1 && AnalysisType==1",
        corr_type="electron",
    )

    # ===============================================================================
    setting.add_region(
        "ttbarCR_Mu",
        f"{common_cuts} && nBJet30>=2 && nLeptons==1 && AnalysisType==2 && {'&&'.join(muon_1lepton)}",
        corr_type="muon",
    )
    setting.add_region(
        "ZjetsCR_Mu",
        f"{common_cuts} && nBJet30==0 && nLeptons==2 && AnalysisType==2 && diLeptonMass>60.0 && diLeptonMass<120.0 && {'&&'.join(muon_2lepton)}",
        corr_type="muon",
    )
    setting.add_region(
        "dijetsCR_Mu",
        # f"{common_cuts} && nBJet30==0 && nLeptons==1 && AnalysisType==2 && lep1Signal==1 && IsoLoose_FixedRad == 0",
        f"{common_cuts} && nBJet30==0 && nLeptons==1 && AnalysisType==2 && met>30.0 && lep1Signal==0",
        corr_type="muon",
    )
    setting.add_region(
        "dijetsCR_Mu_noISO",
        # f"{common_cuts} && nBJet30==0 && nLeptons==1 && AnalysisType==2 && lep1Signal==1 && IsoLoose_FixedRad == 0",
        f"{common_cuts} && nBJet30==0 && nLeptons==1 && AnalysisType==2 && met>30.0",
        corr_type="muon",
    )
    setting.add_region(
        "dijetsCR_Mu_vali",
        # f"{common_cuts} && nBJet30==0 && nLeptons==1 && AnalysisType==2 && lep1Signal==1 && IsoLoose_FixedRad == 0",
        f"{common_cuts} && nBJet30==0 && nLeptons==1 && AnalysisType==2 && lep1Signal==1 && DeltaRLepJetClosest100>2.6 && met<100",
        corr_type="muon",
    )
    setting.add_region(
        "muon",
        f"{common_cuts} && nBJet30==0 && nLeptons==1 && AnalysisType==2",
        corr_type="muon",
    )

    # ===============================================================================

    abcd.reserve_abcd_regions(setting, "met-pid", ("met", 30), ("lep1Signal", 1))
    abcd.reserve_abcd_regions(setting, "met-lepSignal", ("met", 30), ("lep1Signal", 1))
    abcd.reserve_abcd_regions(
        setting, "muon-met-iso", ("met", 30), ("IsoLoose_FixedRad", 1),
    )
    abcd.reserve_abcd_regions(
        setting,
        "met-ELTrack",
        ("met", 30),
        #("IsoTight_VarRad", 1),
        ("ptcone20_TightTTVALooseCone_pt1000/lep1Pt", 0.06),
        reverse_y=True,
    )

    abcd.reserve_abcd_regions(
        setting,
        "et-cone",
        ("met", 30),
        ("lep1Topoetcone20/lep1Pt", 0.06),
        reverse_y=True,
    )

    abcd.create_abcd_regions(
        setting,
        # ["met-pid", "met-ELTrack", "et-cone"],
        # ["met-pid"],
        ["met-lepSignal"],
        base_region_names=["electron"],
        rA_only=True,
    )

    abcd.create_abcd_regions(
        setting,
        # ["muon-met-iso"],
        ["met-lepSignal"],
        base_region_names=["muon"],
        rA_only=True,
    )

    setting.add_observable("lep1Phi", 15, -3, 3, "lepton #phi")

    njet_bins = [1.0,2.0,3.0,4.0,5.0,10.0]
    setting.add_observable("nJet30", njet_bins, 1, 1, "Jet multiplicity (p_{T}>30 GeV)")

    setting.add_observable("lep1Pt", 10, 30, 630, "Leading lepton p_{T} [GeV]")
    setting.add_observable("lep1Eta", 10, -2.5, 2.5, "Leading lepton #eta ")
    setting.add_observable("mt", 30, 0, 300, "m_{T} [GeV]")
    setting.add_observable("diLeptonPt", 20, 0, 1000, "Z p_{T} [GeV]")
    setting.add_observable("diLeptonMass", 20, 60, 120, "m_{ll} [GeV]")

    wpt_bins = list(np.arange(0, 800, 100)) + [800, 1000, 1500]
    setting.add_observable("wPt", wpt_bins, 1, 1, "Leading W p_{T} [GeV]")

    met_bins = [0, 50, 100, 150, 200, 500]
    setting.add_observable("met", met_bins, 1, 1, "E_{T}^{miss} [GeV]")

    ht_bins = list(np.arange(500, 1200, 100)) + list(np.arange(1200, 3000, 200))
    setting.add_observable("Ht30", ht_bins, 1, 1, "H_{T} [GeV]")

    mjj_bins = list(np.arange(0, 2000, 200)) + [2000, 2250, 2500,3000, 3500, 4000]
    setting.add_observable("mjj", mjj_bins, 1, 1, "Leading and sub-leading jet mass [GeV]")

    jetPt_var_bin = [500, 550, 600, 700, 800, 900, 1000, 1250, 1500]
    setting.add_observable("jet1Pt", jetPt_var_bin, 1, 1, "Leading jet p_{T} [GeV]")

    setting.add_observable(
        "DeltaPhiWJetClosest100", 16, 0, 3.2, "min(#Delta#phi(W,jet_{i}^{100}))"
    )

    setting.add_observable(
        "DeltaPhiMetJetClosest100", 16, 0, 3.2, "min(#Delta#phi(W,jet_{i}^{100}))"
    )

    setting.add_observable(
        "DeltaRLepJetClosest100", 25, 0, 5, "min(#DeltaR(lepton,jet_{i}^{100}))"
    )

    setting.add_histogram2D(
        "tf_eta_vs_lepPt_el",
        "TMath::Abs(lep1Eta)",
        "lep1Pt",
        xbin=[0.0, 1.05, 2.5],
        ybin=[30, 100, 150, 200, 300, 400, 600, 800, 1000, 1500, 2000, 2500],
        xtitle="eta (2d)",
        ytitle="l pT (2d)",
    )

    setting.add_histogram2D(
        "tf_eta_vs_lepPt_mu",
        "TMath::Abs(lep1Eta)",
        "lep1Pt",
        xbin=[0.0, 1.05, 2.5],
        ybin=[30, 100, 200, 300, 400, 500, 800, 2500],
        xtitle="eta (2d)",
        ytitle="l pT (2d)",
    )

    setting.add_observable(
        "lep1Pt_varbin-nominal_e",
        [30, 100, 150, 200, 300, 400, 600, 800, 1000, 1500, 2000, 2500],
        xtitle="leading lepton p_{T} [GeV]",
        observable="lep1Pt",
    )

    setting.add_observable(
        "lep1Pt_varbin-nominal_mu",
        [30, 100, 200, 300, 400, 500, 800, 2500],
        xtitle="leading lepton p_{T} [GeV]",
        observable="lep1Pt",
    )

    setting.add_observable(
        "lep1Eta_varbin-nominal_e",
        [0.0, 1.05, 2.5],
        xtitle="leading lepton #eta ",
        observable="TMath::Abs(lep1Eta)",
    )

    setting.add_observable(
        "lep1Eta_varbin-nominal_mu",
        [0.0, 1.05, 2.5],
        xtitle="leading lepton #eta ",
        observable="TMath::Abs(lep1Eta)",
    )

    setting.add_observable(
        "Ht30-jet1Pt",
        bins=[0,200,400,600,800,1000,1200,1400,2500],
        xtitle="Ht30-jet1Pt",
        observable="Ht30-jet1Pt",
        type="reco",
    )

    setting.add_histogram2D(
        "tf_muon",
        "(lep1Pt+jet2Pt)/(jet1Pt+jet2Pt)",
        "Ht30-jet1Pt-jet2Pt",
        xbin=[0.05, 0.25, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.9],
        ybin=[0,100,200,300,400,500,1500],
        xtitle="(lep1Pt+jet2Pt)/(jet1Pt+jet2Pt)(2d)",
        ytitle="Ht30-jet1Pt-jet2Pt(2d)",
        type="reco",
    )

    setting.process_region_type_filter = {
        "data" : {"reco_match", "reco_match_not"},
        "zjets_2211" : {"truth", "reco_match", "reco_match_not"},
        "ttbar" : {"truth", "reco_match", "reco_match_not"},
        "dijets" : {"truth", "reco_match", "reco_match_not"},
        "singletop" : {"truth", "reco_match", "reco_match_not"},
        "singletop_Wt" : {"truth", "reco_match", "reco_match_not"},
        "singletop_Wt_DS" : {"truth", "reco_match", "reco_match_not"},
        "singletop_stchan" : {"truth", "reco_match", "reco_match_not"},
        "diboson" : {"truth", "reco_match", "reco_match_not"},
        "vgamma" : {"truth", "reco_match", "reco_match_not"},
        "diboson_powheg" : {"truth", "reco_match", "reco_match_not"},
        "wjets_tau" : {"truth", "reco_match", "reco_match_not"},
        "wjets" : {"reco", "reco_match_not"},
        "wjets_FxFx" : {"reco", "reco_match_not"},
        "wjets_EW_2211" : {"truth", "reco_match", "reco_match_not"},
    }


    setting.load_systematics("../minimum_sys.jsonnet")#, create_dummy=["wjets_2211", "zjets_2211", "ttbar", "diboson", "singletop"])
    setting.enforce_default_weight = True

    setting.phasespace_corr_obs = "nJet30"

    # setting.prepare(True)

    setting.save("run2_raw_config")

    # flag for raising error when tree does not exist
    setting.RAISE_TREENAME_ERROR = False

    if client is None:
        setting = run_HistMaker.run_HistMaker(
            setting,
            split_type="region",
            filter_ttree=True,
            n_workers=20,
    )
    else:
        setting = run_HistMaker.run_HistMaker(
            setting,
            split_type="region",
            executor=client,
            as_completed=dask_as_completed,
            copy=False,
            #split_ifile=True,
            filter_ttree=True,
        )

    #setting.correcitons.clear_files()

    setting.save("run2.pkl")

if __name__ == "__main__":
    main()
