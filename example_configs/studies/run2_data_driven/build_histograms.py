from collinearw import ConfigMgr
from collinearw import run_HistMaker
from collinearw.strategies import abcd, unfolding
from collinearw.serialization import Serialization
from dask_jobqueue import SLURMCluster
from dask.distributed import Client
from dask.distributed import as_completed as dask_as_completed

import numpy as np
import copy
import os
import pathlib
import logging

logging.basicConfig()
logging.getLogger().setLevel(logging.WARNING)
log = logging.getLogger(__name__)

def booking_control_regions(setting, filter_region_hist_types):
    common_cuts = "isReco && lep1Pt>30.0 && trigMatch_singleLepTrig && jet1Pt>=500.0"

    weight_reco = "genWeight*eventWeight*bTagWeight*pileupWeight*leptonWeight*jvtWeight*triggerWeight"

    electron_1lepton = [
        "met>25.0",
        "ptcone20_TightTTVALooseCone_pt1000/lep1Pt < 0.06",
        "lep1Topoetcone20/lep1Pt < 0.06",
        "lep1Signal==1",
    ]
    electron_2lepton = [
        "ptcone20_TightTTVALooseCone_pt1000/lep1Pt < 0.06",
        "lep1Topoetcone20/lep1Pt < 0.06",
        "lep2_ptcone20_TightTTVALooseCone_pt1000/lep2Pt < 0.06",
        "lep2Topoetcone20/lep1Pt < 0.06",
        "lep1Signal==1",
        "lep2Signal==1",
    ]
    muon_1lepton = [
        "met>25.0",
        "IsoLoose_FixedRad==1",
        "lep1Signal==1",
    ]
    muon_2lepton = [
        "IsoLoose_FixedRad==1",
        "lep1Signal==1",
        "lep2_IsoLoose_FixedRad==1",
        "lep2Signal==1",
    ]
    setting.add_region(
        "ttbarCR_Ele",
        f"{common_cuts} && nBJet30>=2 && nLeptons==1 && AnalysisType==1 && {'&&'.join(electron_1lepton)}",
        corr_type="electron",
        weight=weight_reco,
        study_type="reco",
        filter_hist_types=filter_region_hist_types,
    )
    setting.add_region(
        "ZjetsCR_Ele",
        f"{common_cuts} && nBJet30==0 && nLeptons==2 && AnalysisType==1 && diLeptonMass>60.0 && diLeptonMass<120.0 && {'&&'.join(electron_2lepton)}",
        corr_type="electron",
        weight=weight_reco,
        study_type="reco",
        filter_hist_types=filter_region_hist_types,
    )

    dijetsCR_Ele_Sel = "lep1Topoetcone20/lep1Pt<0.06 && ptcone20_TightTTVALooseCone_pt1000/lep1Pt<0.06 && lep1Signal==0"
    setting.add_region(
        "dijetsCR_Ele",
        f"{common_cuts} && nBJet30==0 && nLeptons==1 && AnalysisType==1 && met>25 && {dijetsCR_Ele_Sel}",
        corr_type="electron",
        weight=weight_reco,
        study_type="reco",
        filter_hist_types=filter_region_hist_types,
    )

    setting.add_region(
        "ttbarCR_Mu",
        f"{common_cuts} && nBJet30>=2 && nLeptons==1 && AnalysisType==2 && {'&&'.join(muon_1lepton)}",
        corr_type="muon",
        weight=weight_reco,
        study_type="reco",
        filter_hist_types=filter_region_hist_types,
    )
    setting.add_region(
        "ZjetsCR_Mu",
        f"{common_cuts} && nBJet30==0 && nLeptons==2 && AnalysisType==2 && diLeptonMass>60.0 && diLeptonMass<120.0 && {'&&'.join(muon_2lepton)}",
        corr_type="muon",
        weight=weight_reco,
        study_type="reco",
        filter_hist_types=filter_region_hist_types,
    )
    setting.add_region(
        "dijetsCR_Mu",
        f"{common_cuts} && nBJet30==0 && nLeptons==1 && AnalysisType==2 && lep1Signal==1 && IsoLoose_FixedRad == 0 && met>25",
        corr_type="muon",
        weight=weight_reco,
        study_type="reco",
        filter_hist_types=filter_region_hist_types,
    )

def booking_processes(setting):
    # name, treename, selection, process_type
    setting.add_process(
        "data",
        process_type="data",
        color=1,
        legendname="Data",
        binerror=1,
    )  # black, kPoisson error

    setting.add_process(
        "wjets_2211",
        process_type="signal",
        color=600,
        markerstyle=8,
        legendname="W+jets (Sh 2.2.11)",
        #selection="isVgammaOverlap == 0",
        selection="(DatasetNumber < 700344 || DatasetNumber > 700349) && isVgammaOverlap==0",
        filename=[
            #"/gpfs/slac/atlas/fs1/d/yuzhan/collinearw_files/Wj_AB212164_v5i/merged_files/test_wjets_2211_mc16a.root",
            "/gpfs/slac/atlas/fs1/d/yuzhan/collinearw_files/Wj_AB212164_v5i/merged_files/Wj_AB212164_v5i_mc16a_weights.root",
            "/gpfs/slac/atlas/fs1/d/yuzhan/collinearw_files/Wj_AB212164_v5i/merged_files/Wj_AB212164_v5i_mc16a_kinematics_1_wjets_2211.root",
            "/gpfs/slac/atlas/fs1/d/yuzhan/collinearw_files/Wj_AB212164_v5i/merged_files/Wj_AB212164_v5i_mc16a_kinematics_2_wjets_2211.root",
            "/gpfs/slac/atlas/fs1/d/yuzhan/collinearw_files/Wj_AB212164_v5i/merged_files/Wj_AB212164_v5i_mc16a_kinematics_3_wjets_2211.root",
            "/gpfs/slac/atlas/fs1/d/yuzhan/collinearw_files/Wj_AB212164_v5i/merged_files/Wj_AB212164_v5i_mc16d_weights.root",
            "/gpfs/slac/atlas/fs1/d/yuzhan/collinearw_files/Wj_AB212164_v5i/merged_files/Wj_AB212164_v5i_mc16d_kinematics_1_wjets_2211.root",
            "/gpfs/slac/atlas/fs1/d/yuzhan/collinearw_files/Wj_AB212164_v5i/merged_files/Wj_AB212164_v5i_mc16d_kinematics_2_wjets_2211.root",
            "/gpfs/slac/atlas/fs1/d/yuzhan/collinearw_files/Wj_AB212164_v5i/merged_files/Wj_AB212164_v5i_mc16d_kinematics_3_wjets_2211.root",
            "/gpfs/slac/atlas/fs1/d/yuzhan/collinearw_files/Wj_AB212164_v5i/merged_files/Wj_AB212164_v5i_mc16e_weights.root",
            "/gpfs/slac/atlas/fs1/d/yuzhan/collinearw_files/Wj_AB212164_v5i/merged_files/Wj_AB212164_v5i_mc16e_kinematics_1_wjets_2211.root",
            "/gpfs/slac/atlas/fs1/d/yuzhan/collinearw_files/Wj_AB212164_v5i/merged_files/Wj_AB212164_v5i_mc16e_kinematics_2_wjets_2211.root",
            "/gpfs/slac/atlas/fs1/d/yuzhan/collinearw_files/Wj_AB212164_v5i/merged_files/Wj_AB212164_v5i_mc16e_kinematics_3_wjets_2211.root",
        ],
    )  # blue

    setting.add_process(
        "wjets_FxFx",
        #treename="wjets_2211_NoSys",
        process_type="signal_alt",
        color=900,
        legendname="W+jets (FxFx)",
        markerstyle=20,
        #selection="isVgammaOverlap == 0",
        filename=[
            "/gpfs/slac/atlas/fs1/d/yuzhan/collinearw_files/Wj_AB212164_v5i/merged_files/Wj_AB212164_v5i_mc16a_weights_wjets_FxFx.root",
            "/gpfs/slac/atlas/fs1/d/yuzhan/collinearw_files/Wj_AB212164_v5i/merged_files/Wj_AB212164_v5i_mc16d_weights_wjets_FxFx.root",
            "/gpfs/slac/atlas/fs1/d/yuzhan/collinearw_files/Wj_AB212164_v5i/merged_files/Wj_AB212164_v5i_mc16e_weights_wjets_FxFx.root",
            # "/gpfs/slac/atlas/fs1/d/yuzhan/collinearw_files/Wj_AB212164_v5i/merged_files/Wj_AB212164_v5i_mc16a_weights.root",
            # "/gpfs/slac/atlas/fs1/d/yuzhan/collinearw_files/Wj_AB212164_v5i/merged_files/Wj_AB212164_v5i_mc16a_kinematics_1_wjets_FxFx.root",
            # "/gpfs/slac/atlas/fs1/d/yuzhan/collinearw_files/Wj_AB212164_v5i/merged_files/Wj_AB212164_v5i_mc16a_kinematics_2_wjets_FxFx.root",
            # "/gpfs/slac/atlas/fs1/d/yuzhan/collinearw_files/Wj_AB212164_v5i/merged_files/Wj_AB212164_v5i_mc16a_kinematics_3_wjets_FxFx.root",
            # "/gpfs/slac/atlas/fs1/d/yuzhan/collinearw_files/Wj_AB212164_v5i/merged_files/Wj_AB212164_v5i_mc16d_weights.root",
            # "/gpfs/slac/atlas/fs1/d/yuzhan/collinearw_files/Wj_AB212164_v5i/merged_files/Wj_AB212164_v5i_mc16d_kinematics_1_wjets_FxFx.root",
            # "/gpfs/slac/atlas/fs1/d/yuzhan/collinearw_files/Wj_AB212164_v5i/merged_files/Wj_AB212164_v5i_mc16d_kinematics_2_wjets_FxFx.root",
            # "/gpfs/slac/atlas/fs1/d/yuzhan/collinearw_files/Wj_AB212164_v5i/merged_files/Wj_AB212164_v5i_mc16d_kinematics_3_wjets_FxFx.root",
            # "/gpfs/slac/atlas/fs1/d/yuzhan/collinearw_files/Wj_AB212164_v5i/merged_files/Wj_AB212164_v5i_mc16e_weights.root",
            # "/gpfs/slac/atlas/fs1/d/yuzhan/collinearw_files/Wj_AB212164_v5i/merged_files/Wj_AB212164_v5i_mc16e_kinematics_1_wjets_FxFx.root",
            # "/gpfs/slac/atlas/fs1/d/yuzhan/collinearw_files/Wj_AB212164_v5i/merged_files/Wj_AB212164_v5i_mc16e_kinematics_2_wjets_FxFx.root",
            # "/gpfs/slac/atlas/fs1/d/yuzhan/collinearw_files/Wj_AB212164_v5i/merged_files/Wj_AB212164_v5i_mc16e_kinematics_3_wjets_FxFx.root",
        ],
    )

    setting.add_process(
        "wjets",
        process_type="signal_alt",
        color=632,
        markerstyle=22,
        legendname="W+jets (Sh 2.2.1)",
        filename=[
            "/gpfs/slac/atlas/fs1/d/yuzhan/collinearw_files/Wj_AB212164_v5i/merged_files/Wj_AB212164_v5i_mc16a_weights_wjets.root",
            "/gpfs/slac/atlas/fs1/d/yuzhan/collinearw_files/Wj_AB212164_v5i/merged_files/Wj_AB212164_v5i_mc16d_weights_wjets.root",
            "/gpfs/slac/atlas/fs1/d/yuzhan/collinearw_files/Wj_AB212164_v5i/merged_files/Wj_AB212164_v5i_mc16e_weights_wjets.root",
        ],
    )  # red

    setting.add_process(
        "wjets_2211_tau",
        treename="wjets_2211_NoSys",
        process_type="bkg",
        color=632,
        markerstyle=8,
        legendname="#tau#nu+jets (Sh 2.2.11)",
        selection="(DatasetNumber >= 700344 && DatasetNumber <= 700349) && isVgammaOverlap==0",
        filename=[
            "/gpfs/slac/atlas/fs1/d/yuzhan/collinearw_files/Wj_AB212164_v5i/merged_files/Wj_AB212164_v5i_mc16a_weights.root",
            "/gpfs/slac/atlas/fs1/d/yuzhan/collinearw_files/Wj_AB212164_v5i/merged_files/Wj_AB212164_v5i_mc16a_kinematics_1_wjets_2211.root",
            "/gpfs/slac/atlas/fs1/d/yuzhan/collinearw_files/Wj_AB212164_v5i/merged_files/Wj_AB212164_v5i_mc16a_kinematics_2_wjets_2211.root",
            "/gpfs/slac/atlas/fs1/d/yuzhan/collinearw_files/Wj_AB212164_v5i/merged_files/Wj_AB212164_v5i_mc16a_kinematics_3_wjets_2211.root",
            "/gpfs/slac/atlas/fs1/d/yuzhan/collinearw_files/Wj_AB212164_v5i/merged_files/Wj_AB212164_v5i_mc16d_weights.root",
            "/gpfs/slac/atlas/fs1/d/yuzhan/collinearw_files/Wj_AB212164_v5i/merged_files/Wj_AB212164_v5i_mc16d_kinematics_1_wjets_2211.root",
            "/gpfs/slac/atlas/fs1/d/yuzhan/collinearw_files/Wj_AB212164_v5i/merged_files/Wj_AB212164_v5i_mc16d_kinematics_2_wjets_2211.root",
            "/gpfs/slac/atlas/fs1/d/yuzhan/collinearw_files/Wj_AB212164_v5i/merged_files/Wj_AB212164_v5i_mc16d_kinematics_3_wjets_2211.root",
            "/gpfs/slac/atlas/fs1/d/yuzhan/collinearw_files/Wj_AB212164_v5i/merged_files/Wj_AB212164_v5i_mc16e_weights.root",
            "/gpfs/slac/atlas/fs1/d/yuzhan/collinearw_files/Wj_AB212164_v5i/merged_files/Wj_AB212164_v5i_mc16e_kinematics_1_wjets_2211.root",
            "/gpfs/slac/atlas/fs1/d/yuzhan/collinearw_files/Wj_AB212164_v5i/merged_files/Wj_AB212164_v5i_mc16e_kinematics_2_wjets_2211.root",
            "/gpfs/slac/atlas/fs1/d/yuzhan/collinearw_files/Wj_AB212164_v5i/merged_files/Wj_AB212164_v5i_mc16e_kinematics_3_wjets_2211.root",
        ],
    )  # blue

    setting.add_process(
        "dijets",
        process_type="bkg",
        color=859,
        legendname="Dijets",
        selection="eventWeight < 1",
        filename=[
            "/gpfs/slac/atlas/fs1/d/yuzhan/collinearw_files/Wj_AB212164_v5i/merged_files/Wj_AB212164_v5i_mc16a_weights.root",
            "/gpfs/slac/atlas/fs1/d/yuzhan/collinearw_files/Wj_AB212164_v5i/merged_files/Wj_AB212164_v5i_mc16a_kinematics_1_dijets.root",
            "/gpfs/slac/atlas/fs1/d/yuzhan/collinearw_files/Wj_AB212164_v5i/merged_files/Wj_AB212164_v5i_mc16a_kinematics_2_dijets.root",
            "/gpfs/slac/atlas/fs1/d/yuzhan/collinearw_files/Wj_AB212164_v5i/merged_files/Wj_AB212164_v5i_mc16a_kinematics_3_dijets.root",
            "/gpfs/slac/atlas/fs1/d/yuzhan/collinearw_files/Wj_AB212164_v5i/merged_files/Wj_AB212164_v5i_mc16d_weights.root",
            "/gpfs/slac/atlas/fs1/d/yuzhan/collinearw_files/Wj_AB212164_v5i/merged_files/Wj_AB212164_v5i_mc16d_kinematics_1_dijets.root",
            "/gpfs/slac/atlas/fs1/d/yuzhan/collinearw_files/Wj_AB212164_v5i/merged_files/Wj_AB212164_v5i_mc16d_kinematics_2_dijets.root",
            "/gpfs/slac/atlas/fs1/d/yuzhan/collinearw_files/Wj_AB212164_v5i/merged_files/Wj_AB212164_v5i_mc16d_kinematics_3_dijets.root",
            "/gpfs/slac/atlas/fs1/d/yuzhan/collinearw_files/Wj_AB212164_v5i/merged_files/Wj_AB212164_v5i_mc16e_weights.root",
            "/gpfs/slac/atlas/fs1/d/yuzhan/collinearw_files/Wj_AB212164_v5i/merged_files/Wj_AB212164_v5i_mc16e_kinematics_1_dijets.root",
            "/gpfs/slac/atlas/fs1/d/yuzhan/collinearw_files/Wj_AB212164_v5i/merged_files/Wj_AB212164_v5i_mc16e_kinematics_2_dijets.root",
            "/gpfs/slac/atlas/fs1/d/yuzhan/collinearw_files/Wj_AB212164_v5i/merged_files/Wj_AB212164_v5i_mc16e_kinematics_3_dijets.root",
        ],
    )  # azure-1

    setting.add_process(
        "vgamma",
        treename="vgamma_sherpa2211_NoSys",
        process_type="bkg",
        color=632,
        legendname="W+#gamma+jets (Sh 2.2.11)",
        #selection="isVgammaOverlap == 0",
        filename=[
            "/gpfs/slac/atlas/fs1/d/yuzhan/collinearw_files/Wj_AB212164_v5i/merged_files/Wj_AB212164_v5i_mc16a_weights.root",
            "/gpfs/slac/atlas/fs1/d/yuzhan/collinearw_files/Wj_AB212164_v5i/merged_files/Wj_AB212164_v5i_mc16a_kinematics_1_dijets.root",
            "/gpfs/slac/atlas/fs1/d/yuzhan/collinearw_files/Wj_AB212164_v5i/merged_files/Wj_AB212164_v5i_mc16a_kinematics_2_dijets.root",
            "/gpfs/slac/atlas/fs1/d/yuzhan/collinearw_files/Wj_AB212164_v5i/merged_files/Wj_AB212164_v5i_mc16a_kinematics_3_dijets.root",
            "/gpfs/slac/atlas/fs1/d/yuzhan/collinearw_files/Wj_AB212164_v5i/merged_files/Wj_AB212164_v5i_mc16d_weights.root",
            "/gpfs/slac/atlas/fs1/d/yuzhan/collinearw_files/Wj_AB212164_v5i/merged_files/Wj_AB212164_v5i_mc16d_kinematics_1_dijets.root",
            "/gpfs/slac/atlas/fs1/d/yuzhan/collinearw_files/Wj_AB212164_v5i/merged_files/Wj_AB212164_v5i_mc16d_kinematics_2_dijets.root",
            "/gpfs/slac/atlas/fs1/d/yuzhan/collinearw_files/Wj_AB212164_v5i/merged_files/Wj_AB212164_v5i_mc16d_kinematics_3_dijets.root",
            "/gpfs/slac/atlas/fs1/d/yuzhan/collinearw_files/Wj_AB212164_v5i/merged_files/Wj_AB212164_v5i_mc16e_weights.root",
            "/gpfs/slac/atlas/fs1/d/yuzhan/collinearw_files/Wj_AB212164_v5i/merged_files/Wj_AB212164_v5i_mc16e_kinematics_1_dijets.root",
            "/gpfs/slac/atlas/fs1/d/yuzhan/collinearw_files/Wj_AB212164_v5i/merged_files/Wj_AB212164_v5i_mc16e_kinematics_2_dijets.root",
            "/gpfs/slac/atlas/fs1/d/yuzhan/collinearw_files/Wj_AB212164_v5i/merged_files/Wj_AB212164_v5i_mc16e_kinematics_3_dijets.root",
        ],
    )  # azure-1

    setting.add_process(
        "zjets_2211",
        process_type="bkg",
        color=861,
        legendname="Z+jets (Sh 2.2.11)",
    )  # azure+1

    setting.add_process(
        "ttbar",
        process_type="bkg",
        color=859,
        legendname="ttbar (Powheg+Py8)"
    )  # azure-1

    setting.add_process(
        "singletop",
        process_type="bkg",
        color=800,
        legendname="Singletop (Powheg+Py8)",
    )  # orange

    setting.add_process(
        "diboson_powheg",
        process_type="bkg",
        color=411,
        legendname="Diboson (Powheg+Py8)",
    )  # green-5

    setting.add_process(
        "wjets_2211_ASSEW",
        treename="wjets_2211_NoSys",
        process_type="signal_alt",
        color=607,
        legendname="W+jets 2.2.11 (ASSEW)",
        weights="LHE3Weight_MUR1_MUF1_PDF303200_ASSEW",
        selection="(DatasetNumber < 700344 || DatasetNumber > 700349) && isVgammaOverlap==0",
        filename=[
            "/gpfs/slac/atlas/fs1/d/yuzhan/collinearw_files/Wj_AB212164_v5i/merged_files/Wj_AB212164_v5i_mc16a_weights.root",
            "/gpfs/slac/atlas/fs1/d/yuzhan/collinearw_files/Wj_AB212164_v5i/merged_files/Wj_AB212164_v5i_mc16d_weights.root",
            "/gpfs/slac/atlas/fs1/d/yuzhan/collinearw_files/Wj_AB212164_v5i/merged_files/Wj_AB212164_v5i_mc16e_weights.root",
        ],
    )

    '''
    setting.add_process(
        "diboson",
        process_type="bkg",
        color=411,
        legendname="Diboson (Sh 2.2.2)",
    )  # green-5

    setting.add_process(
        "wjets_2211_EXPASSEW",
        treename="wjets_2211_NoSys",
        process_type="signal_alt",
        color=800,
        legendname="W+jets 2.2.11 (EXPASSEW)",
        weights="LHE3Weight_MUR1_MUF1_PDF303200_EXPASSEW",
        #selection="isVgammaOverlap == 0",
        filename=[
            "/gpfs/slac/atlas/fs1/d/yuzhan/collinearw_files/Wj_AB212164_v5f_vgammaOR/merged_files/Wj_AB212164_v5f_vgammaOR_mc16a.root",
            "/gpfs/slac/atlas/fs1/d/yuzhan/collinearw_files/Wj_AB212164_v5f_vgammaOR/merged_files/Wj_AB212164_v5f_vgammaOR_mc16d.root",
            "/gpfs/slac/atlas/fs1/d/yuzhan/collinearw_files/Wj_AB212164_v5f_vgammaOR/merged_files/Wj_AB212164_v5f_vgammaOR_mc16e.root",
        ],
    )
    setting.add_process(
        "wjets_2211_MULTIASSEW",
        treename="wjets_2211_NoSys",
        process_type="signal_alt",
        color=900,
        legendname="W+jets 2.2.11 (MULTIASSEW)",
        weights="LHE3Weight_MUR1_MUF1_PDF303200_MULTIASSEW",
        #selection="isVgammaOverlap == 0",
        filename=[
            "/gpfs/slac/atlas/fs1/d/yuzhan/collinearw_files/Wj_AB212164_v5f_vgammaOR/merged_files/Wj_AB212164_v5f_vgammaOR_mc16a.root",
            "/gpfs/slac/atlas/fs1/d/yuzhan/collinearw_files/Wj_AB212164_v5f_vgammaOR/merged_files/Wj_AB212164_v5f_vgammaOR_mc16d.root",
            "/gpfs/slac/atlas/fs1/d/yuzhan/collinearw_files/Wj_AB212164_v5f_vgammaOR/merged_files/Wj_AB212164_v5f_vgammaOR_mc16e.root",
        ],
    )

    '''

syst_files_mc16a = [
    "mc16a_kinematics_1_EG.root",
    "mc16a_kinematics_1_MUON.root",
    "mc16a_kinematics_2_JET.root",
    "mc16a_kinematics_2_MET.root",
    "mc16a_kinematics_3_JET_GroupedNP.root",
    "mc16a_kinematics_3_JET_JER_down.root",
    "mc16a_kinematics_3_JET_JER_up.root",
]

syst_files_mc16d = [
    "mc16d_kinematics_1_EG.root",
    "mc16d_kinematics_1_MUON.root",
    "mc16d_kinematics_2_JET.root",
    "mc16d_kinematics_2_MET.root",
    "mc16d_kinematics_3_JET_GroupedNP.root",
    "mc16d_kinematics_3_JET_JER_down.root",
    "mc16d_kinematics_3_JET_JER_up.root",
]

syst_files_mc16e = [
    "mc16e_kinematics_1_EG.root",
    "mc16e_kinematics_1_MUON_up.root",
    "mc16e_kinematics_1_MUON_down.root",
    "mc16e_kinematics_2_JET.root",
    "mc16e_kinematics_2_MET.root",
    "mc16e_kinematics_3_JET_GroupedNP.root",
    "mc16e_kinematics_3_JET_JER_down.root",
    "mc16e_kinematics_3_JET_JER_up.root",
]

def booking_histograms(setting, add_TF=False):
    #setting.add_observable("lep1Phi", 15, -3, 3, "lepton #phi", type="reco")
    unfolding.add_observables(setting, "mt", "mtTruth", 10, 0, 100, "mt [GeV]")
    unfolding.add_observables(setting, "jet2Pt", "jet2TruthPt", 10, 0, 500, "Sub-leading jet p_{T} [GeV]")
    unfolding.add_observables(setting, "lep1Pt", "lep1TruthPt", 10, 30, 530, "Leading lepton p_{T} [GeV]")
    # unfolding.add_observables(setting, "TMath::Abs(lep1Eta)", "TMath::Abs(lep1TruthEta)", 25, 0, 2.5, "Abs(lepton p_{eta})")
    unfolding.add_observables(setting, "lep1Eta", "lep1TruthEta", 50, -2.5, 2.5, "lepton #eta")
    #.add_observables(setting, "lep1Phi", "lep1TruthPhi", 15, -3, 3, "lepton phi")
    unfolding.add_observables(setting, "nJet30", "nTruthJet30", 9, 1, 10, "Number of jets (p_{T} > 30 GeV)")

    wpt_bins = list(np.arange(0, 800, 100)) + [800, 1000, 1500]
    unfolding.add_observables(setting, "wPt", "wTruthPt", wpt_bins, 1, 1, "Leading W p_{T} [GeV]")

    ht_bins = list(np.arange(500, 1200, 100)) + list(np.arange(1200, 3000, 200))
    unfolding.add_observables(setting, "Ht30", "HtTruth30", ht_bins, 1, 1, "H_{T} [GeV]")

    mjj_bins = list(np.arange(0, 2000, 200)) + [2000, 2250, 2500,3000, 3500, 4000]
    unfolding.add_observables(setting, "mjj", "mjjTruth", mjj_bins, 1, 1, "Leading and sub-leading jet mass [GeV]")

    jetPt_var_bin = [500, 550, 600, 700, 800, 900, 1000, 1250, 1500]
    unfolding.add_observables(setting, "jet1Pt", "jet1TruthPt", jetPt_var_bin, 1, 1, "Leading jet p_{T} [GeV]")

    unfolding.add_observables(setting, "met", "metTruth", 20, 0, 500, "E_{T}^{miss} [GeV]")
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
        "DeltaPhiWJetClosest100",
        "DeltaPhiTruthWJetClosest100",
        16,
        0,
        3.2,
        "min(#Delta#phi(W,jet_{i}^{100}))",
    )
    unfolding.add_observables(
        setting,
        "DeltaRLepJetClosest100",
        "DeltaRTruthLepJetClosest100",
        20,
        0,
        4,
        "min(#DeltaR(lepton,jet_{i}^{100}))",
    )
    unfolding.add_observables(
        setting,
        "DeltaRLepJetClosest30",
        "DeltaRTruthLepJetClosest30",
        20,
        0,
        4,
        "min(#DeltaR(lepton,jet_{i}^{30}))",
    )

    if add_TF:
        setting.add_histogram2D(
            "eta_vs_lepPt_electron",
            "TMath::Abs(lep1Eta)",
            "lep1Pt",
            xbin=[0.0, 1.05, 2.5],
            ybin=[30, 100, 150, 200, 300, 400, 600, 800, 1000, 1500, 2000, 2500],
            xtitle="eta (2d)",
            ytitle="Lepton pT (2d)",
            type="tf",
        )

        setting.add_histogram2D(
            "eta_vs_lepPt_muon",
            "TMath::Abs(lep1Eta)",
            "lep1Pt",
            xbin=[0.0, 1.05, 2.5],
            ybin=[30, 100, 200, 300, 400, 500, 800, 2500],
            xtitle="eta (2d)",
            ytitle="Lepton pT (2d)",
            type="tf",
        )

        setting.add_observable(
            "Ht30-jet1Pt",
            bins=[0,200,400,600,800,1000,1200,1400,2500],
            xtitle="Ht30-jet1Pt",
            observable="Ht30-jet1Pt",
            type="tf",
        )

        setting.add_histogram2D(
            "jet1Pt_Ht30",
            "jet1Pt",
            "Ht30-jet1Pt+lep1Pt",
            xbin=[500, 550, 600, 700, 800, 900, 1000, 1250, 1500],
            ybin=[0,600,1200],
            #ybin=[0,200,400,600,800,1000,1200,1400,2500],
            xtitle="jet1Pt (2d)",
            ytitle="Ht30-jet1Pt+lep1Pt (2d)",
            type="tf",
        )

        '''
        setting.add_histogram2D(
            "dR_dPhi_reco",
            "DeltaRLepJetClosest100",
            "DeltaPhiWJetClosest100",
            xbin=list(np.arange(0, 5, 0.2)),
            ybin=list(np.arange(0, 3.2, 0.2)),
            xtitle="min(#DeltaR(lepton,jet_{i}^{100}))",
            ytitle="min(#Delta#phi(W,jet_{i}^{100}))",
        )

        setting.add_histogram2D(
            "dR_dPhi_truth",
            "DeltaRTruthLepJetClosest100",
            "DeltaPhiTruthWJetClosest100",
            xbin=list(np.arange(0, 5, 0.2)),
            ybin=list(np.arange(0, 3.2, 0.2)),
            xtitle="truth min(#DeltaR(lepton,jet_{i}^{100}))",
            ytitle="truth min(#Delta#phi(W,jet_{i}^{100}))",
        )
        '''


def main():
    setting = ConfigMgr(
        src_path="/gpfs/slac/atlas/fs1/d/yuzhan/collinearw_files/Wj_AB212164_v5/merged_files/",
        #out_path="./unfoldTest_v3_fullrun2/",  # output
        description="unfolding",
        histogram_backend="numpy",  # histograms backend
    )

    output_path = pathlib.Path(os.path.realpath(__file__)).parent
    setting.set_output_location(output_path)
    #
    setting.set_singlefile(
        [
            "mc16a_weights.root",
            "mc16d_weights.root",
            "mc16e_weights.root",
        ]
        + syst_files_mc16a
        + syst_files_mc16d
        + syst_files_mc16e
    )

    booking_processes(setting)

    # selections
    # NB: ABCD plane for muons needs to be in the met > 25 GeV phase-space
    lepton_selections = {
        "electron": ["AnalysisType==1"],
        "muon": ["AnalysisType==2 && lep1Signal == 1"],
        #"muon_LowMet": ["AnalysisType==2 && met <= 25"],
        #"muon_FailPID" : ["AnalysisType==2 && lep1Signal == 0"],
    }

    phasespace_selections = {
        "inclusive": [],
        "collinear": ["DeltaRLepJetClosest100<=2.6"],
        "backtoback": ["DeltaRLepJetClosest100>2.6"],
        "inclusive_2j" : ["nJet30 >= 2"],
        "backtoback_lowMET" : ["DeltaRLepJetClosest100>2.6 && met<100"],
        # "AbsEtaLess2p47": ["TMath::Abs(lep1Eta) < 2.47"],
    }

    truth_phasespace_selections = {
        "inclusive": [],
        "collinear": ["DeltaRTruthLepJetClosest100<=2.6"],
        "backtoback": ["DeltaRTruthLepJetClosest100>2.6"],
        "inclusive_2j" : ["nTruthJet30 >= 2"],
        "backtoback_lowMET" : ["DeltaRTruthLepJetClosest100>2.6 && metTruth<100"]
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
        "inclusive": ["TM_lepDR<0.4", "TM_leadJetDR<0.4", "TM_closestJet100DR<0.4"],
        "collinear": ["TM_lepDR<0.4", "TM_leadJetDR<0.4", "TM_closestJet100DR<0.4"],
        "backtoback": ["TM_lepDR<0.4", "TM_leadJetDR<0.4", "TM_closestJet100DR<0.4"],
        "inclusive_2j": ["TM_lepDR<0.4", "TM_leadJetDR<0.4", "TM_subleadJetDR<0.4"],
        "backtoback_lowMET": ["TM_lepDR<0.4", "TM_leadJetDR<0.4", "TM_closestJet100DR<0.4"],
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
            tSelection += truth_phasespace_selections[phasespace]
            rSelection = copy.copy(reco_selection)
            rSelection += lepton_selection
            rSelection += phasespace_selection

            # We should improve this, maybe by having the truth selection
            # bundled with the lepton selection? Works for now...
            if lepton_flavor == 'electron' or 'electron' in lepton_flavor:
                tSelection.append('TMath::Abs(lep1TruthPdgId)==11')
                corr_type = "electron"
            if lepton_flavor == "muon" or "muon" in lepton_flavor:
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
                notmatching_selection=truth_notmatching[phasespace] + isolation_selection,
                weight_truth=weight_truth,
                weight_reco=weight_reco,
                corr_type=corr_type,
                filter_region_hist_types=filter_region_hist_types,
            )

    abcd.reserve_abcd_regions(setting, "met-pid", ("met", 25), ("lep1Signal", 1))
    #abcd.reserve_abcd_regions(setting, "fake-muon", ("IsoLoose_FixedRad", 1), ("lep1Signal", 1))
    abcd.reserve_abcd_regions(setting, "muon-met-iso", ("met", 25), ("IsoLoose_FixedRad", 1))
    abcd.reserve_abcd_regions(
        setting,
        "met-ELTrack",
        ("met", 25),
        #("IsoTight_VarRad", 1),
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
        ["muon-met-iso"],
        tag_name="fake-MU",
        base_region_names=[
            region.name for region in setting.regions if 'muon' in region.name and region.type not in ["truth", "reco_match", "reco_match_not"]
        ],
    )
    abcd.create_abcd_regions(
        setting,
        ["muon-met-iso"],
        tag_name="fake-MU-FailPID",
        base_region_names=[
            region.name for region in setting.regions if 'muon_FailPID' in region.name and region.type not in ["truth", "reco_match", "reco_match_not"]
        ],
    )
    abcd.create_abcd_regions(
        setting,
        ["met-pid", "met-ELTrack", "et-cone"],
        tag_name="fake-EL",
        base_region_names=[
            region.name for region in setting.regions if 'electron' in region.name and region.type not in ["truth", "reco_match", "reco_match_not"]
        ],
    )

    # histogram booking
    booking_histograms(setting, add_TF=True)

    # Control regions booking
    booking_control_regions(setting, filter_region_hist_types.get("reco"))

    # Phase-space corrections, but only load the non signal processes (e.g zjets, ttbar)
    # the correction format is a python dict with key as the form (correction type, process name, observable)
    # unfortunatly the correction are derived separetely for each generator, if you want to have single
    # correction file for different wjets generator, you might need to merge and rename the key yourself.
    bkgd_correction = "./iterative_sf/run2_wjets_2211_bkgd_correction.shelf"
    setting.corrections.add_correction_file(bkgd_correction)

    #setting.corrections.add_correction_file("./iterative_sf/dijets_correlation.shelf")
    setting.phasespace_corr_obs = "nJet30"

    # Note: no need to create dummy and fill if nominal phase-space correction is applied for all
    # dummy can be created later after everything is filled. see create_dummy_systematics()
    '''
    setting.load_systematics(
        "minimum_sys.jsonnet",
        # create_dummy=["wjets_2211", "zjets_2211", "ttbar", "diboson", "singletop"],
    )
    '''

    setting.process_region_type_filter = {
        "data" : {"reco_match", "reco_match_not"},
        "zjets_2211" : {"truth", "reco_match", "reco_match_not"},
        "singletop" : {"truth", "reco_match", "reco_match_not"},
        "ttbar" : {"truth", "reco_match", "reco_match_not"},
        "diboson" : {"truth", "reco_match", "reco_match_not"},
        "diboson_powheg" : {"truth", "reco_match", "reco_match_not"},
        "wjets_2211_tau" : {"truth", "reco_match", "reco_match_not"},
        "dijets" : {"truth", "reco_match", "reco_match_not"},
        "wjets" : {"reco"},
        "wjets_FxFx" : {"reco"},
    }
    #setting.prepare(True)

    setting.RAISE_TREENAME_ERROR = False

    workers = setting.count_object("process")
    workers = 8 if workers < 8 else workers
    workers = min(200, workers)
    print(f"using {workers=}")
    cores = 12
    account = "shared"
    cluster = SLURMCluster(
        queue=account,
        walltime='02:00:00',
        project="collinear_Wjets",
        cores=1,
        processes=1,
        memory="64GB",
        job_extra=[f'--account={account}', f'--partition={account}', f'--cpus-per-task={cores}'],
        local_directory="dask_output",
        log_directory="dask_logs",
        #death_timeout=36000,
        n_workers=workers,
        nanny=False,
    )
    print(cluster.job_script())
    client = Client(cluster)
    client.get_versions(check=True)
    # client.wait_for_workers(workers)
    # print(f"{workers=}")

    mc16a_setting = run_HistMaker.run_HistMaker(
        setting,
        split_type="region",
        merge_buffer_size=50,
        #submission_buffer=50,
        executor=client,
        as_completed=dask_as_completed,
        #r_copy=False,
        copy=False,
    )
    mc16a_setting.corrections.clear_buffer()

    # create dummy systematics
    # mc16a_setting.create_dummy_systematics(["wjets_2211", "zjets_2211", "ttbar", "singletop", "diboson_powheg", "vgamma", "dijets"])

    # if input file splitting is use, needs to update file from record
    # mc16a_setting.update_input_file_from_record()

    mc16a_setting.save("run2_2211.pkl")

    # mc16a_setting.save("run2_2211", "shelve")

if __name__ == "__main__":
    main()
