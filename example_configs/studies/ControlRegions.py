from collinearw import ConfigMgr, HistManipulate
from collinearw import run_HistMaker
from collinearw import run_HistManipulate
from collinearw import run_PlotMaker

from collinearw.strategies import abcd

import logging
import os
import ROOT

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
log = logging.getLogger(__name__)

folder_name = "Wj_AB212108_v3/ControlRegions/"


def getConfig(name):

    if os.path.isfile(f"../../output/{folder_name}/{name}.pkl"):
        c = ConfigMgr.open(f"../../output/{folder_name}/{name}.pkl")
        return c
    return None


def run(name, corrections=None):

    configMgr = ConfigMgr(
        "/nfs/slac/atlas/fs1/d/yuzhan/collinearw_files/Wj_AB212108_v3/merged_files/withSelector/",
    )

    # Apply corrections if passed
    if corrections:
        configMgr.corrections = corrections.corrections

    configMgr.set_singlefile(
        [
            "mc16a.root",
            "mc16d.root",
            "mc16e.root",
        ]
    )

    # Common phase-space cuts
    common_cuts = "isReco && lep1Pt>30.0 && trigMatch_singleLepTrig && jet1Pt>=500.0"

    # Process level configuration
    configMgr.add_process("data", color=ROOT.kBlack)
    configMgr.add_process("wjets", color=ROOT.kAzure + 1)
    configMgr.add_process("ttbar", color=ROOT.kGreen - 9)
    configMgr.add_process("zjets", color=ROOT.kAzure - 1)
    configMgr.add_process("singletop", color=ROOT.kGreen - 5)
    configMgr.add_process("diboson", color=ROOT.kOrange)

    # Electron selections
    electron_1lepton = (
        'met>25.0 && (ptcone20_TightTTVALooseCone_pt500/lep1Pt)<0.06 && lep1Signal==1'
    )
    electron_2lepton = (
        '(ptcone20_TightTTVALooseCone_pt500/lep1Pt)<0.06 && lep1Signal==1'
    )

    # Muon selections
    muon_1lepton = 'met>25.0 && IsoFCLoose_FixedRad==1 && lep1Signal==1'
    muon_2lepton = 'IsoFCLoose_FixedRad==1 && lep1Signal==1'

    # Regions
    configMgr.add_region(
        'ttbarCR_Ele',
        f"{common_cuts} && nBJet25>=2 && nLeptons==1 && AnalysisType==1 && {electron_1lepton}",
        corr_type="electron",
    )
    configMgr.add_region(
        'ZjetsCR_Ele',
        f"{common_cuts} && nBJet25==0 && nLeptons==2 && AnalysisType==1 && lep2Signal==1 && diLeptonMass>60.0 && diLeptonMass<120.0 && {electron_2lepton}",
        corr_type="electron",
    )
    configMgr.add_region(
        'WjetsCR_Ele',
        f"{common_cuts} && nBJet25==0 && nLeptons==1 && AnalysisType==1  && {electron_1lepton}",
        corr_type="electron",
    )

    configMgr.add_region(
        'ttbarCR_Mu',
        f"{common_cuts} && nBJet25>=2 && nLeptons==1 && AnalysisType==2 && {muon_1lepton}",
        corr_type="muon",
    )
    configMgr.add_region(
        'ZjetsCR_Mu',
        f"{common_cuts} && nBJet25==0 && nLeptons==2 && AnalysisType==2 && lep2Signal==1 && diLeptonMass>60.0 && diLeptonMass<120.0 && {muon_2lepton}",
        corr_type="muon",
    )
    configMgr.add_region(
        'WjetsCR_Mu',
        f"{common_cuts} && nBJet25==0 && nLeptons==1 && AnalysisType==2 && {muon_1lepton}",
        corr_type="muon",
    )

    configMgr.add_observable("nJet25", 9, 1, 10, "Jet multiplicity (p_{T}>25 GeV)")
    configMgr.add_observable("met", 40, 0, 400, "E_{T}^{miss} [GeV]")
    configMgr.add_observable("jet1Pt", 20, 500, 1500, "Leading jet p_{T} [GeV]")
    configMgr.add_observable("lep1Pt", 10, 30, 630, "Leading lepton p_{T} [GeV]")
    configMgr.add_observable("lep1Eta", 10, -2.5, 2.5, "Leading lepton #eta ")
    configMgr.add_observable("mt", 30, 0, 300, "m_{T} [GeV]")
    configMgr.add_observable("mjj", 30, 0, 3000, "m_{jj} [GeV]")
    configMgr.add_observable("wPt", 24, 0, 1200, "p_{T}(lepton+MET) [GeV]")
    configMgr.add_observable("nJet25", 9, 1, 10, "Jet multiplicity (p_{T}>25 GeV)")
    configMgr.add_observable("diLeptonPt", 20, 0, 1000, "Z p_{T} [GeV]")
    configMgr.add_observable("diLeptonMass", 20, 60, 120, "m_{ll} [GeV]")

    configMgr.add_observable(
        "DeltaPhiWJetClosest25", 20, 0, 2, "min(#Delta#phi(W,jet_{i}^{25}))"
    )
    configMgr.add_observable(
        "DeltaPhiWJetClosest30", 20, 0, 2, "min(#Delta#phi(W,jet_{i}^{30}))"
    )
    configMgr.add_observable(
        "DeltaPhiWJetClosest100", 32, 0, 3.2, "min(#Delta#phi(W,jet_{i}^{100}))"
    )
    configMgr.add_observable(
        "DeltaRLepJetClosest100", 32, 0, 3.2, "min(#DeltaR(lepton,jet_{i}^{100}))"
    )

    configMgr.save(f"{name}_raw_config")
    configMgr = run_HistMaker.run_HistMaker(configMgr)
    configMgr.save()

    if corrections:
        return configMgr
    else:
        histManipulate = HistManipulate()
        histManipulate.addCorrectionRegion("electron", "ttbar", "ttbarCR_Ele")
        histManipulate.addCorrectionRegion("electron", "zjets", "ZjetsCR_Ele")
        histManipulate.addCorrectionRegion("muon", "ttbar", "ttbarCR_Mu")
        histManipulate.addCorrectionRegion("muon", "zjets", "ZjetsCR_Mu")

        # Derive corrections and write the resulting condigMgr
        cConfigMgr = histManipulate.DeriveCorrection(configMgr)
        cConfigMgr.save(name)

        return cConfigMgr


# Grab baseline, if it exists, skip
uncorrected_configMgr = getConfig('corrections')
if not uncorrected_configMgr:
    uncorrected_configMgr = run('corrections')

corrected_configMgr = getConfig('corrected')
if not corrected_configMgr:
    corrected_configMgr = run('corrected', uncorrected_configMgr)


run_PlotMaker.run_stack(uncorrected_configMgr, "CR-prefit", low_yrange=[0.0, 2.0])
run_PlotMaker.run_stack(corrected_configMgr, "CR-postfit", low_yrange=[0.0, 2.0])
