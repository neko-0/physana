from collinearw import ConfigMgr
from collinearw import run_HistMaker
from collinearw.serialization import Serialization
import os
import pathlib
from collinearw.strategies import abcd
import numpy

import logging
import ROOT

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
log = logging.getLogger(__name__)

src_path = "/gpfs/slac/atlas/fs1/d/yuzhan/collinearw_files/Wj_AB212108_v3p3/merged/"
# src_path = "/nfs/slac/atlas/fs1/d/yuzhan/collinearw_files/Wj_AB212108_v3/merged_files/withSelector/"
configMgr = ConfigMgr(src_path)

output_path = pathlib.Path(os.path.realpath(__file__)).parent
configMgr.set_output_location(output_path)

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
'''
configMgr.add_process(
    "data",
    "dijets_NoSys",
    color=ROOT.kBlack,
    selection="eventWeight < 1",
    combine_tree=[
        "wjets_NoSys",
        "zjets_NoSys",
        "ttbar_NoSys",
        "singletop_NoSys",
        "diboson_NoSys",
    ],
)
'''

configMgr.add_process("data", color=ROOT.kBlack)
# configMgr.add_process("wjets", color=ROOT.kAzure + 1, selection="DatasetNumber!=364197 && DatasetNumber!=364196 && DatasetNumber!=364193 && DatasetNumber!=364185 && DatasetNumber!=364186 && DatasetNumber!=364184 && DatasetNumber!=364187 && DatasetNumber!=364189 && DatasetNumber!=364188 && DatasetNumber!=364191 && DatasetNumber!=364190 && DatasetNumber!= 364192 && DatasetNumber!=364195 && DatasetNumber!=364194")
configMgr.add_process("wjets", color=ROOT.kAzure + 1)
configMgr.add_process("wjets_mg", color=ROOT.kAzure + 1)
# configMgr.add_process("wjets_mg", color=ROOT.kAzure + 1, selection="DatasetNumber!=363659 && DatasetNumber!=363658 && DatasetNumber!=363657 && DatasetNumber!=363656 && DatasetNumber!=363655 && DatasetNumber!=363654 && DatasetNumber!=363653 && DatasetNumber!=363652 && DatasetNumber!=363651 && DatasetNumber!=363650 && DatasetNumber!=363649 && DatasetNumber!=363648")
# configMgr.add_process("wjets_FxFx", color=ROOT.kAzure + 1)
configMgr.add_process("ttbar", color=ROOT.kGreen - 9)
configMgr.add_process("zjets", color=ROOT.kAzure - 1)
configMgr.add_process("singletop", color=ROOT.kGreen - 5)
configMgr.add_process("diboson", color=ROOT.kOrange)

configMgr.add_process(
    "dijets", "dijets_NoSys", color=ROOT.kBlack, selection="eventWeight < 1"
)
'''
configMgr.add_process("NonDefined", "dijets_NoSys", selection="lep1Origin==0 && eventWeight<1")
configMgr.add_process("PhotonConv", "dijets_NoSys", selection="lep1Origin==5")
configMgr.add_process("BottomMeson", "dijets_NoSys", selection="lep1Origin==26")
configMgr.add_process("CharmedMeson", "dijets_NoSys", selection="lep1Origin==25")
configMgr.add_process(
    "LeftOver",
    "dijets_NoSys",
    selection="(lep1Origin!=0 && lep1Origin!=5 && lep1Origin!=25 && lep1Origin!=26)",
)

configMgr.add_process(
    "truth-type-NonIsoE",
    "dijets_NoSys",
    selection="(lep1Type==3)",
)
configMgr.add_process(
    "truth-type-NonIsoMu",
    "dijets_NoSys",
    selection="(lep1Type==7)",
)
configMgr.add_process(
    "truth-type-Hadron",
    "dijets_NoSys",
    selection="(lep1Type==17)",
)
configMgr.add_process(
    "truth-type-Unknown",
    "dijets_NoSys",
    selection="(lep1Type==0 && eventWeight<1)",
)
configMgr.add_process(
    "truth-type-BkgdE",
    "dijets_NoSys",
    selection="(lep1Type==4)",
)
configMgr.add_process(
    "truth-type-BkgdMu",
    "dijets_NoSys",
    selection="(lep1Type==8)",
)
configMgr.add_process(
    "truth-type-others",
    "dijets_NoSys",
    selection="(lep1Type!=0 && lep1Type!=3 && lep1Type!=4 && lep1Type!=7 && lep1Type!=8 && lep1Type!=17)",
)
'''

# Electron selections
electron_1lepton = (
    "met>25.0 && (ptcone20_TightTTVALooseCone_pt500/lep1Pt)<0.06 && lep1Signal==1"
)
electron_2lepton = "(ptcone20_TightTTVALooseCone_pt500/lep1Pt)<0.06 && lep1Signal==1"

# Muon selections
muon_1lepton = "met>25.0 && IsoFCLoose_FixedRad==1 && lep1Signal==1"
muon_2lepton = "IsoFCLoose_FixedRad==1 && lep1Signal==1"

# Regions
configMgr.add_region(
    "ttbarCR_Ele",
    f"{common_cuts} && nBJet25>=2 && nLeptons==1 && AnalysisType==1 && {electron_1lepton}",
    corr_type="electron",
)
configMgr.add_region(
    "ZjetsCR_Ele",
    f"{common_cuts} && nBJet25==0 && nLeptons==2 && AnalysisType==1 && lep2Signal==1 && diLeptonMass>60.0 && diLeptonMass<120.0 && {electron_2lepton}",
    corr_type="electron",
)
configMgr.add_region(
    "WjetsCR_Ele",
    f"{common_cuts} && nBJet25==0 && nLeptons==1 && AnalysisType==1  && {electron_1lepton}",
    corr_type="electron",
)

configMgr.add_region(
    "electron",
    f"{common_cuts} && nBJet25==0 && nLeptons==1 && AnalysisType==1 && (TMath::Abs(lep1Eta) < 1.35 || TMath::Abs(lep1Eta)>1.55)",
    corr_type="electron",
)


abcd.reserve_abcd_regions(
    configMgr,
    "pid",
    ("met", 25),
    ("lep1Signal", 1),
)
abcd.reserve_abcd_regions(
    configMgr,
    "muon_official_iso",
    ("met", 25),
    ("IsoFCLoose_FixedRad", 1),
)
abcd.reserve_abcd_regions(
    configMgr,
    "electron_track_iso",
    ("met", 25),
    ("ptcone20_TightTTVALooseCone_pt500/lep1Pt", 0.06),
    reverse_y=True,
)

abcd.reserve_abcd_regions(
    configMgr,
    "IP_Iso",
    ("lep1Signal", 1),
    ("IsoFCLoose_FixedRad", 1),
)

abcd.reserve_abcd_regions(
    configMgr,
    "Iso_IP",
    ("IsoFCLoose_FixedRad", 1),
    ("lep1Signal", 1),
)

# ===============================================================================
configMgr.add_region(
    "ttbarCR_Mu_muon_Official",
    f"{common_cuts} && nBJet25>=2 && nLeptons==1 && AnalysisType==2 && {muon_1lepton}",
    corr_type="muon_Official",
)
configMgr.add_region(
    "ZjetsCR_Mu_muon_Official",
    f"{common_cuts} && nBJet25==0 && nLeptons==2 && AnalysisType==2 && lep2Signal==1 && diLeptonMass>60.0 && diLeptonMass<120.0 && {muon_2lepton}",
    corr_type="muon_Official",
)
configMgr.add_region(
    "muon_Official",
    f"{common_cuts} && nBJet25==0 && nLeptons==1 && AnalysisType==2",
    corr_type="muon_Official",
)

# ===============================================================================
configMgr.add_region(
    "ttbarCR_Mu_muon_IsoIp",
    f"{common_cuts} && nBJet25>=2 && nLeptons==1 && AnalysisType==2 && {muon_1lepton}",
    corr_type="muon_IsoIp",
)
configMgr.add_region(
    "ZjetsCR_Mu_muon_IsoIp",
    f"{common_cuts} && nBJet25==0 && nLeptons==2 && AnalysisType==2 && lep2Signal==1 && diLeptonMass>60.0 && diLeptonMass<120.0 && {muon_2lepton}",
    corr_type="muon_IsoIp",
)
configMgr.add_region(
    "muon_IsoIp",
    f"{common_cuts} && nBJet25==0 && nLeptons==1 && AnalysisType==2 && met>=25",
    corr_type="muon_IsoIp",
)
# ===============================================================================
configMgr.add_region(
    "muon_IpIso",
    f"{common_cuts} && nBJet25==0 && nLeptons==1 && AnalysisType==2 && met>=25",
    corr_type="muon_IpIso",
)
configMgr.add_region(
    "ttbarCR_Mu_muon_IpIso",
    f"{common_cuts} && nBJet25>=2 && nLeptons==1 && AnalysisType==2 && {muon_1lepton}",
    corr_type="muon_IpIso",
)
configMgr.add_region(
    "ZjetsCR_Mu_muon_IpIso",
    f"{common_cuts} && nBJet25==0 && nLeptons==2 && AnalysisType==2 && lep2Signal==1 && diLeptonMass>60.0 && diLeptonMass<120.0 && {muon_2lepton}",
    corr_type="muon_IpIso",
)
# ===============================================================================
configMgr.add_region(
    "muon_IsoIpPid",
    f"{common_cuts} && nBJet25==0 && nLeptons==1 && AnalysisType==2",
    corr_type="muon_IsoIpPid",
)
configMgr.add_region(
    "ttbarCR_Mu_muon_IsoIpPid",
    f"{common_cuts} && nBJet25>=2 && nLeptons==1 && AnalysisType==2 && {muon_1lepton}",
    corr_type="muon_IsoIpPid",
)
configMgr.add_region(
    "ZjetsCR_Mu_muon_IsoIpPid",
    f"{common_cuts} && nBJet25==0 && nLeptons==2 && AnalysisType==2 && lep2Signal==1 && diLeptonMass>60.0 && diLeptonMass<120.0 && {muon_2lepton}",
    corr_type="muon_IsoIpPid",
)

configMgr.add_region(
    "muon_BKB_lowMet_Mu",
    f"{common_cuts} && nBJet25==0 && nLeptons==1 && AnalysisType==2 && DeltaPhiWJetClosest25 > 1.5 && met<100",
    corr_type="muon_IsoIp",
)
configMgr.add_region(
    "muon_BKB_lowMet25_Mu",
    f"{common_cuts} && nBJet25==0 && nLeptons==1 && AnalysisType==2 && DeltaPhiWJetClosest25 > 1.5 && met<25",
    corr_type="muon_IsoIp",
)

configMgr.add_region(
    "muon_Col_lowMet25_Mu",
    f"{common_cuts} && nBJet25==0 && nLeptons==1 && AnalysisType==2 && DeltaPhiWJetClosest25 < 1.5 && met<25",
    corr_type="muon_IsoIp",
)

configMgr.add_region(
    "electron_BKB_lowMet_El",
    f"{common_cuts} && nBJet25==0 && nLeptons==1 && AnalysisType==1 && (TMath::Abs(lep1Eta) < 1.35 || TMath::Abs(lep1Eta)>1.55) && DeltaPhiWJetClosest25 > 1.5 && met<100",
    corr_type="electron",
)


abcd.create_abcd_regions(
    configMgr,
    ["electron_track_iso", "pid"],
    base_region_names=["electron", "electron_BKB_lowMet_El"],
)

abcd.create_abcd_regions(
    configMgr, ["muon_official_iso", "pid"], base_region_names=["muon_Official"]
)

abcd.create_abcd_regions(configMgr, ["IP_Iso"], base_region_names=["muon_IpIso"])

abcd.create_abcd_regions(
    configMgr,
    ["Iso_IP"],
    base_region_names=[
        "muon_IsoIp",
        "muon_BKB_lowMet_Mu",
        "muon_BKB_lowMet25_Mu",
        "muon_Col_lowMet25_Mu",
    ],
)

abcd.create_abcd_regions(
    configMgr, ["Iso_IP", "pid"], base_region_names=["muon_IsoIpPid"], axis=1
)

configMgr.add_observable("nJet25", 9, 1, 10, "Jet multiplicity (p_{T}>25 GeV)")
configMgr.add_observable("met", 50, 0, 500, "E_{T}^{miss} [GeV]")
configMgr.add_observable("jet1Pt", 20, 500, 1500, "Leading jet p_{T} [GeV]")
configMgr.add_observable("lep1Pt", 10, 30, 630, "Leading lepton p_{T} [GeV]")
configMgr.add_observable("lep1Eta", 10, -2.5, 2.5, "Leading lepton #eta ")
configMgr.add_observable("mt", 30, 0, 300, "m_{T} [GeV]")
configMgr.add_observable("mjj", 30, 0, 3000, "m_{jj} [GeV]")
configMgr.add_observable("wPt", 24, 0, 1200, "p_{T}(lepton+MET) [GeV]")
configMgr.add_observable("diLeptonPt", 20, 0, 1000, "Z p_{T} [GeV]")
configMgr.add_observable("diLeptonMass", 20, 60, 120, "m_{ll} [GeV]")
configMgr.add_observable("Ht", 25, 500, 3000, "H_{T} [GeV]")

configMgr.add_observable(
    "DeltaPhiWJetClosest25", 20, 0, 2, "min(#Delta#phi(W,jet_{i}^{25}))"
)
configMgr.add_observable(
    "DeltaRLepJetClosest25", 20, 0, 2, "min(#DeltaR(l,jet_{i}^{25}))"
)
configMgr.add_observable(
    "DeltaPhiWJetClosest30", 20, 0, 2, "min(#Delta#phi(W,jet_{i}^{30}))"
)
"""
configMgr.add_observable(
    "DeltaPhiWJetClosest100", 32, 0, 3.2, "min(#Delta#phi(W,jet_{i}^{100}))"
)
configMgr.add_observable(
    "DeltaRLepJetClosest100", 32, 0, 3.2, "min(#DeltaR(lepton,jet_{i}^{100}))"
)
"""
configMgr.add_observable(
    "ptcone20_TightTTVA_pt1000/lep1Pt", 50, 0, 0.2, "lepton TrackCone/pT"
)
configMgr.add_observable("lep1Topoetcone20/lep1Pt", 50, 0, 0.2, "lepton CaloCone/pT")


configMgr.add_histogram2D(
    "tf_eta_vs_lepPt_el",
    "TMath::Abs(lep1Eta)",
    "lep1Pt",
    xbin=[0.0, 1.05, 2.5],
    ybin=[30, 100, 150, 200, 300, 400, 600, 800, 1000, 1500, 2000, 2500],
    xtitle="eta (2d)",
    ytitle="l pT (2d)",
)

xx = (
    list(numpy.arange(0, 1.5, 0.05))
    + list(numpy.arange(1.5, 3.5, 0.2))[1:]
    + list(numpy.arange(3.5, 5, 0.5))[1:]
)
configMgr.add_histogram2D(
    "tf_dRLJet_Ht",
    "DeltaRLepJetClosest25",
    "Ht",
    xbin=xx,
    # ybin=list(numpy.arange(500, 3000, 50)),
    ybin=[30, 100, 200, 300, 400, 500, 800, 2500],
    xtitle="min(#DeltaR(lepton,jet_{i}^{25})) (2d)",
    ytitle="Ht (2d)",
)

configMgr.add_histogram2D(
    "tf_dRLJet_Pt",
    "DeltaRLepJetClosest25",
    "lep1Pt",
    xbin=list(numpy.arange(0, 2, 0.5)) + list(numpy.arange(2, 6, 0.25)),
    # xbin = [0,0.5,1.5,2,2.5,3,3.5],
    # ybin=list(numpy.arange(500, 2500, 50)),
    ybin=[0, 50, 100, 150, 200, 250, 300, 350, 400, 500, 1000, 1500],
    xtitle="dR l jet (2d)",
    ytitle="l pt (2d)",
)

configMgr.add_histogram2D(
    "tf_eta_vs_lepPt_mu",
    "TMath::Abs(lep1Eta)",
    "lep1Pt",
    xbin=[0.0, 1.05, 2.5],
    ybin=[30, 100, 200, 300, 400, 500, 800, 2500],
    xtitle="eta (2d)",
    ytitle="l pT (2d)",
)

configMgr.add_observable(
    "lep1Pt_varbin-nominal_e",
    [30, 100, 150, 200, 300, 400, 600, 800, 1000, 1500, 2000, 2500],
    xtitle="leading lepton p_{T} [GeV]",
    observable="lep1Pt",
)

configMgr.add_observable(
    "lep1Pt_varbin-nominal_mu",
    [0, 100, 200, 300, 500, 700, 1000, 2000],
    xtitle="leading lepton p_{T} [GeV]",
    observable="lep1Pt",
)

configMgr.add_observable(
    "lep1Eta_varbin-nominal_e",
    [0.0, 1.05, 2.5],
    xtitle="leading lepton #eta ",
    observable="TMath::Abs(lep1Eta)",
)

configMgr.add_observable(
    "lep1Eta_varbin-nominal_mu",
    [0.0, 1.05, 2.5],
    xtitle="leading lepton #eta ",
    observable="TMath::Abs(lep1Eta)",
)

m_serial = Serialization()
corr = m_serial.from_pickle(
    f"{output_path}/../Correction/wjets_inclusive_correction.pkl"
)
# configMgr.corrections = {}
# configMgr.corrections[("electron", "wjets_NoSys", "nJet25")] = corr[("electron", "wjets", "nJet25")]
# configMgr.corrections[("electron", "wjets", "nJet25")] = corr[("electron", "wjets", "nJet25")]
# configMgr.corrections[("muon", "wjets_NoSys", "nJet25")] = corr[("muon", "wjets", "nJet25")]
# configMgr.corrections[("muon", "wjets", "nJet25")] = corr[("muon", "wjets", "nJet25")]

# configMgr.corrections[("muon", "dijets_NoSys", "nJet25")] = 10.0
# configMgr.corrections[("muon", "dijets", "nJet25")] = 10.0

configMgr.save("run2_realData_v2_raw_config")
configMgr = run_HistMaker.run_HistMaker(configMgr, rsplit=False)
configMgr.save("run2_realData_v2.pkl")
