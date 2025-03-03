from collinearw import ConfigMgr
from collinearw import run_HistMaker
from collinearw import run_HistManipulate
from collinearw import run_PlotMaker
from collinearw.strategies import abcd

import logging

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
log = logging.getLogger(__name__)

# preparation

setting = ConfigMgr(
    "/nfs/slac/atlas/fs1/d/yuzhan/collinearw_files/Sep2020_Production/merged_files/"
)

setting.set_output_location(
    "/nfs/slac/atlas/fs1/d/yuzhan/collinearw_ana_2020/Sep_Production_Wj_AB212108_v2/ABCD_study/dijets_closure_standalone_v1/"
)

# overwirte the file name if you only have one file with everything.
# etting.set_singlefile("allTrees_mc16a_Wj_AB212108_v1_jet1Pt500.root")

setting.set_singlefile(
    [
        "Wj_AB212108_v2_mc16a_dijets.root",
        "Wj_AB212108_v2_mc16d_dijets.root",
        "Wj_AB212108_v2_mc16e_dijets.root",
    ]
)

# reserving abcd tags:
abcd.reserve_abcd_regions(
    setting,
    "met-lep1Signal",
    ("met", 25),
    ("lep1Signal", 1),
)
abcd.reserve_abcd_regions(
    setting,
    "met-IsoFCLoose_FixedRad",
    ("met", 25),
    ("IsoFCLoose_FixedRad", 1),
)
abcd.reserve_abcd_regions(
    setting,
    "met-ptcone20_TightTTVALooseCone_pt500",
    ("met", 25),
    ("ptcone20_TightTTVALooseCone_pt500/lep1Pt", 0.06),
    reverse_y=True,
)
abcd.reserve_abcd_regions(
    setting,
    "met-ptvarcone20_TightTTVA_pt1000",
    ("met", 25),
    ("ptvarcone20_TightTTVA_pt1000/lep1Pt", 0.06),
    reverse_y=True,
)
abcd.reserve_abcd_regions(
    setting,
    "muon_met-ptvarcone30_TightTTVA_pt1000",
    ("met", 25),
    ("ptvarcone30_TightTTVA_pt1000/lep1Pt", 0.15),
    reverse_y=True,
)
abcd.reserve_abcd_regions(
    setting,
    "muon_met-ptcone20_TightTTVA_pt1000",
    ("met", 25),
    ("ptcone20_TightTTVA_pt1000/lep1Pt", 0.15),
    reverse_y=True,
)

# name, treename, selection, process_type
setting.add_process("dijets", "dijets_NoSys")

setting.add_process(
    "dijets-origin-NonDefined", "dijets_NoSys", selection="lep1Origin==0"
)
setting.add_process(
    "dijets-origin-PhotonConv", "dijets_NoSys", selection="lep1Origin==5"
)
setting.add_process(
    "dijets-origin-BottomMeson", "dijets_NoSys", selection="lep1Origin==26"
)
setting.add_process(
    "dijets-origin-CharmedMeson", "dijets_NoSys", selection="lep1Origin==25"
)
setting.add_process(
    "dijets-origin-LeftOver",
    "dijets_NoSys",
    selection="lep1Origin!=0 && lep1Origin!=5 && lep1Origin!=25 && lep1Origin!=26",
)
'''
setting.add_process(
    "dijets-origin-Hadronized_Others",
    "dijets_NoSys",
    selection="lep1Origin!=0 && lep1Origin!=5",
)
'''

setting.add_process("dijets-type-Hadron", "dijets_NoSys", selection="lep1Type==17")
setting.add_process("dijets-type-Unk", "dijets_NoSys", selection="lep1Type==0")
setting.add_process("dijets-type-NonIsoE", "dijets_NoSys", selection="lep1Type==3")
setting.add_process("dijets-type-BkgE", "dijets_NoSys", selection="lep1Type==4")
setting.add_process("dijets-type-NonIsoMu", "dijets_NoSys", selection="lep1Type==7")
setting.add_process("dijets-type-BkgMu", "dijets_NoSys", selection="lep1Type==8")
setting.add_process(
    "dijets-type-LeftOver",
    "dijets_NoSys",
    selection="lep1Type!=0 && lep1Type!=3 && lep1Type!=4 && lep1Type!=7 && lep1Type!=8 && lep1Type!=17",
)

# setting.add_process("dijets-2jets","dijets_NoSys", "nJet25>=2","mc", data_file_name)

common_cut = (
    "isReco && nBJet25==0 && lep1Pt>30 && trigMatch_singleLepTrig && jet1Pt>=500"
)

fail_e_iso_pid = "isReco && nBJet25==0 && lep1Pt>30 && trigMatch_singleLepTrig && jet1Pt>=500 && lep1Signal==0 && ptcone20_TightTTVA_pt1000/lep1Pt > 0.06"

fail_mu_iso_pid = "isReco && nBJet25==0 && lep1Pt>30 && trigMatch_singleLepTrig && jet1Pt>=500 && lep1Signal==0 && IsoFCLoose_FixedRad==0"


lepton_type = [
    ("electron", "&& AnalysisType==1"),
    ("muon", "&& AnalysisType==2"),
]

lepton_pT = [
    # ("_L200", "&& lep1Pt<=200"),
    # ("_H200", "&& lep1Pt>200"),
    ("", ""),
]

lepton_pid = [
    # ("_lep1Signal", "&& lep1Signal==1"),
    ("", ""),
]

collinear = [
    ("_collinear", "&& DeltaPhiWJetClosest25 < 1.5"),
    ("_backToBack", "&& DeltaPhiWJetClosest25 >= 1.5"),
    ("", ""),
]

met_selection = [
    ("METL100", "&& met < 100"),
    # ("METH200", "&& met > 200"),
    ("", ""),
]

jet_pt_selection = [
    # ("JETL1000", "&& jet1Pt < 1000"),
    # ("JETH1000", "&& jet1Pt >= 1000"),
    ("", ""),
]

abcd_electron_list = []
abcd_muon_list = []

for lepton, type in lepton_type:
    for pt, pt_cut in lepton_pT:
        for pid, pid_cut in lepton_pid:
            for col, col_cut in collinear:
                for met_n, met_cut in met_selection:
                    for jPt, jPt_cut in jet_pt_selection:
                        setting.add_region(
                            f"{lepton}{pt}{pid}{col}{met_n}",
                            f"{common_cut}{type}{pt_cut}{pid_cut}{col_cut}{met_cut}",
                        )
                        if "electron" == lepton:
                            abcd_electron_list.append(f"{lepton}{pt}{pid}{col}{met_n}")
                        else:
                            abcd_muon_list.append(f"{lepton}{pt}{pid}{col}{met_n}")

abcd.create_abcd_regions(
    setting,
    ["met-ptcone20_TightTTVALooseCone_pt500", "met-lep1Signal"],
    base_region_names=abcd_electron_list,
)
abcd.create_abcd_regions(
    setting,
    ["met-ptvarcone20_TightTTVA_pt1000", "met-lep1Signal"],
    base_region_names=abcd_electron_list,
)
abcd.create_abcd_regions(
    setting,
    ["met-IsoFCLoose_FixedRad", "met-lep1Signal"],
    base_region_names=abcd_muon_list,
)
abcd.create_abcd_regions(
    setting,
    ["muon_met-ptvarcone30_TightTTVA_pt1000", "met-lep1Signal"],
    base_region_names=abcd_muon_list,
)
abcd.create_abcd_regions(
    setting,
    ["muon_met-ptcone20_TightTTVA_pt1000", "met-lep1Signal"],
    base_region_names=abcd_muon_list,
)


setting.add_observable("met", 40, 0, 1000, "met [GeV]")
# setting.add_observable("binW50_met", 20, 0, 1000, "met [GeV]", observable="met")
# setting.add_observable("binW100_met", 10, 0, 1000, "met [GeV]", observable="met")
setting.add_observable("jet1Pt", 21, 400, 2500, "leading jet Pt [GeV]")
setting.add_observable("jet2Pt", 21, 0, 1000, "sub-leading jet Pt [GeV]")
setting.add_observable("lep1Pt", 20, 30, 1500, "leading lepton Pt [GeV]")
setting.add_observable("TMath::Abs(lep1Eta)", 2, -5, 5, "leading lep #eta ")
setting.add_observable("lep1Eta", 10, -5, 5, "leading lep #eta ")
setting.add_observable("mt", 40, 0, 1000, "mt [GeV]")
setting.add_observable("wPt", 20, 0, 1500, "W Pt [GeV]")
setting.add_observable("nJet25", 10, 0, 10, "number of jet25")
# setting.add_observable("DeltaPhiWJetClosest25", 20, 0, 5, "#Delta#Phi(W, closest jet25)")
# setting.add_observable("DeltaRLepJetClosest25", 25, 0, 5, "#Delta R(l, closest jet25)")
# setting.add_observable("DeltaPhiMetJetClosest25", 20, 0, 5, "#Delta#Phi(met, closest jet25)")
# setting.add_observable("DeltaPhiLepMet", 20, -3.5, 3.5, "#Delta#Phi(lep, met)")
# setting.add_observable("ptvarcone20_TightTTVA_pt1000", 100, 0, 5, "ptVarCone20_TightTTVA")
# setting.add_observable("ptvarcone20_TightTTVA_pt1000/lep1Pt", 100, 0, 5, "ptVarCone20_TightTTVA/lep1Pt")


# setting.add_histogram2D("W_vs_jetPt", "wPt", "jet1Pt", 20, 0, 1500, 21, 400, 2500, "W pT", "j pT")
# setting.add_histogram2D("mt_vs_jetPt", "mt", "jet1Pt", 40, 0, 1000, 21, 400, 2500, "mt", "j pT")
# setting.add_histogram2D("lepPt_vs_jetPt", "lep1Pt", "jet1Pt", 20, 30, 1500, 21, 400, 2500, "l Pt", "j pT")
# setting.add_histogram2D("jetPt_vs_jetPt", "jet1Pt", "jet1Pt", 21, 400, 2500, 21, 400, 2500, "j Pt", "j pT")
# setting.add_histogram2D("nJet_vs_jetPt", "nJet25", "jet1Pt", 10, 0, 10, 21, 400, 2500, "n j25", "j pT")

# setting.add_histogram2D("W_vs_lepPt", "wPt", "lep1Pt", 20, 0, 1500, 20, 30, 1500, "W pT", "l pT")
# setting.add_histogram2D("mt_vs_lepPt", "mt", "lep1Pt", 40, 0, 1000, 20, 30, 1500, "mt", "l pT")
# setting.add_histogram2D("lepPt_vs_lepPt", "lep1Pt", "lep1Pt", 20, 30, 1500, 20, 30, 1500, "l Pt", "l pT")
# setting.add_histogram2D("jetPt_vs_lepPt", "jet1Pt", "lep1Pt", 21, 400, 2500, 20, 30, 1500, "j Pt", "l pT")
# setting.add_histogram2D("nJet_vs_lepPt", "nJet25", "lep1Pt", 10, 0, 10, 20, 30, 1500, "n j25", "l pT")
setting.add_histogram2D(
    "eta_vs_lepPt",
    "lep1Eta",
    "lep1Pt",
    10,
    -5,
    5,
    20,
    30,
    1500,
    "eta (2d)",
    "l pT (2d)",
)
'''

setting.add_observable("met", 40, 0, 1000, "met [GeV]")
setting.add_observable("jet1Pt", 21, 400, 2500, "leading jet Pt [GeV]")
setting.add_observable("lep1Pt", 20, 30, 1500, "leading lepton Pt [GeV]")
setting.add_observable("lep1Pt_lessBin", 10, 30, 1500, "leading lepton Pt [GeV]", observable="lep1Pt")
setting.add_observable("lep1Eta", 10, -5, 5, "leading lep #eta ")
setting.add_observable("TMath::Abs(lep1Eta)", 4, -5, 5, "leading lep abs(#eta) ")
setting.add_observable("mt", 40, 0, 1000, "mt [GeV]")
setting.add_observable("wPt", 20, 0, 1500, "W Pt [GeV]")
setting.add_observable("nJet25", 10, 0, 10, "number of jet25")
setting.add_observable("lep1Pt_varbin", [0,100,200,300,400,450,550,600,700,800,1000,1500,2000], xtitle="leading lepton Pt [GeV]", observable="lep1Pt")

#setting.add_observable("DeltaPhiWJetClosest25", 20, 0, 5, "#Delta#Phi(W, closest jet25)")
#setting.add_observable("DeltaRLepJetClosest25", 25, 0, 5, "#Delta R(l, closest jet25)")
#setting.add_observable("DeltaPhiMetJetClosest25", 20, 0, 5, "#Delta#Phi(met, closest jet25)")
#setting.add_observable("nJet50", 10, 0, 10, "number of jet50")
#setting.add_observable("DeltaPhiWJetClosest50", 20, 0, 5, "#Delta#Phi(W, closest jet50)")
#setting.add_observable("DeltaRLepJetClosest50", 25, 0, 5, "#Delta R(l, closest jet50)")
#setting.add_observable("DeltaPhiMetJetClosest50", 20, 0, 5, "#Delta#Phi(met, closest jet50)")
#setting.add_observable("nJet100", 10, 0, 10, "number of jet100")
#setting.add_observable("DeltaPhiWJetClosest100", 20, 0, 5, "#Delta#Phi(W, closest jet100)")
#setting.add_observable("DeltaRLepJetClosest100", 25, 0, 5, "#Delta R(l, closest jet100)")
#setting.add_observable("DeltaPhiMetJetClosest100", 20, 0, 5, "#Delta#Phi(met, closest jet100)")
#setting.add_observable("nJet200", 10, 0, 10, "number of jet200")
#setting.add_observable("DeltaPhiWJetClosest200", 20, 0, 5, "#Delta#Phi(W, closest jet200)")
#setting.add_observable("DeltaRLepJetClosest200", 25, 0, 5, "#Delta R(l, closest jet200)")
#setting.add_observable("DeltaPhiMetJetClosest200", 20, 0, 5, "#Delta#Phi(met, closest jet200)")
#setting.add_histogram2D("W_vs_jetPt", "wPt", "jet1Pt", 20, 0, 1500, 21, 400, 2500, "W pT (2d)", "j pT")
#setting.add_histogram2D("mt_vs_jetPt", "mt", "jet1Pt", 40, 0, 1000, 21, 400, 2500, "mt (2d)", "j pT")
#setting.add_histogram2D("lepPt_vs_jetPt", "lep1Pt", "jet1Pt", 20, 30, 1500, 21, 400, 2500, "l Pt (2d)", "j pT")
#setting.add_histogram2D("jetPt_vs_jetPt", "jet1Pt", "jet1Pt", 21, 400, 2500, 21, 400, 2500, "j Pt (2d)", "j pT")
#setting.add_histogram2D("nJet_vs_jetPt", "nJet25", "jet1Pt", 10, 0, 10, 21, 400, 2500, "n j25 (2d)", "j pT")
setting.add_histogram2D("eta_vs_lepPt", "lep1Eta", "lep1Pt", 10, -5, 5, 20, 30, 1500, "eta (2d)", "l pT (2d)")
setting.add_histogram2D("abs(eta)_vs_lepPt", "TMath::Abs(lep1Eta)", "lep1Pt", 4, -5, 5, 10, 30, 1500, "eta (2d)", "l pT (2d)")

setting.add_histogram2D("eta_vs_lepPt_varbin", "lep1Eta", "lep1Pt", xbin=[-5,-4,-3,-2,-1,0,1,2,3,4,5], ybin=[0,100,200,300,400,450,550,600,700,800,1000,1500,2000], xtitle="eta (2d)", ytitle="l pT (2d)")
setting.add_histogram2D("abs(eta)_vs_lepPt_varbin", "TMath::Abs(lep1Eta)", "lep1Pt", xbin=[0,2.5,5], ybin=[0,100,200,300,400,450,550,600,700,800,1000,1500,2000], xtitle="eta (2d)", ytitle="l pT (2d)")

setting.add_histogram2D("abs(eta)_vs_lepPt_varbin2", "TMath::Abs(lep1Eta)", "lep1Pt", xbin=[0,1,2,3], ybin=[0,30,100,150,200,300,400,450,550,600,700,2000], xtitle="eta (2d)", ytitle="l pT (2d)")
setting.add_histogram2D("abs(eta)_vs_lepPt_varbin3", "TMath::Abs(lep1Eta)", "lep1Pt", xbin=[0,0.5,3], ybin=[0,100,120,150,200,300,400,450,550,600,700,2000], xtitle="eta (2d)", ytitle="l pT (2d)")
setting.add_histogram2D("abs(eta)_vs_lepPt_varbin4", "TMath::Abs(lep1Eta)", "lep1Pt", xbin=[0,2,3], ybin=[0,100,120,150,200,300,400,450,550,600,700,2000], xtitle="eta (2d)", ytitle="l pT (2d)")

setting.add_histogram2D("eta_vs_lepPt_varbin2", "lep1Eta", "lep1Pt", xbin=[-5,-0.5,0,0.5,5], ybin=[0,100,120,150,200,300,600,1000,2000], xtitle="eta (2d)", ytitle="l pT (2d)")
'''
'''
setting.add_histogram2D("eta_vs_lepPt_varbin3", "lep1Eta", "lep1Pt", xbin=[-5,-1.5,0,1.5,5], ybin=[0,100,120,150,200,300,600,1000,2000], xtitle="eta (2d)", ytitle="l pT (2d)")
setting.add_histogram2D("eta_vs_lepPt_varbin4", "lep1Eta", "lep1Pt", xbin=[-5,-0.5,0,0.5,5], ybin=[0,100,120,150,200,450,850,2000], xtitle="eta (2d)", ytitle="l pT (2d)")
setting.add_histogram2D("eta_vs_lepPt_varbin5", "lep1Eta", "lep1Pt", xbin=[-5,-0.5,0,0.5,5], ybin=[0,100,150,200,450,700,2000], xtitle="eta (2d)", ytitle="l pT (2d)")
setting.add_histogram2D("eta_vs_lepPt_varbin6", "lep1Eta", "lep1Pt", xbin=[-5,-0.5,0,0.5,5], ybin=[0,100,150,200,450,550,2000], xtitle="eta (2d)", ytitle="l pT (2d)")
setting.add_histogram2D("eta_vs_lepPt_varbin7", "lep1Eta", "lep1Pt", xbin=[-5,-0.5,0,0.5,5], ybin=[0,100,150,200,450,2000], xtitle="eta (2d)", ytitle="l pT (2d)")
setting.add_histogram2D("eta_vs_lepPt_varbin8", "lep1Eta", "lep1Pt", xbin=[-5,-0.5,0,0.5,5], ybin=[0,100,150,200,300,550,2000], xtitle="eta (2d)", ytitle="l pT (2d)")
setting.add_histogram2D("eta_vs_lepPt_varbin9", "lep1Eta", "lep1Pt", xbin=[-5,-0.5,0,0.5,5], ybin=[0,100,150,200,550,2000], xtitle="eta (2d)", ytitle="l pT (2d)")
setting.add_histogram2D("eta_vs_lepPt_varbin10", "lep1Eta", "lep1Pt", xbin=[-5,-0.5,0,0.5,5], ybin=[0,100,300,550,2000], xtitle="eta (2d)", ytitle="l pT (2d)")
setting.add_histogram2D("eta_vs_lepPt_varbin11", "lep1Eta", "lep1Pt", xbin=[-5,-0.5,0,0.5,5], ybin=[0,100,350,2000], xtitle="eta (2d)", ytitle="l pT (2d)")
setting.add_histogram2D("eta_vs_lepPt_varbin12", "lep1Eta", "lep1Pt", xbin=[-5,0,5], ybin=[0,100,350,2000], xtitle="eta (2d)", ytitle="l pT (2d)")
setting.add_histogram2D("eta_vs_lepPt_varbin13", "lep1Eta", "lep1Pt", xbin=[-5,0,5], ybin=[0,50,100,150,200,250,300,350,400,450,500,550,2000], xtitle="eta (2d)", ytitle="l pT (2d)")
setting.add_histogram2D("eta_vs_lepPt_varbin14", "lep1Eta", "lep1Pt", xbin=[-5,0,5], ybin=[0,50,100,150,350,550,2000], xtitle="eta (2d)", ytitle="l pT (2d)")
setting.add_histogram2D("eta_vs_lepPt_varbin15", "lep1Eta", "lep1Pt", xbin=[-5,0,5], ybin=[0,50,100,550,2000], xtitle="eta (2d)", ytitle="l pT (2d)")
'''

setting.add_histogram2D(
    "eta_vs_lepPt_varbin1",
    "lep1Eta",
    "lep1Pt",
    xbin=[-5, -1, -0.5, 0, 0.5, 1, 5],
    ybin=[0, 50, 100, 120, 150, 200, 300, 400, 500, 600, 1000, 2000],
    xtitle="eta (2d)",
    ytitle="l pT (2d)",
)

setting.add_histogram2D(
    "eta_vs_lepPt_varbin16",
    "lep1Eta",
    "lep1Pt",
    xbin=[-5, 0, 5],
    ybin=[0, 30, 50, 100, 200, 400, 600, 2000],
    xtitle="eta (2d)",
    ytitle="l pT (2d)",
)

setting.add_histogram2D(
    "abs(eta)_vs_lepPt_varbin17",
    "TMath::Abs(lep1Eta)",
    "lep1Pt",
    xbin=[0, 5],
    ybin=[0, 30, 50, 100, 200, 400, 600, 2000],
    xtitle="abs(eta) (2d)",
    ytitle="l pT (2d)",
)

setting.add_histogram2D(
    "eta_vs_lepPt_good",
    "lep1Eta",
    "lep1Pt",
    xbin=[-5, -1, 0, 1, 5],
    ybin=[0, 400, 800, 1200, 5000],
    xtitle="eta (2d)",
    ytitle="l pT (2d)",
)
setting.add_histogram2D(
    "abs(eta)_vs_lepPt_good",
    "TMath::Abs(lep1Eta)",
    "lep1Pt",
    xbin=[0, 1, 5],
    ybin=[0, 400, 800, 1200, 5000],
    xtitle="eta (2d)",
    ytitle="l pT (2d)",
)


setting.save("raw_config")
parsed_configMgr = run_HistMaker.run_HistMaker(setting, rsplit=False)
parsed_configMgr.save("abcd_closure.pkl")

# hist manipulation

# tf_configMgr = run_HistManipulate.run_ABCD_TF(parsed_configMgr, "dijets")

# bin_tf_fakes = run_HistManipulate.run_ABCD_Fakes(tf_configMgr, False)
# const_tf_fakes = run_HistManipulate.run_ABCD_Fakes(tf_configMgr, True)

# bin_tf_fakes_ROOT = run_HistManipulate.run_ROOT(bin_tf_fakes)
# const_tf_fakes_ROOT = run_HistManipulate.run_ROOT(const_tf_fakes)

# producing plots

# run_PlotMaker.run_PlotABCD_TF(bin_tf_fakes, "Transfer Factor")

# run_PlotMaker.run_PlotABCD(bin_tf_fakes_ROOT, "Fakes", data_process="dijets", mc_process="", text="Closure Test")
# run_PlotMaker.run_PlotABCD(const_tf_fakes_ROOT, "const_tf_fakes_plots", data_process="dijets", mc_process="", text="Closure Test")

# bin_tf_fakes_ROOT.save("dijets_fake")
