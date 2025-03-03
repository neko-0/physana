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

common_cut = (
    "isReco && nBJet25==0 && lep1Pt>30 && trigMatch_singleLepTrig && jet1Pt>=500"
)

lepton_type = [
    ("electron", "&& AnalysisType==1"),
    ("muon", "&& AnalysisType==2"),
]

lepton_pT = [
    # ("_L200", "&& lep1Pt<=200"),
    # ("_H200", "&& lep1Pt>200"),
    ("", ""),
]

collinear = [
    # ("_collinear", "&& DeltaPhiWJetClosest25 < 1.5"),
    ("_backToBack", "&& DeltaPhiWJetClosest25 >= 1.5"),
    ("", ""),
]

lepton_pid = [
    # ("_lep1Signal", "&& lep1Signal"),
    ("", ""),
]

met_selection = [
    ("METL100", "&& met < 100"),
    # ("METH200", "&& met > 200"),
    ("", ""),
]
# preparation for mc16a

run2_setting = ConfigMgr(
    "/nfs/slac/atlas/fs1/d/yuzhan/collinearw_files/June2020_Production/merged_files/",
)

run2_setting.set_output_location(
    "/nfs/slac/atlas/fs1/d/yuzhan/collinearw_ana_2020/June_Production_Wj_AB212108_v2/ABCD_study/run2_v13/"
)

# overwirte the file name if you only have one file with everything.
run2_setting.set_singlefile(
    [
        "Wj_AB212108_v2_mc16a.root",
        "Wj_AB212108_v2_mc16e.root",
        "Wj_AB212108_v2_mc16d.root",
    ]
)

# reserving abcd tags:
abcd.reserve_abcd_regions(
    run2_setting,
    "met-lep1Signal",
    ("met", 25),
    ("lep1Signal", 1),
)
abcd.reserve_abcd_regions(
    run2_setting,
    "met-IsoFCLoose_FixedRad",
    ("met", 25),
    ("IsoFCLoose_FixedRad", 1),
)
abcd.reserve_abcd_regions(
    run2_setting,
    "met-ptcone20_TightTTVALooseCone_pt500",
    ("met", 25),
    ("ptcone20_TightTTVALooseCone_pt500/lep1Pt", 0.06),
    reverse_y=True,
)
abcd.reserve_abcd_regions(
    run2_setting,
    "met-ptvarcone20_TightTTVA_pt1000",
    ("met", 25),
    ("ptvarcone20_TightTTVA_pt1000/lep1Pt", 0.06),
    reverse_y=True,
)
abcd.reserve_abcd_regions(
    run2_setting,
    "muon_met-ptvarcone30_TightTTVA_pt1000",
    ("met", 25),
    ("ptvarcone30_TightTTVA_pt1000/lep1Pt", 0.15),
    reverse_y=True,
)
abcd.reserve_abcd_regions(
    run2_setting,
    "muon_met-ptcone20_TightTTVA_pt1000",
    ("met", 25),
    ("ptcone20_TightTTVA_pt1000/lep1Pt", 0.15),
    reverse_y=True,
)

# name, treename, selection, process_type
run2_setting.add_process("data", "data", "data")
run2_setting.add_process("wjets", "wjets_NoSys", "mc")
run2_setting.add_process("zjets", "zjets_NoSys", "mc")
run2_setting.add_process("ttbar", "ttbar_NoSys", "mc")
run2_setting.add_process("singletop", "singletop_NoSys", "mc")
run2_setting.add_process("diboson", "diboson_NoSys", "mc")

abcd_electron_list = []
abcd_muon_list = []

for lepton, type in lepton_type:
    for pt, pt_cut in lepton_pT:
        for pid, pid_cut in lepton_pid:
            for col, col_cut in collinear:
                for met_n, met_cut in met_selection:
                    run2_setting.add_region(
                        f"{lepton}{pt}{pid}{col}{met_n}",
                        f"{common_cut}{type}{pt_cut}{pid_cut}{col_cut}{met_cut}",
                    )
                    if "electron" == lepton:
                        abcd_electron_list.append(f"{lepton}{pt}{pid}{col}{met_n}")
                    else:
                        abcd_muon_list.append(f"{lepton}{pt}{pid}{col}{met_n}")

abcd.create_abcd_regions(
    run2_setting,
    ["met-ptcone20_TightTTVALooseCone_pt500", "met-lep1Signal"],
    base_region_names=abcd_electron_list,
)
abcd.create_abcd_regions(
    run2_setting,
    ["met-ptvarcone20_TightTTVA_pt1000", "met-lep1Signal"],
    base_region_names=abcd_electron_list,
)
abcd.create_abcd_regions(
    run2_setting,
    ["met-IsoFCLoose_FixedRad", "met-lep1Signal"],
    base_region_names=abcd_muon_list,
)
abcd.create_abcd_regions(
    run2_setting,
    ["muon_met-ptvarcone30_TightTTVA_pt1000", "met-lep1Signal"],
    base_region_names=abcd_muon_list,
)
abcd.create_abcd_regions(
    run2_setting,
    ["muon_met-ptcone20_TightTTVA_pt1000", "met-lep1Signal"],
    base_region_names=abcd_muon_list,
)

run2_setting.add_observable("met", 40, 0, 1000, "met [GeV]")
run2_setting.add_observable("jet1Pt", 21, 400, 2500, "leading jet Pt [GeV]")
run2_setting.add_observable("lep1Pt", 20, 30, 1500, "leading lepton Pt [GeV]")
run2_setting.add_observable(
    "lep1Pt_lessBin", 10, 30, 1500, "leading lepton Pt [GeV]", observable="lep1Pt"
)
run2_setting.add_observable("lep1Eta", 10, -5, 5, "leading lep #eta ")
run2_setting.add_observable("TMath::Abs(lep1Eta)", 4, -5, 5, "leading lep abs(#eta) ")
run2_setting.add_observable("mt", 40, 0, 1000, "mt [GeV]")
run2_setting.add_observable("wPt", 20, 0, 1500, "W Pt [GeV]")
run2_setting.add_observable("nJet25", 10, 0, 10, "number of jet25")

run2_setting.add_observable(
    "lep1Pt_varbin",
    [0, 100, 200, 300, 400, 450, 550, 600, 700, 800, 1000, 1500, 2000],
    xtitle="leading lepton Pt [GeV]",
    observable="lep1Pt",
)

# run2_setting.add_observable("DeltaPhiWJetClosest25", 20, 0, 5, "#Delta#Phi(W, closest jet25)")
# run2_setting.add_observable("DeltaRLepJetClosest25", 25, 0, 5, "#Delta R(l, closest jet25)")
# run2_setting.add_observable("DeltaPhiMetJetClosest25", 20, 0, 5, "#Delta#Phi(met, closest jet25)")
# run2_setting.add_observable("nJet50", 10, 0, 10, "number of jet50")
# run2_setting.add_observable("DeltaPhiWJetClosest50", 20, 0, 5, "#Delta#Phi(W, closest jet50)")
# run2_setting.add_observable("DeltaRLepJetClosest50", 25, 0, 5, "#Delta R(l, closest jet50)")
# run2_setting.add_observable("DeltaPhiMetJetClosest50", 20, 0, 5, "#Delta#Phi(met, closest jet50)")
# run2_setting.add_observable("nJet100", 10, 0, 10, "number of jet100")
# run2_setting.add_observable("DeltaPhiWJetClosest100", 20, 0, 5, "#Delta#Phi(W, closest jet100)")
# run2_setting.add_observable("DeltaRLepJetClosest100", 25, 0, 5, "#Delta R(l, closest jet100)")
# run2_setting.add_observable("DeltaPhiMetJetClosest100", 20, 0, 5, "#Delta#Phi(met, closest jet100)")
# run2_setting.add_observable("nJet200", 10, 0, 10, "number of jet200")
# run2_setting.add_observable("DeltaPhiWJetClosest200", 20, 0, 5, "#Delta#Phi(W, closest jet200)")
# run2_setting.add_observable("DeltaRLepJetClosest200", 25, 0, 5, "#Delta R(l, closest jet200)")
# run2_setting.add_observable("DeltaPhiMetJetClosest200", 20, 0, 5, "#Delta#Phi(met, closest jet200)")
# run2_setting.add_histogram2D("W_vs_jetPt", "wPt", "jet1Pt", 20, 0, 1500, 21, 400, 2500, "W pT (2d)", "j pT")
# run2_setting.add_histogram2D("mt_vs_jetPt", "mt", "jet1Pt", 40, 0, 1000, 21, 400, 2500, "mt (2d)", "j pT")
# run2_setting.add_histogram2D("lepPt_vs_jetPt", "lep1Pt", "jet1Pt", 20, 30, 1500, 21, 400, 2500, "l Pt (2d)", "j pT")
# run2_setting.add_histogram2D("jetPt_vs_jetPt", "jet1Pt", "jet1Pt", 21, 400, 2500, 21, 400, 2500, "j Pt (2d)", "j pT")
# run2_setting.add_histogram2D("nJet_vs_jetPt", "nJet25", "jet1Pt", 10, 0, 10, 21, 400, 2500, "n j25 (2d)", "j pT")
run2_setting.add_histogram2D(
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
run2_setting.add_histogram2D(
    "abs(eta)_vs_lepPt",
    "TMath::Abs(lep1Eta)",
    "lep1Pt",
    4,
    -5,
    5,
    10,
    30,
    1500,
    "eta (2d)",
    "l pT (2d)",
)

run2_setting.add_histogram2D(
    "eta_vs_lepPt_varbin",
    "lep1Eta",
    "lep1Pt",
    xbin=[-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    ybin=[0, 100, 200, 300, 400, 450, 550, 600, 700, 800, 1000, 1500, 2000],
    xtitle="eta (2d)",
    ytitle="l pT (2d)",
)
run2_setting.add_histogram2D(
    "abs(eta)_vs_lepPt_varbin",
    "TMath::Abs(lep1Eta)",
    "lep1Pt",
    xbin=[0, 2.5, 5],
    ybin=[0, 100, 200, 300, 400, 450, 550, 600, 700, 800, 1000, 1500, 2000],
    xtitle="eta (2d)",
    ytitle="l pT (2d)",
)

run2_setting.add_histogram2D(
    "abs(eta)_vs_lepPt_varbin2",
    "TMath::Abs(lep1Eta)",
    "lep1Pt",
    xbin=[0, 1, 2, 3],
    ybin=[0, 30, 100, 150, 200, 300, 400, 450, 550, 600, 700, 2000],
    xtitle="eta (2d)",
    ytitle="l pT (2d)",
)
run2_setting.add_histogram2D(
    "abs(eta)_vs_lepPt_varbin3",
    "TMath::Abs(lep1Eta)",
    "lep1Pt",
    xbin=[0, 0.5, 3],
    ybin=[0, 100, 120, 150, 200, 300, 400, 450, 550, 600, 700, 2000],
    xtitle="eta (2d)",
    ytitle="l pT (2d)",
)
run2_setting.add_histogram2D(
    "abs(eta)_vs_lepPt_varbin4",
    "TMath::Abs(lep1Eta)",
    "lep1Pt",
    xbin=[0, 2, 3],
    ybin=[0, 100, 120, 150, 200, 300, 400, 450, 550, 600, 700, 2000],
    xtitle="eta (2d)",
    ytitle="l pT (2d)",
)

run2_setting.add_histogram2D(
    "eta_vs_lepPt_varbin10",
    "lep1Eta",
    "lep1Pt",
    xbin=[-5, -0.5, 0, 0.5, 5],
    ybin=[0, 100, 300, 550, 2000],
    xtitle="eta (2d)",
    ytitle="l pT (2d)",
)
run2_setting.add_histogram2D(
    "eta_vs_lepPt_varbin12",
    "lep1Eta",
    "lep1Pt",
    xbin=[-5, 0, 5],
    ybin=[0, 100, 350, 2000],
    xtitle="eta (2d)",
    ytitle="l pT (2d)",
)
run2_setting.add_histogram2D(
    "eta_vs_lepPt_varbin14",
    "lep1Eta",
    "lep1Pt",
    xbin=[-5, 0, 5],
    ybin=[0, 50, 100, 150, 350, 550, 2000],
    xtitle="eta (2d)",
    ytitle="l pT (2d)",
)
run2_setting.add_histogram2D(
    "eta_vs_lepPt_varbin15",
    "lep1Eta",
    "lep1Pt",
    xbin=[-5, 0, 5],
    ybin=[0, 50, 100, 550, 2000],
    xtitle="eta (2d)",
    ytitle="l pT (2d)",
)
run2_setting.add_histogram2D(
    "eta_vs_lepPt_varbin16",
    "lep1Eta",
    "lep1Pt",
    xbin=[-5, 0, 5],
    ybin=[0, 30, 50, 100, 200, 400, 600, 2000],
    xtitle="eta (2d)",
    ytitle="l pT (2d)",
)

run2_setting.add_histogram2D(
    "abs(eta)_vs_lepPt_varbin17",
    "TMath::Abs(lep1Eta)",
    "lep1Pt",
    xbin=[0, 5],
    ybin=[0, 30, 50, 100, 200, 400, 600, 2000],
    xtitle="abs(eta) (2d)",
    ytitle="l pT (2d)",
)

run2_setting.add_histogram2D(
    "eta_vs_lepPt_good",
    "lep1Eta",
    "lep1Pt",
    xbin=[-5, -1, 0, 1, 5],
    ybin=[0, 400, 800, 1200, 5000],
    xtitle="eta (2d)",
    ytitle="l pT (2d)",
)
run2_setting.add_histogram2D(
    "abs(eta)_vs_lepPt_good",
    "TMath::Abs(lep1Eta)",
    "lep1Pt",
    xbin=[0, 1, 5],
    ybin=[0, 400, 800, 1200, 5000],
    xtitle="eta (2d)",
    ytitle="l pT (2d)",
)


# run2_setting.add_histogram2D("W_vs_lepPt", "wPt", "lep1Pt", 20, 0, 1500, 20, 30, 1500, "W p,T", "l pT")
# run2_setting.add_histogram2D("mt_vs_lepPt", "mt", "lep1Pt", 40, 0, 1000, 20, 30, 1500, "mt", "l pT")
# run2_setting.add_histogram2D("lepPt_vs_lepPt", "lep1Pt", "lep1Pt", 20, 30, 1500, 20, 30, 1500, "l Pt", "l pT")
# run2_setting.add_histogram2D("jetPt_vs_lepPt", "jet1Pt", "lep1Pt", 21, 400, 2500, 20, 30, 1500, "j Pt", "l pT")
# run2_setting.add_histogram2D("nJet_vs_lepPt", "nJet25", "lep1Pt", 10, 0, 10, 20, 30, 1500, "n j25", "l pT")

run2_setting.save("run2_raw_config")

# run_HistMaker_split_process run_HistMaker
parsed_run2_setting = run_HistMaker.run_HistMaker(run2_setting, "run2.pkl", rsplit=True)

parsed_run2_setting.save()


'''
# hist manipulation

tf_configMgr = run_HistManipulate.run_ABCD_TF(parsed_configMgr, "dijets")

bin_tf_fakes = run_HistManipulate.run_ABCD_Fakes(tf_configMgr, False)
const_tf_fakes = run_HistManipulate.run_ABCD_Fakes(tf_configMgr, True)

# producing plots

run_PlotMaker.run_PlotABCD_TF(bin_tf_fakes, "Transfer Factor")

run_PlotMaker.run_PlotABCD(bin_tf_fakes, "bin_tf_fakes_plots", data_process="dijets", mc_process="", text="Closure Test")
run_PlotMaker.run_PlotABCD(const_tf_fakes, "const_tf_fakes_plots", data_process="dijets", mc_process="", text="Closure Test")
'''
