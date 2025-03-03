from collinearw import ConfigMgr
from collinearw import run_HistMaker
from collinearw import run_HistManipulate
from collinearw import run_PlotMaker

import logging

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
log = logging.getLogger(__name__)

# preparation

setting = ConfigMgr(
    "/nfs/slac/atlas/fs1/d/yuzhan/collinearw_files/Feb2020_production/merged_files/",
    "./dijet_closure/",
    "ABCD_BG",
    "_Wj_AB21298_v1.root",  # file_suffix
    "uproot",  # io backend
    "numpy",  # histograms backend
)

# overwirte the file name if you only have one file with everything.
# setting.set_singlefile("allTrees_mc16a_Wj_AB212108_v1_jet1Pt500.root")

data_file_name = "merged_mc16a_Wj_AB212108_v1f_dijets.root"

# name, treename, selection, process_type
setting.add_process("dijets", "dijets_NoSys", "", "mc", data_file_name)
# setting.add_process("dijets-2jets","dijets_NoSys", "nJet25>=2","mc", data_file_name)

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

lepton_pid = [
    # ("_lep1Signal", "&& lep1Signal"),
    ("", ""),
]

for lepton, type in lepton_type:
    for pt, pt_cut in lepton_pT:
        for pid, pid_cut in lepton_pid:
            setting.add_region(
                f"{lepton}{pt}{pid}",
                f"{common_cut}{type}{pt_cut}{pid_cut}",
                study_type="abcd",
            )

setting.reserve_abcd_regions("met-lep1Signal", ("met", 25), ("lep1Signal", 1))
setting.reserve_abcd_regions("met-IsoFCTight", ("met", 25), ("IsoFCTight", 1))
setting.reserve_abcd_regions(
    "met-IsoFCLoose_FixedRad", ("met", 25), ("IsoFCLoose_FixedRad", 1)
)
setting.reserve_abcd_regions(
    "met-ptcone20_TightTTVA_pt1000",
    ("met", 25),
    ("ptcone20_TightTTVA_pt1000/lep1Pt", 0.06),
    reverse_y=True,
)
setting.reserve_abcd_regions(
    "muon_met-ptcone20_TightTTVA_pt1000",
    ("met", 25),
    ("ptcone20_TightTTVA_pt1000/lep1Pt", 0.15),
    reverse_y=True,
)
setting.reserve_abcd_regions(
    "met-ptcone20_TightTTVALooseCone_pt1000",
    ("met", 25),
    ("ptcone20_TightTTVALooseCone_pt1000/lep1Pt", 0.06),
    reverse_y=True,
)
setting.reserve_abcd_regions(
    "met-ptcone20_TightTTVALooseCone_pt500",
    ("met", 25),
    ("ptcone20_TightTTVALooseCone_pt500/lep1Pt", 0.06),
    reverse_y=True,
)
setting.reserve_abcd_regions(
    "met-ptvarcone20_TightTTVA_pt1000",
    ("met", 25),
    ("ptvarcone20_TightTTVA_pt1000/lep1Pt", 0.06),
    reverse_y=True,
)
setting.reserve_abcd_regions(
    "muon_met-ptvarcone20_TightTTVA_pt1000",
    ("met", 25),
    ("ptvarcone20_TightTTVA_pt1000/lep1Pt", 0.15),
    reverse_y=True,
)
setting.reserve_abcd_regions(
    "met-lep1Topoetcone20",
    ("met", 25),
    ("lep1Topoetcone20/lep1Pt", 0.06),
    reverse_y=True,
)
setting.reserve_abcd_regions(
    "muon_met-lep1Topoetcone20",
    ("met", 25),
    ("lep1Topoetcone20/lep1Pt", 0.3),
    reverse_y=True,
)

setting.create_abcd_regions(["met-lep1Signal"])
setting.create_abcd_regions(["met-IsoFCTight"])
setting.create_abcd_regions(["met-IsoFCLoose_FixedRad"])
setting.create_abcd_regions(["met-ptcone20_TightTTVA_pt1000"])
setting.create_abcd_regions(["muon_met-ptcone20_TightTTVA_pt1000"])
setting.create_abcd_regions(["met-ptcone20_TightTTVALooseCone_pt1000"])
setting.create_abcd_regions(["met-ptvarcone20_TightTTVA_pt1000"])
setting.create_abcd_regions(["muon_met-ptvarcone20_TightTTVA_pt1000"])
setting.create_abcd_regions(["met-lep1Topoetcone20"])
setting.create_abcd_regions(["muon_met-lep1Topoetcone20"])
setting.create_abcd_regions(["met-ptvarcone20_TightTTVA_pt1000", "met-lep1Signal"])
setting.create_abcd_regions(["met-ptcone20_TightTTVA_pt1000", "met-lep1Signal"])
setting.create_abcd_regions(["muon_met-ptcone20_TightTTVA_pt1000", "met-lep1Signal"])
setting.create_abcd_regions(
    ["met-ptcone20_TightTTVALooseCone_pt1000", "met-lep1Signal"]
)
setting.create_abcd_regions(["met-ptcone20_TightTTVALooseCone_pt500", "met-lep1Signal"])

setting.add_observable("met", 40, 0, 1000, "met [GeV]")
setting.add_observable("jet1Pt", 20, 50, 1500, "leading jet Pt [GeV]")
setting.add_observable("lep1Pt", 20, 30, 1500, "leading lepton Pt [GeV]")
setting.add_observable("lep1Eta", 10, -5, 5, "leading lep #eta ")
setting.add_observable("mt", 40, 0, 1000, "mt [GeV]")
setting.add_observable("wPt", 20, 0, 1500, "W Pt [GeV]")
setting.add_observable("nJet25", 10, 0, 10, "number of jet25")
setting.add_observable("DeltaPhiWJetClosest25", 20, 0, 5, "min #Delta#Phi(W,jet)")

setting.save("raw_config")
parsed_configMgr = run_HistMaker.run_HistMaker(setting)
parsed_configMgr.save("abcd_closure.pkl")

# hist manipulation

tf_configMgr = run_HistManipulate.run_ABCD_TF(parsed_configMgr, "dijets")

bin_tf_fakes = run_HistManipulate.run_ABCD_Fakes(tf_configMgr, False)
const_tf_fakes = run_HistManipulate.run_ABCD_Fakes(tf_configMgr, True)

# producing plots

run_PlotMaker.run_PlotABCD_TF(bin_tf_fakes, "Transfer Factor")

run_PlotMaker.run_PlotABCD(
    bin_tf_fakes,
    "bin_tf_fakes_plots",
    data_process="dijets",
    mc_process="",
    text="Closure Test",
)
run_PlotMaker.run_PlotABCD(
    const_tf_fakes,
    "const_tf_fakes_plots",
    data_process="dijets",
    mc_process="",
    text="Closure Test",
)
