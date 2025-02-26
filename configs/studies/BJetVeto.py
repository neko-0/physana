from collinearw import ConfigMgr, HistManipulate
from collinearw import run_HistMaker
from collinearw import run_HistManipulate
from collinearw import run_PlotMaker

import logging
import os
import ROOT

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
log = logging.getLogger(__name__)

write_name = 'run2.pkl'

configMgr = ConfigMgr(
    "/nfs/slac/atlas/fs1/d/yuzhan/collinearw_files/June2020_Production/merged_files/",
    "./BJetStudies_ABCD/",  # output
    "ABCD_BG",
    "_Wj_AB21298_v2.root",  # file_suffix
    "uproot",  # io backend
    "numpy",  # histograms backend
)

if os.path.isfile(f"{configMgr.out_path}/{write_name}"):
    pconfigMgr = ConfigMgr.open(f"{configMgr.out_path}/{write_name}")
    pconfigMgr.out_path = f"{configMgr.out_path}/plots/"

    if not os.path.exists(pconfigMgr.out_path):
        os.makedirs(pconfigMgr.out_path)

else:

    # overwirte the file name if you only have one file with everything.
    configMgr.set_singlefile("Wj_AB212108_v2_run2.root")

    # Common phase-space cuts
    common_cuts = "isReco && lep1Pt>30.0 && trigMatch_singleLepTrig && jet1Pt>=500.0"

    ele_cuts = '1'  #'(AnalysisType==1 && lep1Signal==1 && ptcone20_TightTTVALooseCone_pt500/lep1Pt<0.06)'
    muon_cuts = '1'  #'(AnalysisType==2 && lep1Signal==1 && IsoFCLoose_FixedRad==1)'

    # name, treename, selection, process_type
    configMgr.add_process("data", "data", "data", ROOT.kBlack)
    configMgr.add_process("wjets", "wjets_NoSys", "mc", ROOT.kAzure + 1)
    configMgr.add_process("ttbar", "ttbar_NoSys", "mc", ROOT.kGreen - 9)
    configMgr.add_process("zjets", "zjets_NoSys", "mc", ROOT.kAzure - 1)
    configMgr.add_process("singletop", "singletop_NoSys", "mc", ROOT.kGreen - 5)
    configMgr.add_process("diboson", "diboson_NoSys", "mc", ROOT.kOrange)

    # Regions
    # configMgr.add_region(
    #    'Inclusive_Ele', f"{common_cuts}" , study_type="abcd"
    #    )

    configMgr.add_region('BTag_Ele', f"{common_cuts} && nBJet25>=2", study_type="abcd")

    # configMgr.add_region(
    #    'BVeto_Ele', f"{common_cuts} && nBJet25==0" , study_type="abcd"
    #    )

    # configMgr.add_region(
    #    'Inclusive_Mu', f"{common_cuts}" , study_type="abcd"
    #    )

    configMgr.add_region('BTag_Mu', f"{common_cuts} && nBJet25>=2", study_type="abcd")

    # configMgr.add_region(
    #    'BVeto_Mu', f"{common_cuts} && nBJet25==0" , study_type="abcd"
    #    )

    # Observables
    configMgr.add_observable("met", 40, 0, 1000, "E_{T}^{miss} [GeV]")
    configMgr.add_observable("jet1Pt", 20, 500, 1500, "Leading jet p_{T} [GeV]")
    configMgr.add_observable("lep1Pt", 15, 30, 630, "Leading lepton p_{T} [GeV]")
    configMgr.add_observable("lep1Eta", 10, -2.5, 2.5, "Leading lepton #eta ")
    configMgr.add_observable("mt", 30, 0, 300, "m_{T} [GeV]")
    configMgr.add_observable("mjj", 20, 0, 2000, "m_{jj} [GeV]")
    configMgr.add_observable("wPt", 24, 0, 1200, "p_{T}(lepton+MET) [GeV]")
    configMgr.add_observable("nJet25", 9, 1, 10, "Jet multiplicity (p_{T}>25 GeV)")
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

    configMgr.reserve_abcd_regions("PID", ("met", 25), ("lep1Signal", 1))
    configMgr.reserve_abcd_regions("MuonIso", ("met", 25), ("IsoFCLoose_FixedRad", 1))
    configMgr.reserve_abcd_regions(
        "ElectronIso",
        ("met", 25),
        ("ptcone20_TightTTVALooseCone_pt500/lep1Pt", 0.06),
        reverse_y=True,
    )

    configMgr.create_abcd_regions(["PID", "ElectronIso"])
    configMgr.create_abcd_regions(["PID", "MuonIso"])

    configMgr.save("run2_raw_config")
    run_HistMaker.run_HistMaker(configMgr, "run2.pkl").save()
    pconfigMgr = ConfigMgr.open(f"{configMgr.out_path}/{write_name}")

# Make plots
subed_run2 = HistManipulate.Subtract_MC(pconfigMgr, "data", "subtracted_data")
tf_configMgr = run_HistManipulate.run_ABCD_TF(subed_run2, "subtracted_data")
bin_tf_fakes = run_HistManipulate.run_ABCD_Fakes(tf_configMgr, False)
const_tf_fakes = run_HistManipulate.run_ABCD_Fakes(tf_configMgr, True)

# run_PlotMaker.run_PlotABCD_TF(bin_tf_fakes, "Transfer Factor")

run_PlotMaker.run_PlotABCD(
    bin_tf_fakes,
    "BTagModelling",
    data_process="data",
    low_yrange=(0.5, 1.5),
    text="Fake Estimation with Run2",
)
