from collinearw import ConfigMgr, HistManipulate, PlotMaker
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
    "./TopCRShape/",  # output
    "ABCD_BG",
    "_Wj_AB21298_v2.root",  # file_suffix
    "uproot",  # io backend
    "numpy",  # histograms backend
)

if os.path.isfile(f"{configMgr.out_path}/{write_name}"):
    pconfigMgr = ConfigMgr.open(f"{configMgr.out_path}/{write_name}")
else:

    # overwirte the file name if you only have one file with everything.
    configMgr.set_singlefile("Wj_AB212108_v2_run2.root")

    # Common phase-space cuts
    common_cuts = "isReco && lep1Pt>30.0 && trigMatch_singleLepTrig && jet1Pt>=500.0"

    # name, treename, selection, process_type
    configMgr.add_process(
        "ttbar-veto", "ttbar_NoSys", "mc", ROOT.kBlack, selection="nBJet25==0"
    )
    configMgr.add_process(
        "ttbar-veto-Herwig7",
        "ttbar_PowhegHerwig713_NoSys",
        "mc",
        ROOT.kRed,
        selection="nBJet25==0",
    )
    configMgr.add_process(
        "ttbar-veto-aMC@NLO",
        "ttbar_aMcAtNloPy8_NoSys",
        "mc",
        ROOT.kGreen,
        selection="nBJet25==0",
    )
    configMgr.add_process(
        "ttbar-CR", "ttbar_NoSys", "mc", ROOT.kBlue, selection="nBJet25>=2"
    )

    # Regions
    configMgr.add_region('InclusivePhaseSpace', f"{common_cuts}")

    # Observables
    configMgr.add_observable("met", 40, 0, 1000, "E_{T}^{miss} [GeV]")
    configMgr.add_observable("jet1Pt", 20, 500, 1500, "Leading jet p_{T} [GeV]")
    configMgr.add_observable("lep1Pt", 15, 30, 630, "Leading lepton p_{T} [GeV]")
    configMgr.add_observable("lep1Eta", 10, -2.5, 2.5, "Leading lepton #eta ")
    configMgr.add_observable("mt", 30, 0, 300, "m_{T} [GeV]")
    configMgr.add_observable("wPt", 24, 0, 1200, "p_{T}(lepton+MET) [GeV]")
    configMgr.add_observable("nJet25", 9, 1, 10, "Jet multiplicity (p_{T}>25 GeV)")
    configMgr.add_observable(
        "DeltaPhiWJetClosest25", 20, 0, 2, "min(#Delta#phi(W,jet_{i}))"
    )
    configMgr.add_observable(
        "DeltaPhiWJetClosest30", 20, 0, 2, "min(#Delta#phi(W,jet_{i}))"
    )
    configMgr.add_observable(
        "DeltaPhiWJetClosest100", 32, 0, 3.2, "min(#Delta#phi(W,jet_{i}))"
    )
    configMgr.add_observable(
        "DeltaRLepJetClosest100", 32, 0, 3.2, "min(#Delta#phi(W,jet_{i}))"
    )

    # Prepare histograms and write
    configMgr.save("run2_raw_config")
    run_HistMaker.run_HistMaker(configMgr, "run2.pkl").save()
    pconfigMgr = ConfigMgr.open(f"{configMgr.out_path}/{write_name}")


plotMaker = PlotMaker(pconfigMgr)
plotMaker.draw_cmp_process(normalize=True)
