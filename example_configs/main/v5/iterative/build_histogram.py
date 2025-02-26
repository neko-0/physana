"""
Histogram ConfigMgr builder for Production Wj_AB212164_v4.
Use for deriving phase-space correction factors.
"""

from collinearw import ConfigMgr
from collinearw import run_HistMaker
from collinearw.strategies import abcd
import pathlib
import ROOT
import logging
import os
from dask_jobqueue import SLURMCluster
from dask.distributed import Client
from dask.distributed import as_completed as dask_as_completed

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
log = logging.getLogger(__name__)

workers = 100  # setting.count_object("process")
workers = 8 if workers < 8 else workers
account = "shared"
cluster = SLURMCluster(
    queue=account,
    walltime='01:00:00',
    project="collinear_Wjets",
    cores=2,
    processes=1,
    memory="32GB",
    job_extra=[f'--account={account}', f'--partition={account}'],
    local_directory="dask_output",
    log_directory="dask_logs",
    # death_timeout=36000,
    n_workers=workers,
    nanny=False,
    # extra=["--resources 'process=1'"],
)
print(cluster.job_script())
client = Client(cluster)
client.get_versions(check=True)


def main():
    # src_path = "/gpfs/slac/atlas/fs1/d/yuzhan/collinearw_files/Wj_AB212164_v4c/merged_files/"
    src_path = (
        "/gpfs/slac/atlas/fs1/d/yuzhan/collinearw_files/Wj_AB212164_v5/merged_files/"
    )

    setting = ConfigMgr(src_path)

    output_path = pathlib.Path(os.path.realpath(__file__)).parent
    setting.set_output_location(output_path)

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

    # Common phase-space cuts
    common_cuts = "isReco && lep1Pt>30.0 && trigMatch_singleLepTrig && jet1Pt>=500.0"

    setting.add_process("data", process_type="data", color=ROOT.kBlack)
    # setting.add_process("wjets", color=ROOT.kAzure + 1, selection="DatasetNumber!=364197 && DatasetNumber!=364196 && DatasetNumber!=364193 && DatasetNumber!=364185 && DatasetNumber!=364186 && DatasetNumber!=364184 && DatasetNumber!=364187 && DatasetNumber!=364189 && DatasetNumber!=364188 && DatasetNumber!=364191 && DatasetNumber!=364190 && DatasetNumber!= 364192 && DatasetNumber!=364195 && DatasetNumber!=364194")
    # setting.add_process("wjets", process_type="signal", color=ROOT.kAzure + 1)
    # setting.add_process("wjets_mg", process_type="signal_alt", color=ROOT.kAzure + 1)
    setting.add_process("wjets_2211", process_type="signal", color=ROOT.kAzure + 1)
    # setting.add_process("wjets_mg", color=ROOT.kAzure + 1, selection="DatasetNumber!=363659 && DatasetNumber!=363658 && DatasetNumber!=363657 && DatasetNumber!=363656 && DatasetNumber!=363655 && DatasetNumber!=363654 && DatasetNumber!=363653 && DatasetNumber!=363652 && DatasetNumber!=363651 && DatasetNumber!=363650 && DatasetNumber!=363649 && DatasetNumber!=363648")
    # setting.add_process("wjets_FxFx", color=ROOT.kAzure + 1)
    # setting.add_process("zjets", color=ROOT.kAzure - 1)
    setting.add_process("zjets_2211", color=ROOT.kAzure - 1)
    setting.add_process("ttbar", color=ROOT.kGreen - 9)
    setting.add_process("singletop", color=ROOT.kGreen - 5)
    setting.add_process("diboson", color=ROOT.kOrange)

    # Electron selections
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

    # Muon selections
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
    '''
    setting.add_region(
        "WjetsCR_Ele",
        f"{common_cuts} && nBJet30==0 && nLeptons==1 && AnalysisType==1  && {'&&'.join(electron_1lepton)}",
        corr_type="electron",
    )
    '''

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
        f"{common_cuts} && nBJet30==0 && nLeptons==2 && AnalysisType==2 && diLeptonMass>60.0 && diLeptonMass<120.0 && {'&&'.join(muon_1lepton)}",
        corr_type="muon",
    )
    setting.add_region(
        "muon",
        f"{common_cuts} && nBJet30==0 && nLeptons==1 && AnalysisType==2 && met>25",
        corr_type="muon",
    )

    # ===============================================================================

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
        ["PID", "fake-electron", "et-cone"],
        base_region_names=["electron"],
    )

    abcd.create_abcd_regions(setting, ["fake-muon"], base_region_names=["muon"])

    setting.add_observable("lep1Phi", 15, -3, 3, "lepton #phi")
    setting.add_observable("nJet30", 9, 1, 10, "Jet multiplicity (p_{T}>30 GeV)")
    setting.add_observable("met", 50, 0, 500, "E_{T}^{miss} [GeV]")
    setting.add_observable("jet1Pt", 20, 500, 1500, "Leading jet p_{T} [GeV]")
    setting.add_observable("lep1Pt", 10, 30, 630, "Leading lepton p_{T} [GeV]")
    setting.add_observable("lep1Eta", 10, -2.5, 2.5, "Leading lepton #eta ")
    setting.add_observable("mt", 30, 0, 300, "m_{T} [GeV]")
    setting.add_observable("mjj", 30, 0, 3000, "m_{jj} [GeV]")
    setting.add_observable("wPt", 24, 0, 1200, "p_{T}(lepton+MET) [GeV]")
    setting.add_observable("diLeptonPt", 20, 0, 1000, "Z p_{T} [GeV]")
    setting.add_observable("diLeptonMass", 20, 60, 120, "m_{ll} [GeV]")
    setting.add_observable("Ht30", 25, 500, 3000, "H_{T}^{30} [GeV]")

    setting.add_observable(
        "DeltaPhiWJetClosest30", 20, 0, 2, "min(#Delta#phi(W,jet_{i}^{30}))"
    )
    setting.add_observable(
        "DeltaRLepJetClosest30", 20, 0, 2, "min(#DeltaR(l,jet_{i}^{30}))"
    )
    setting.add_observable(
        "DeltaPhiMetJetClosest30", 20, 0, 2, "min(#Delta#phi(W,jet_{i}^{30}))"
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

    # setting.load_systematics("../minimum_sys.jsonnet")#, create_dummy=["wjets_2211", "zjets_2211", "ttbar", "diboson", "singletop"])
    setting.enforce_default_weight = True

    setting.prepare(True)

    setting.save("run2_raw_config")

    # flag for raising error when tree does not exist
    setting.RAISE_TREENAME_ERROR = False

    setting = run_HistMaker.run_HistMaker(
        setting, split_type="region", executor=client, as_completed=dask_as_completed
    )

    setting.save("run2")


if __name__ == "__main__":
    main()
