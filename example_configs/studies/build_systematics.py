"""
Example of building config objects with systematics
"""
from collinearw import run_HistMaker, ConfigMgr
from collinearw.strategies import abcd


# path to the data files
src_path = "/nfs/slac/atlas/fs1/d/yuzhan/collinearw_files/Wj_AB212108_v3p3/download/mc16a/Wj_AB212108_v3p3/merged/"

# create configMgr instance
config = ConfigMgr()

# define and add process into config
config.add_process("diboson", filename=f"{src_path}/mc16a.root")

# define tree systematics
config.define_tree_systematics(
    "MET_SoftTrk",  # this is just a name you define for this type of systematics
    [
        "MET_SoftTrk_ResoPara",
        "MET_SoftTrk_ScaleDown",
        "MET_SoftTrk_ScaleUp",
    ],  # this is tree name in the ROOT file
    sys_type="stdev",  # latter for dealing with systematics, might need improvements as we go
    normalize=True,
    symmetrize=False,
)

# define weight systematics
config.define_weight_systematics(
    "LHE3Weight_MUR1_MUF1",  # just a name
    [
        f"LHE3Weight_MUR1_MUF1_PDF261{i:03}" for i in range(100)
    ],  # this is the branch name in the ttree
    sys_type="stdev",  #
)

# setting the systematic to process
config.set_systematics(
    "diboson",  # name of process you want to add systmatic
    [
        "LHE3Weight_MUR1_MUF1",
        "MET_SoftTrk",
    ],  # list of systematics to add to the process
)

# add regions
config.add_region(
    "my_signal_region",
    "isReco && nBJet25==0 && lep1Pt>30 && trigMatch_singleLepTrig && jet1Pt>500 && nLeptons==1",
)

# add some observables
config.add_observable("met", 40, 0, 1000, "met [GeV]")
config.add_observable("jet1Pt", 25, 50, 2500, "leading jet P_{T} [GeV]")
config.add_observable("lep1Pt", 20, 30, 1500, "leading lepton P_{T} [GeV]")
config.add_observable("lep1Pt", 34, 30, 2500, "leading lepton P_{T} [GeV]")
config.add_observable("lep1Eta", 10, -5, 5, "leading lepton #eta ")
config.add_observable("mt", 40, 0, 1000, "mt [GeV]")
config.add_observable("wPt", 50, 0, 1500, "W P_{T} [GeV]")

config.save(f"sys_raw_config")

# start filling config
parsed_config = run_HistMaker.run_HistMaker(config)  # , rsplit=False, n_workers=10)
parsed_config.save("run2_sys.pkl")
