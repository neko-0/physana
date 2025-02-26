from collinearw import ConfigMgr
from collinearw import TableMaker

configMgr = ConfigMgr.open(
    "/nfs/slac/atlas/fs1/d/yuzhan/configs_bk/slac_configs/IterativeCorrectionValidation4/with_stats_realData_wjets_Sherpa_wjets_fake_iter10.pkl"
)
tm = TableMaker("Tables")

processNameMap = {
    'wjets': 'W+jets',
    'zjets': 'Z+jets',
    'diboson': 'Diboson',
    'ttbar': '$t\\bar{t}$',
    'singletop': 'Single top',
}

regions_ele = {'ttbarCR_Ele', 'ZjetsCR_Ele', 'WjetsCR_Ele'}
regionNameMap_ele = {
    'ttbarCR_Ele': '$t\\bar{t} CR$',
    'ZjetsCR_Ele': 'Z+jets CR',
    'WjetsCR_Ele': "W+jets preselection",
}

tm.makeTables(
    configMgr,
    regions=regions_ele,
    processes=['wjets', 'zjets', 'diboson', 'ttbar', 'singletop'],
    regionNameMap=regionNameMap_ele,
    processNameMap=processNameMap,
)

regions_mu = {'ttbarCR_Mu_muon_IpIso', 'ZjetsCR_Mu_muon_IpIso', 'muon_IsoIpPid'}
regionNameMap_mu = {
    'ttbarCR_Mu_muon_IpIso': '$t\\bar{t} CR$',
    'ZjetsCR_Mu_muon_IpIso': 'Z+jets CR',
    'muon_IsoIpPid': "W+jets preselection",
}

tm.makeTables(
    configMgr,
    regions=regions_mu,
    processes=['wjets', 'zjets', 'diboson', 'ttbar', 'singletop'],
    regionNameMap=regionNameMap_mu,
    processNameMap=processNameMap,
)
