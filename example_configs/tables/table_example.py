from collinearw import ConfigMgr
from collinearw import TableMaker

configMgr = ConfigMgr.open("./Yields/run2.pkl")
tm = TableMaker("Tables")

processNameMap = {
    'wjets': 'W+jets',
    'zjets': 'Z+jets',
    'diboson': 'Diboson',
    'ttbar': '$t\\bar{t}$',
    'singletop': 'Single top',
}

regionNameMap = {'Inclusive': 'Inclusive selection', 'BVeto': 'B-jet veto'}

# B-jet veto effect
regions = {'Inclusive', 'BVeto'}
tm.makeTables(
    configMgr,
    regions=regions,
    regionNameMap=regionNameMap,
    processNameMap=processNameMap,
)
