from collinearw import ConfigMgr, PlotMaker

configMgr = ConfigMgr.open(
    "/nfs/slac/atlas/fs1/d/yuzhan/configs_bk/slac_configs/IterativeCorrectionValidation4/with_stats_realData_wjets_Sherpa_wjets_fake_iter10.pkl"
)

plotMaker = PlotMaker(output_dir='./tfs')

plotMaker.plot_corrections(
    configMgr,
    [
        ('electron', 'wjets', 'nJet25'),
        ('electron', 'zjets', 'nJet25'),
        ('electron', 'ttbar', 'nJet25'),
    ],
)

plotMaker.plot_corrections(
    configMgr,
    [
        ('muon_IsoIp', 'wjets', 'nJet25'),
        ('muon_IsoIp', 'zjets', 'nJet25'),
        ('muon_IsoIp', 'ttbar', 'nJet25'),
    ],
)
