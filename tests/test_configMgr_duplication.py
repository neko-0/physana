from physana import ConfigMgr


def test_duplicated_config_process():
    config = ConfigMgr()
    config.add_process("data")
    config.add_process("data")
    assert len(config.processes) == 1, "duplicated processes."


def test_duplicated_config_region():
    config = ConfigMgr()
    config.add_region("collinear", "minDeltPhiWJets<1.5")
    config.add_region("collinear", "")
    assert len(config.regions) == 1, "duplicated region."


def test_duplicated_config_observable():
    config = ConfigMgr()
    config.add_observable("wPt", 100, 0, 2000, "W pT")
    config.add_observable("wPt", 100, 0, 2000, "W pT")
    assert len(config.histograms) == 1, "duplicated observable."
