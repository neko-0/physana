import pytest

from collinearw import ConfigMgr


def test_systematics_preparation():
    config = ConfigMgr()

    grp1 = [
        {"name": "JER", "syst_type": "exp", "tag": "1UP"},
        {"name": "JER", "syst_type": "exp", "tag": "1DOWN"},
    ]
    grp2 = [
        {"name": "PDF", "syst_type": "thoery", "tag": "PDF10000"},
    ]

    config.define_systematics_group(**grp1[0])
    config.define_systematics_group(**grp1[1])
    config.define_systematics_group(**grp2[0])

    assert len(config.systematics) == 2
    assert len(config.systematics["JER"]) == 2
    assert len(config.systematics["PDF"]) == 1

    config.add_process("ttbar")

    with pytest.raises(KeyError):
        # case where process is not added before set systematics
        config.set_systematics("wjets", "JER")
        # case where the name of the systematics is not found
        config.set_systematics("ttbar", "PDF")
        # case where one of the systematics is not found
        config.set_systematics("ttbar", ["JER", "JES"])

    # no systematics for ttbar since we have not set it to any systematics.
    assert len(config.get_process_set("ttbar").systematics) == 0

    # set systematics and check the size of ProcessSet.systematics
    config.set_systematics("ttbar", "JER")
    assert len(config.get_process_set("ttbar").systematics) == 2

    config.add_process("wjets")
    config.set_systematics("wjets", ["JER", "PDF"])
    assert len(config.get_process_set("wjets").systematics) == 3

    # confirming ttbar remains the same
    assert len(config.get_process_set("ttbar").systematics) == 2


def test_systematics_duplication():
    config = ConfigMgr()

    grp1 = [
        {"name": "JER", "syst_type": "exp", "tag": "1UP"},
    ]

    config.define_systematics_group(**grp1[0])

    with pytest.raises(ValueError):
        config.define_systematics_group(**grp1[0])
