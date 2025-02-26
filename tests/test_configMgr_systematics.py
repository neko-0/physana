import pytest
from collinearw import ConfigMgr


def test_systematics_preparation():
    config = ConfigMgr()

    config.define_tree_systematics("tree1", ["up1", "down1"], sys_type="dummy_type")
    config.define_tree_systematics("tree2", ["up2", "down2"], sys_type="dummy_type")

    config.define_weight_systematics("weight1", ["PDF1", "PDF2"], sys_type="dummy_type")

    assert len(config.systematics) == 3
    assert "tree1" in config.systematics
    assert "weight1" in config.systematics
    assert len(config.systematics["tree1"]) == 2
    assert len(config.systematics["weight1"]) == 2

    config.add_process("ttbar")

    with pytest.raises(KeyError):
        # case where process is not added before set systematics
        config.set_systematics("wjets", "tree1")
        # case where the name of the systematics is not found
        config.set_systematics("ttbar", "tree10")
        # case where one of the systematics is not found
        config.set_systematics("ttbar", ["tree1", "tree10"])

    # no systematics for ttbar since we have not set it to any systematics.
    assert len(config.get_process_set("ttbar").systematics) == 0

    # set systematics and check the size of ProcessSet.systematics
    config.set_systematics("ttbar", "tree1")
    assert len(config.get_process_set("ttbar").systematics) == 2

    config.add_process("wjets")
    config.set_systematics("wjets", ["tree1", "weight1"])
    assert len(config.get_process_set("wjets").systematics) == 4

    # confirming ttbar remains the same
    assert len(config.get_process_set("ttbar").systematics) == 2


def test_systematics_duplication():
    config = ConfigMgr()
    config.define_tree_systematics("tree1", ["up1", "up2"], sys_type="dummpy_type")
    with pytest.raises(ValueError):
        config.define_tree_systematics("tree1", ["up1", "up2"], sys_type="dummpy_type")

    # the name of the tree systematics cannot be reused in weight systematics
    with pytest.raises(ValueError):
        config.define_weight_systematics(
            "tree1", ["PDF1", "PDF2"], sys_type="dummy_type"
        )
