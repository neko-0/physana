from collinearw import ConfigMgr


def test_process_swapping():
    config = ConfigMgr()
    config.define_tree_systematics("tree1", ["up1", "down1"], sys_type="dummy_type")
    config.define_weight_systematics("weight1", ["PDF1", "PDF2"], sys_type="dummy_type")
    config.add_process("ttbar")
    config.set_systematics("ttbar", "tree1")
    config.add_process("wjets")
    config.set_systematics("wjets", ["tree1", "weight1"])
    config.add_region("region-1", "pt>500", weight="eventWeight")
    config.prepare(use_mp=False)

    config2 = ConfigMgr()
    config2.define_tree_systematics("tree1", ["up1", "down1"], sys_type="dummy_type")
    config2.define_weight_systematics(
        "weight2", ["PDF21", "PDF22"], sys_type="dummy_type"
    )
    config2.add_process("ttbar")
    config2.add_process("wjets")
    config2.set_systematics("wjets", ["tree1", "weight2"])
    config2.add_region("region-2", "pt>800", weight="eventWeight")
    config2.prepare(use_mp=False)

    # perform process swapping between configs
    config.swap_processes(config2)

    # only ttbar nominal will be swapped
    assert config.get("ttbar").nominal[0].name == "region-2"
    assert config.get("ttbar").get(("tree1", "up1", ""))[0].name == "region-1"
    assert config.get("ttbar").get(("tree1", "down1", ""))[0].name == "region-1"

    # wjets will nominal and matched systematics swapped
    assert config.get("wjets").nominal[0].name == "region-2"
    assert config.get("wjets").get(("tree1", "up1", ""))[0].name == "region-2"
    assert config.get("wjets").get(("tree1", "down2", ""))[0].name == "region-2"
    assert config.get("wjets").get(("weight1", "NoSys", "PDF1"))[0].name == "region-1"
    assert config.get("wjets").get(("weight1", "NoSys", "PDF2"))[0].name == "region-1"
    assert (
        config.get("wjets").get(("weight2", "NoSys", "PDF21"))
        is config.get("wjets").nominal
    )
    assert (
        config.get("wjets").get(("weight2", "NoSys", "PDF22"))
        is config.get("wjets").nominal
    )
