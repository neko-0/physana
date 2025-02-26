from collinearw import ConfigMgr


def test_process_swapping():
    config = ConfigMgr()

    config.define_systematics_group("sys1", "exp", "up1")
    config.define_systematics_group("sys1", "exp", "up2")

    config.define_systematics_group("sys2", "theory", "PDF1")
    config.define_systematics_group("sys2", "theory", "PDF2")

    config.add_process("ttbar")

    config.set_systematics("ttbar", "sys1")

    config.add_process("wjets")

    config.set_systematics("wjets", ["sys1", "sys2"])

    config.add_region("region-1", "pt>500", weight="eventWeight")

    config.prepare(use_mp=False)

    config2 = ConfigMgr()

    config2.define_systematics_group("sys2", "theory", "PDF1")
    config2.define_systematics_group("sys2", "theory", "PDF2")

    config2.define_systematics_group("sys3", "exp", "up1")
    config2.define_systematics_group("sys3", "exp", "down1")

    config2.add_process("ttbar")
    config2.add_process("wjets")

    config2.set_systematics("wjets", ["sys2", "sys3"])

    config2.add_region("region-2", "pt>800", weight="eventWeight")

    config2.prepare(use_mp=False)

    # perform process swapping between configs
    config.swap_processes(config2)

    ttbar_nominal_regions = list(config.get("ttbar").nominal)
    wjets_nominal_regions = list(config.get("wjets").nominal)
    ttbar_regions = lambda x: list(config.get("ttbar").get(x))
    wjets_regions = lambda x: list(config.get("wjets").get(x))

    # only ttbar nominal will be swapped
    assert ttbar_nominal_regions[0].name == "region-2"
    assert ttbar_regions(("sys1", "exp", "up1"))[0].name == "region-1"
    assert ttbar_regions(("sys1", "exp", "up2"))[0].name == "region-1"

    # wjets will have nominal and matched systematics swapped
    assert wjets_nominal_regions[0].name == "region-2"
    assert wjets_regions(("sys1", "exp", "up1"))[0].name == "region-1"
    assert wjets_regions(("sys1", "exp", "up2"))[0].name == "region-1"
    assert wjets_regions(("sys2", "theory", "PDF1"))[0].name == "region-2"
    assert wjets_regions(("sys2", "theory", "PDF2"))[0].name == "region-2"
    assert (
        config.get("wjets").get(("sys4", "theory", "PDF21"))
        is config.get("wjets").nominal
    )
    assert (
        config.get("wjets").get(("sys4", "theory", "PDF22"))
        is config.get("wjets").nominal
    )
