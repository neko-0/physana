import numpy as np
import copy
from collinearw import HistManipulate
from collinearw import ConfigMgr, Process, Region, Histogram


def test_subtract_mc():

    config = ConfigMgr()

    data = Process("data", "data")
    wjets = Process("wjets", "wjets_NoSys")

    data_region1 = Region("region1", "recoWeight", "nJet25>2")
    data_region2 = Region("region2", "recoWeight", "nJet25<2")

    wjets_region1 = Region("region1", "recoWeight", "nJet25>2")
    wjets_region2 = Region("region2", "recoWeight", "nJet25<2")

    wjets_region1_jetPt = np.array([0, 0, 1, 5, 7, 0])
    data_region1_jetPt = np.array([0, 0, 2, 4, 3, 0])

    wjets_region2_jetPt = np.array([1, 0, 6, 4, 1, 0])
    data_region2_jetPt = np.array([1, 0, 2, 4, 7, 0])

    wjets_region1_jetPt_histo = Histogram("jetPt", 4, 0, 100, "x title")
    wjets_region1_jetPt_histo.bin_content = copy.deepcopy(wjets_region1_jetPt)

    data_region1_jetPt_histo = Histogram("jetPt", 4, 0, 100, "x title")
    data_region1_jetPt_histo.bin_content = copy.deepcopy(data_region1_jetPt)

    wjets_region2_jetPt_histo = Histogram("jetPt", 4, 0, 100, "x title")
    wjets_region2_jetPt_histo.bin_content = copy.deepcopy(wjets_region2_jetPt)

    data_region2_jetPt_histo = Histogram("jetPt", 4, 0, 100, "x title")
    data_region2_jetPt_histo.bin_content = copy.deepcopy(data_region2_jetPt)

    data_region1.add_histogram(data_region1_jetPt_histo)
    data_region2.add_histogram(data_region2_jetPt_histo)

    wjets_region1.add_histogram(wjets_region1_jetPt_histo)
    wjets_region2.add_histogram(wjets_region2_jetPt_histo)

    data.add_region(data_region1)
    data.add_region(data_region2)

    wjets.add_region(wjets_region1)
    wjets.add_region(wjets_region2)

    config.append_process(data)
    config.append_process(wjets)

    sub_config = HistManipulate.Subtract_MC(
        config, "data", "sub_data", remove_neg=False
    )

    assert "sub_data" in sub_config._processes_dict
    assert "sub_data" not in config._processes_dict

    expect_region1_jetPt = data_region1_jetPt - wjets_region1_jetPt

    expect_region2_jetPt = data_region2_jetPt - wjets_region2_jetPt

    sub_data_r1 = sub_config.get_process("sub_data").get_region("region1")
    sub_data_r2 = sub_config.get_process("sub_data").get_region("region2")

    sub_data_r1_hist = sub_data_r1.get_histogram("jetPt").bin_content
    sub_data_r2_hist = sub_data_r2.get_histogram("jetPt").bin_content

    data_r1 = config.get_process("data").get_region("region1")
    data_r2 = config.get_process("data").get_region("region2")

    data_r1_hist = data_r1.get_histogram("jetPt").bin_content
    data_r2_hist = data_r2.get_histogram("jetPt").bin_content

    wjets_r1 = sub_config.get_process("wjets").get_region("region1")
    wjets_r2 = sub_config.get_process("wjets").get_region("region2")

    wjets_r1_hist = wjets_r1.get_histogram("jetPt").bin_content
    wjets_r2_hist = wjets_r2.get_histogram("jetPt").bin_content

    assert (sub_data_r1_hist == expect_region1_jetPt).all()

    assert (sub_data_r2_hist == expect_region2_jetPt).all()

    assert (data_r1_hist == data_region1_jetPt).all()
    assert (data_r2_hist == data_region2_jetPt).all()

    assert (wjets_r1_hist == wjets_region1_jetPt).all()
    assert (wjets_r2_hist == wjets_region2_jetPt).all()
