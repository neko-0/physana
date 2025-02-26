import pytest
import numpy as np

from collinearw import Process
from collinearw import Region
from collinearw import Histogram


def test_process_operators():
    lhs_process = Process("data", "data_NoSys")
    rhs_process = Process("wjets", "data_NoSys")
    rhs_process2 = Process("data", "data_NoSys")

    lhs_region = Region("lhs", "recoWeight", "nJet25>2")
    rhs_region = Region("rhs", "recoWeight", "nJet25>2")
    rhs_region2 = Region("lhs", "recoWeight", "nJet25>2")

    lbin_content = np.array([0, 1, 5, 7, 0])
    lsumW2 = np.array([0, 2, 4, 1, 0])

    rbin_content = np.array([0, 2, 4, 3, 0])
    rsumW2 = np.array([0, 3, 1, 2, 0])

    add_result_bin = lbin_content + rbin_content
    # add_result_bin_sumW2 = lsumW2 + rsumW2

    sub_result_bin = lbin_content - rbin_content
    # sub_result_bin_sumW2 = lsumW2 + rsumW2

    lhs_histo = Histogram("jetPt", 3, 0, 100, "x title")
    lhs_histo.bin_content = lbin_content
    lhs_histo.sumW2 = lsumW2
    lhs_region.append(lhs_histo)

    rhs_histo = Histogram("jetPt", 3, 0, 100, "x title")
    rhs_histo.bin_content = rbin_content
    rhs_histo.sumW2 = rsumW2
    rhs_histo2 = Histogram("lepPt", 3, 0, 100, "x title")
    rhs_histo2.bin_content = rbin_content
    rhs_histo2.sumW2 = rsumW2
    rhs_region.append(rhs_histo)
    rhs_region.append(rhs_histo2)
    rhs_region2.append(rhs_histo)
    rhs_region2.append(rhs_histo2)

    lhs_process.append(lhs_region)
    rhs_process.append(rhs_region)
    rhs_process2.append(rhs_region2)

    result_process = lhs_process + rhs_process
    assert result_process.name == f"{lhs_process.name}+{rhs_process.name}"
    assert "lhs" in result_process.list_regions()
    assert "rhs" in result_process.list_regions()
    assert (result_process.get("lhs").get("jetPt").bin_content == lbin_content).all()
    assert (result_process.get("rhs").get("jetPt").bin_content == rbin_content).all()
    assert (result_process.get("rhs").get("lepPt").bin_content == rbin_content).all()

    result_process = lhs_process + rhs_process2
    assert result_process.name == lhs_process.name
    assert "lhs" in result_process.list_regions()
    assert "rhs" not in result_process.list_regions()
    assert (result_process.get("lhs").get("jetPt").bin_content == add_result_bin).all()
    assert (result_process.get("lhs").get("lepPt").bin_content == rbin_content).all()

    result_process = lhs_process - rhs_process
    assert result_process.name == f"{lhs_process.name}-{rhs_process.name}"
    assert "lhs" in result_process.list_regions()
    assert "rhs" in result_process.list_regions()
    assert (result_process.get("lhs").get("jetPt").bin_content == lbin_content).all()
    assert (result_process.get("rhs").get("lepPt").bin_content == rbin_content).all()

    result_process = lhs_process - rhs_process2
    assert result_process.name == lhs_process.name
    assert "lhs" in result_process.list_regions()
    assert "rhs" not in result_process.list_regions()
    assert (result_process.get("lhs").get("jetPt").bin_content == sub_result_bin).all()
    assert (result_process.get("lhs").get("lepPt").bin_content == rbin_content).all()


def test_process_rescale():
    process = Process("test_process", "test")
    regionA = Region("test_regionA", "recoWeight", "nJet25>2", "test-regionA")
    regionB = Region("test_regionB", "recoWeight", "nJet25>2", "test-regionB")
    histogram = Histogram("test_histo", 3, 0, 100, "x title")

    bin_content = np.array([0, 1, 5, 7, 0])
    sumW2 = np.array([0, 2, 4, 1, 0])
    bin_content2 = np.array([0, 1, 5, 7, 0])
    sumW2_2 = np.array([0, 2, 4, 1, 0])
    histogram.bin_content = bin_content
    histogram.sumW2 = sumW2
    regionA.append(histogram)
    regionB.append(histogram)

    process.append(regionA)
    process.append(regionB)

    SF = 0.5

    scale_bin_content = bin_content2 * SF
    scale_bin_sumW2 = sumW2_2 * (SF**2)

    process.scale(SF, skip_type="regionA")

    process_0_0 = process.get("test_regionA").get("test_histo")
    process_1_0 = process.get("test_regionB").get("test_histo")

    assert (process_0_0.bin_content == bin_content2).all()
    assert (process_0_0.sumW2 == sumW2_2).all()
    assert (process_1_0.bin_content == scale_bin_content).all()
    assert (process_1_0.sumW2 == scale_bin_sumW2).all()


def test_process_repr():
    lhs_process = Process("data", "data_NoSys")
    assert str(lhs_process)


def test_invalid_process_type():
    with pytest.raises(AssertionError):
        Process("data", "data_NoSys", dtype="not_a_real_type")
