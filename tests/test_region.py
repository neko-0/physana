import numpy as np

from collinearw import Histogram
from collinearw import Region


def test_region_add_operation():
    lhs_region = Region("lhs", "recoWeight", "nJet25>2")
    rhs_region = Region("rhs", "recoWeight", "nJet25>2")
    rhs_region2 = Region("lhs", "recoWeight", "nJet25>2")

    lbin_content = np.array([0, 1, 5, 7, 0])
    lsumW2 = np.array([0, 2, 4, 1, 0])

    rbin_content = np.array([0, 2, 4, 3, 0])
    rsumW2 = np.array([0, 3, 1, 2, 0])

    result_bin = lbin_content + rbin_content
    # result_bin_sumW2 = lsumW2 + rsumW2

    lhs_histo = Histogram("jetPt", 3, 0, 100, "x title")
    lhs_histo.bin_content = lbin_content
    lhs_histo.sumW2 = lsumW2
    lhs_region.append(lhs_histo)

    rhs_histo = Histogram("jetPt", 3, 0, 100, "x title")
    rhs_histo.bin_content = rbin_content
    rhs_histo.sumW2 = rsumW2
    rhs_histo2 = Histogram("lepPt", 3, 0, 100, "x title")
    rhs_histo2.bin_content = rbin_content
    rhs_region.append(rhs_histo)
    rhs_region.append(rhs_histo2)
    rhs_region2.append(rhs_histo)
    rhs_region2.append(rhs_histo2)

    result_region = lhs_region + rhs_region
    assert result_region.name == f"{lhs_region.name}+{rhs_region.name}"
    assert (result_region.get("jetPt").bin_content == result_bin).all()
    assert (result_region.get("lepPt").bin_content == rbin_content).all()

    result_region = lhs_region + rhs_region2
    assert result_region.name == lhs_region.name, "fail add"
    assert (result_region.get("jetPt").bin_content == result_bin).all()
    assert (result_region.get("lepPt").bin_content == rbin_content).all()

    c_lhs_region = lhs_region.copy()
    c_lhs_region.add(rhs_region2)
    assert c_lhs_region.name == lhs_region.name, "fail add"
    assert c_lhs_region.name == rhs_region2.name, "fail add"
    assert (c_lhs_region.get("jetPt").bin_content == result_bin).all()
    assert (c_lhs_region.get("lepPt").bin_content == rbin_content).all()

    c_lhs_region = lhs_region.copy()
    c_lhs_region.add(rhs_region)
    assert c_lhs_region.name != lhs_region.name, "fail add"
    assert c_lhs_region.name == f"{lhs_region.name}+{rhs_region.name}"
    assert (c_lhs_region.get("jetPt").bin_content == result_bin).all()
    assert (c_lhs_region.get("lepPt").bin_content == rbin_content).all()


def test_region_sub_operation():
    lhs_region = Region("lhs", "recoWeight", "nJet25>2")
    rhs_region = Region("rhs", "recoWeight", "nJet25>2")
    rhs_region2 = Region("lhs", "recoWeight", "nJet25>2")

    lbin_content = np.array([0, 1, 5, 7, 0])
    # lsumW2 = np.array([2, 4, 1])

    rbin_content = np.array([0, 2, 4, 3, 0])
    # rsumW2 = np.array([3, 1, 2])

    result_bin = lbin_content - rbin_content
    # result_bin_sumW2 = lsumW2 + rsumW2

    lhs_histo = Histogram("jetPt", 3, 0, 100, "x title")
    lhs_histo.bin_content = lbin_content
    lhs_region.append(lhs_histo)

    rhs_histo = Histogram("jetPt", 3, 0, 100, "x title")
    rhs_histo.bin_content = rbin_content
    rhs_histo2 = Histogram("lepPt", 3, 0, 100, "x title")
    rhs_histo2.bin_content = rbin_content
    rhs_region.append(rhs_histo)
    rhs_region.append(rhs_histo2)
    rhs_region2.append(rhs_histo)
    rhs_region2.append(rhs_histo2)

    result_region = lhs_region - rhs_region
    assert result_region.name == f"{lhs_region.name}-{rhs_region.name}"
    assert (result_region.get("jetPt").bin_content == result_bin).all()
    assert (result_region.get("lepPt").bin_content == rbin_content).all()

    result_region = lhs_region - rhs_region2
    assert result_region.name == lhs_region.name
    assert (result_region.get("jetPt").bin_content == result_bin).all()
    assert (result_region.get("lepPt").bin_content == rbin_content).all()

    c_lhs_region = lhs_region.copy()
    c_lhs_region.sub(rhs_region2)
    assert c_lhs_region.name == lhs_region.name
    assert c_lhs_region.name == rhs_region2.name
    assert (c_lhs_region.get("jetPt").bin_content == result_bin).all()
    assert (c_lhs_region.get("lepPt").bin_content == rbin_content).all()

    c_lhs_region = lhs_region.copy()
    c_lhs_region.sub(rhs_region)
    assert c_lhs_region.name != lhs_region.name
    assert c_lhs_region.name == f"{lhs_region.name}-{rhs_region.name}"
    assert (c_lhs_region.get("jetPt").bin_content == result_bin).all()
    assert (c_lhs_region.get("lepPt").bin_content == rbin_content).all()
