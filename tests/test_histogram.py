import pytest
import numpy as np
import sys

from collinearw import Histogram
from collinearw import Histogram2D

try:
    import ROOT
except ImportError:
    pytest.skip("ROOT module not imported", allow_module_level=True)


def test_histogram_addition():
    """Test Histogram addition."""
    # Create Histogram objects
    lhist = Histogram("lhs", 3, 0, 100, "x title")
    rhist = Histogram("rhs", 3, 0, 100, "x title")

    # Set bin_content and sumW2
    lbin_content = np.array([0, 1, 5, 7, 0])
    lsumW2 = np.array([0, 2, 4, 1, 0])
    rbin_content = np.array([0, 2, 4, 3, 0])
    rsumW2 = np.array([0, 3, 1, 2, 0])

    lhist.bin_content = lbin_content.copy()
    lhist.sumW2 = lsumW2.copy()
    rhist.bin_content = rbin_content.copy()
    rhist.sumW2 = rsumW2.copy()

    # Test Histogram addition
    result = lhist + rhist
    assert (result.bin_content == lbin_content + rbin_content).all()
    assert (result.sumW2 == lsumW2 + rsumW2).all()

    # Test in-place addition
    lhist.add(rhist)
    assert (lhist.bin_content == lbin_content + rbin_content).all()
    assert (lhist.sumW2 == lsumW2 + rsumW2).all()

    # Test scalar addition
    neg_const = -10
    rhist.bin_content = rbin_content.copy()
    rhist.sumW2 = rsumW2.copy()

    result_neg_const = rbin_content + neg_const
    result_neg_const_sumW2 = rsumW2.copy()
    rhist.add(neg_const)
    assert (rhist.bin_content == result_neg_const).all()
    assert (rhist.sumW2 == result_neg_const_sumW2).all()

    pos_const = 10
    result_pos_const = result_neg_const + pos_const
    rhist.add(pos_const)
    assert (rhist.bin_content == result_pos_const).all()
    assert (rhist.sumW2 == result_neg_const_sumW2).all()

    # Test invalid input types
    with pytest.raises(TypeError):
        lhist.add("I am histogram!")
    with pytest.raises(TypeError):
        lhist.add((1, 2, 3))
    with pytest.raises(TypeError):
        lhist.add([1, 2, 3])


def test_histogram_subtraction():
    """Test Histogram subtraction."""
    # Create Histogram objects
    lhist = Histogram("lhs", 3, 0, 100, "x title")
    rhist = Histogram("rhs", 3, 0, 100, "x title")

    # Set bin_content and sumW2
    lbin_content = np.array([0, 1, 5, 7, 0])
    lsumW2 = np.array([0, 2, 4, 1, 0])
    rbin_content = np.array([0, 2, 4, 3, 0])
    rsumW2 = np.array([0, 3, 1, 2, 0])

    lhist.bin_content = lbin_content.copy()
    lhist.sumW2 = lsumW2.copy()
    rhist.bin_content = rbin_content.copy()
    rhist.sumW2 = rsumW2.copy()

    # Test Histogram subtraction
    result = lhist - rhist
    assert (result.bin_content == lbin_content - rbin_content).all()
    assert (result.sumW2 == lsumW2 + rsumW2).all()

    # Test in-place subtraction
    lhist.sub(rhist)
    assert (lhist.bin_content == lbin_content - rbin_content).all()
    assert (lhist.sumW2 == lsumW2 + rsumW2).all()

    # Test scalar subtraction
    neg_const = -10
    rhist.bin_content = rbin_content.copy()
    rhist.sumW2 = rsumW2.copy()

    result_neg_const = rbin_content - neg_const
    result_neg_const_sumW2 = rsumW2.copy()
    rhist.sub(neg_const)
    assert (rhist.bin_content == result_neg_const).all()
    assert (rhist.sumW2 == result_neg_const_sumW2).all()

    pos_const = 10
    result_pos_const = result_neg_const - pos_const
    rhist.sub(pos_const)
    assert (rhist.bin_content == result_pos_const).all()
    assert (rhist.sumW2 == result_neg_const_sumW2).all()

    # Test invalid input types
    with pytest.raises(TypeError):
        lhist.sub("I am histogram!")
    with pytest.raises(TypeError):
        lhist.sub((1, 2, 3))
    with pytest.raises(TypeError):
        lhist.sub([1, 2, 3])


@pytest.mark.skipif("ROOT" not in sys.modules, reason="ROOT module not imported")
def test_histogram_root_cache():
    """Test that histogram.root is cached."""
    histogram = Histogram("test", 1, 0, 1, "x title")
    root_histogram = histogram.root
    # check if another call to `root` returns the cached histogram or not
    assert root_histogram is histogram.root

    # Ensure that histograms with the same name but different properties
    # are not cached identically
    another_histogram = Histogram("test", 10, -10, 10, "a different title")
    assert root_histogram is not another_histogram.root


@pytest.mark.skipif("ROOT" not in sys.modules, reason="ROOT module not imported")
def test_histogram_from_root():
    root_hist = ROOT.TH1F("hist_from_root", "hist_from_root", 10, 0, 1)
    for i in range(12):
        root_hist.SetBinContent(i, i**2.0)

    hist = Histogram("hist_from_root", 10, 0, 1, "x title")
    hist.root = root_hist

    expected_bin_content = [float(i**2) for i in range(12)]
    assert hist.bin_content.tolist() == expected_bin_content

    bin2d_args = (2, 0, 1, 1, 0, 1)
    root_hist2d = ROOT.TH2F("hist2d_from_root", "hist2d_from_root", *bin2d_args)
    for coord in np.ndindex((4, 3)):
        root_hist2d.SetBinContent(*coord, np.square(coord).sum())

    hist2d = Histogram2D("hist2d_from_root", "xvar", "yvar", *bin2d_args)
    hist2d.root = root_hist2d

    expected_bin_content_2d = [
        [0.0, 1.0, 4.0],
        [1.0, 2.0, 5.0],
        [4.0, 5.0, 8.0],
        [9.0, 10.0, 13.0],
    ]
    assert hist2d.bin_content.tolist() == expected_bin_content_2d
