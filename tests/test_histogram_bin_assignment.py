import pytest
import numpy as np
from physana import Histogram


def test_histogram_bin_assignment_check():
    """
    check bin content shape when assigning value to the bin_content attribute
    """

    # create Histogram object
    # the bin_content size here should be 25 + under/overflow bin
    m_hist = Histogram("jetPt", 3, 0, 100, "x title")

    # since we have only 3 bin over the given range,
    # the expected bin_content size will be 3 + 2(under/overflow)
    assert len(m_hist.bin_content) == 5
    assert m_hist.bin_content.shape == (5,)

    # create array with different size to test the shape checking during
    # bin_content assignment
    trial_1 = np.array([1, 2, 3])
    trial_2 = np.array([0, 1, 2, 3, 4, 5])

    # raise ValueError when the shape does not match
    with pytest.raises(ValueError):
        m_hist.bin_content = trial_1
    with pytest.raises(ValueError):
        m_hist.sumW2 = trial_1**2

    # shape matched
    with pytest.raises(ValueError):
        m_hist.bin_content = trial_2
    with pytest.raises(ValueError):
        m_hist.sumW2 = trial_2**2
