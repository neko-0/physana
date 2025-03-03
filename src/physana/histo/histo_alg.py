import numpy as np

from .region import Region
from .process import Process
from .process_set import ProcessSet


class HistogramAlgorithm:
    @staticmethod
    def stdev(hlist):
        """
        compute the std bin by bin for list of hitograms
        """
        # just copy the 1st histogram for all information
        m_hist = hlist[0].copy(shallow=True)
        if len(hlist) <= 1:
            return m_hist
        # bin content and error buffer
        # construct a len(hist.bin_content) x len(hist) dimension array
        bin_content_buffer = np.array([h.bin_content for h in hlist])
        # note we need proper treatment for error
        bin_error_buffer = np.array([h.sumW2 for h in hlist])

        m_hist.bin_content = np.std(bin_content_buffer, axis=0)
        m_hist.sumW2 = np.std(bin_error_buffer, axis=0)
        m_hist.type = "stdev"

        return m_hist

    @staticmethod
    def mean(hlist):
        """
        compute the mean bin by bin for list of hitograms
        """
        # just copy the 1st histogram for all information
        m_hist = hlist[0].copy(shallow=True)
        if len(hlist) <= 1:
            return m_hist
        # bin content and error buffer
        # construct a len(hist.bin_content) x len(hist) dimension array

        bin_content_buffer = np.array([h.bin_content for h in hlist])
        # note we need proper treatment for error
        bin_error_buffer = np.array([h.sumW2 for h in hlist])

        m_hist.bin_content = np.mean(bin_content_buffer, axis=0)
        m_hist.sumW2 = np.sum(bin_error_buffer, axis=0)
        m_hist.type = "mean"

        return m_hist

    @staticmethod
    def min_max(hlist):
        """
        compute the max/min bin by bin for list of hitograms
        """
        # just copy the 1st histogram for all information
        max_hist, min_hist = hlist[0].copy(shallow=True), hlist[0].copy(shallow=True)
        max_hist.type, min_hist.type = "max", "min"
        if len(hlist) <= 1:
            return min_hist, max_hist
        # bin content and error buffer
        # construct a len(hist.bin_content) x len(hist) dimension array
        content_cache = [h.bin_content for h in hlist]
        max_bin_content_buffer = np.maximum.reduce(content_cache)
        min_bin_content_buffer = np.minimum.reduce(content_cache)
        # note we need proper treatment for error
        max_bin_error_buffer = np.zeros(max_bin_content_buffer.shape)
        min_bin_error_buffer = np.zeros(min_bin_content_buffer.shape)

        max_hist.bin_content = max_bin_content_buffer
        max_hist.sumW2 = max_bin_error_buffer

        min_hist.bin_content = min_bin_content_buffer
        min_hist.sumW2 = min_bin_error_buffer

        return min_hist, max_hist


# ==============================================================================
def _iter_histograms_region(obj, htype_filter=None):
    for h in obj:
        if htype_filter and h.hist_type in htype_filter:
            continue
        yield h


def _iter_histograms_process(obj, rtype_filter=None):
    for r in obj:
        if rtype_filter and r.type in rtype_filter:
            continue
        yield from _iter_histograms_region(r)


def _iter_histograms_process_set(obj, skip_nominal=False):
    for p in obj:
        if p is obj.nominal and skip_nominal:
            continue
        yield from _iter_histograms_process(p)


def iter_histograms(obj, *args, **kwargs):
    if isinstance(obj, ProcessSet):
        yield from _iter_histograms_process_set(obj, *args, **kwargs)
    elif isinstance(obj, Process):
        yield from _iter_histograms_process(obj, *args, **kwargs)
    elif isinstance(obj, Region):
        yield from _iter_histograms_region(obj, *args, **kwargs)
    else:
        raise TypeError(f"invalid type {type(obj)}")
