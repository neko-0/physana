import numpy as np
from numba import jit, prange


@jit(nopython=True, cache=True, parallel=True)
def apply_phsp_correction(w, sumW2, phsp, phsp_err):
    n = len(w)
    for i in prange(n):
        sumW2[i] = sumW2[i] * phsp[i] ** 2 + (w[i] * phsp_err[i]) ** 2
        w[i] *= phsp[i]


@jit(nopython=True, cache=True)
def jit__add__(lcontent, lsumW2, rcontent, rsumW2):
    n = len(lcontent)
    output_content, output_sumW2 = np.empty(n, np.double), np.empty(n, np.double)
    for i in prange(n):
        output_content[i] = lcontent[i] + rcontent[i]
        output_sumW2[i] = lsumW2[i] + rsumW2[i]
    return output_content, output_sumW2


@jit(nopython=True, cache=True)
def jit__sub__(lcontent, lsumW2, rcontent, rsumW2):
    n = len(lcontent)
    output_content, output_sumW2 = np.empty(n, np.double), np.empty(n, np.double)
    for i in prange(n):
        output_content[i] = lcontent[i] - rcontent[i]
        output_sumW2[i] = lsumW2[i] + rsumW2[i]
    return output_content, output_sumW2


@jit(nopython=True, cache=True)
def hadd(lcontent, lsumW2, rcontent, rsumW2):
    for i in prange(len(lcontent)):
        lcontent[i] += rcontent[i]
        lsumW2[i] += rsumW2[i]


@jit(nopython=True, cache=True)
def hsub(lcontent, lsumW2, rcontent, rsumW2):
    for i in prange(len(lcontent)):
        lcontent[i] -= rcontent[i]
        lsumW2[i] += rsumW2[i]


@jit(nopython=True, cache=True)
def hmul(lcontent, lsumW2, rcontent, rsumW2):
    for i in prange(len(lcontent)):
        lsumW2[i] *= rcontent[i] ** 2
        lsumW2[i] += rsumW2[i] * lcontent[i] ** 2
        lcontent[i] *= rcontent[i]


@jit(nopython=True, cache=True)
def hdiv(lcontent, lsumW2, rcontent, rsumW2):
    for i in prange(len(lcontent)):
        lsumW2[i] /= lcontent[i] ** 2
        lsumW2[i] += rsumW2[i] / rcontent[i] ** 2
        lcontent[i] /= rcontent[i]
        lsumW2[i] *= lcontent[i] ** 2


@jit(nopython=True, cache=True, parallel=True, nogil=True)
def np_digitize(x, edge):
    return np.digitize(x, edge)


@jit(nopython=True, cache=True, parallel=True, nogil=True)
def find_content_and_error_1d(x, edge, bin_content, bin_sumW2):
    _digitized = np.digitize(x, edge)
    # return bin_content[_digitized], np.sqrt(bin_sumW2[_digitized])
    n = len(x)
    output_content, output_error = np.empty(n, np.double), np.empty(n, np.double)
    for i in prange(n):
        output_content[i] = bin_content[_digitized[i]]
        output_error[i] = np.sqrt(bin_sumW2[_digitized[i]])
    return output_content, output_error


# mmh, this is acutally not very fast
@jit(nopython=True, cache=True, parallel=True, nogil=True)
def find_content_and_error_2d(x, y, edge_x, edge_y, bin_content, bin_sumW2):
    _digitized_x, _digitized_y = np.digitize(x, edge_x), np.digitize(y, edge_y)
    # index_list = list(zip(_digitized_x, _digitized_y))
    # index_list = tuple(np.transpose(index_list))
    # return bin_content[index_list], np.sqrt(bin_sumW2[index_list])
    n = len(x)
    output_content, output_error = np.empty(n, np.double), np.empty(n, np.double)
    for i in prange(n):
        output_content[i] = bin_content[_digitized_x[i]][_digitized_y[i]]
        output_error[i] = np.sqrt(bin_sumW2[_digitized_x[i]][_digitized_y[i]])
    return output_content, output_error


@jit(nopython=True, cache=True, nogil=True)
def digitize_1d(data, bins, weight=None, w2=None):
    # bins, data = np.asarray(ibins, np.double), np.asarray(idata, np.double)
    m_w = weight if weight is not None else np.ones(len(data), np.double)
    m_w2 = m_w**2 if w2 is None else w2
    assert m_w.shape == m_w2.shape
    digitized = np.digitize(data, bins)
    max_bin = len(bins) + 1
    n_digi = len(digitized)
    content, sumW2 = np.zeros(max_bin, np.double), np.zeros(max_bin, np.double)
    # WARNING: don't thread the loop because of race condiction, don't use prange
    for j in range(n_digi):
        i = digitized[j]
        content[i] += m_w[j]
        sumW2[i] += m_w2[j]
    return content, sumW2


@jit(nopython=True, parallel=True, cache=True, nogil=True)
def alt_digitize_1d(idata, ibins, weight=None, w2=None):
    bins, data = np.asarray(ibins, np.double), np.asarray(idata, np.double)
    m_w = weight if weight is not None else np.ones(len(data), np.double)
    m_w2 = m_w**2 if w2 is None else w2
    assert m_w.shape == m_w2.shape
    digitized = np.digitize(data, bins)
    max_bin = len(bins) + 1
    # n_digi = len(digitized)
    content, sumW2 = np.empty(max_bin, np.double), np.empty(max_bin, np.double)
    for i in prange(max_bin):
        # s_w = s_w2 = 0
        # for j in prange(n_digi):
        #     if i != digitized[j]:
        #         continue
        #     s_w += m_w[j]
        #     s_w2 += m_w2[j]
        # content[i] = s_w
        # sumW2[i] = s_w2
        i_mask = digitized == i
        content[i] = m_w[i_mask].sum()
        sumW2[i] = m_w2[i_mask].sum()
    return content, sumW2


@jit(nopython=True, cache=True, nogil=True, parallel=True)
def alt_chuck_digitize_1d(data, bins, weight=None, w2=None, nchunks=8):
    m_w = weight if weight is not None else np.ones(len(data), np.double)
    m_w2 = m_w**2 if w2 is None else w2
    assert m_w.shape == m_w2.shape
    max_bin = len(bins) + 1
    n_digi = len(data)
    content, sumW2 = np.zeros((nchunks, max_bin), np.double), np.zeros(
        (nchunks, max_bin), np.double
    )
    for n in prange(nchunks):
        start = n_digi * n // nchunks
        stop = n_digi * (n + 1) // nchunks
        digitized = np.digitize(data[start:stop], bins)
        tmp_w = m_w[start:stop]
        tmp_w2 = m_w2[start:stop]
        for i, j in enumerate(digitized):
            content[n, j] += tmp_w[i]
            sumW2[n, j] += tmp_w2[i]
    return content.sum(axis=0), sumW2.sum(axis=0)


@jit(nopython=True, cache=True, nogil=True, parallel=True)
def chuck_digitize_2d(xdata, ydata, xbins, ybins, weight=None, w2=None, nchunks=8):
    m_w = weight if weight is not None else np.ones(len(xdata), np.double)
    m_w2 = m_w**2 if w2 is None else w2
    assert m_w.shape == m_w2.shape
    max_xbin = len(xbins) + 1
    max_ybin = len(ybins) + 1
    n_xdigi = len(xdata)
    n_ydigi = len(ydata)
    assert n_xdigi == n_ydigi
    content, sumW2 = np.zeros((nchunks, max_xbin, max_ybin), np.double), np.zeros(
        (nchunks, max_xbin, max_ybin), np.double
    )
    for n in prange(nchunks):
        start = n_xdigi * n // nchunks
        stop = n_xdigi * (n + 1) // nchunks
        digitized_x = np.digitize(xdata[start:stop], xbins)
        digitized_y = np.digitize(ydata[start:stop], ybins)
        tmp_w = m_w[start:stop]
        tmp_w2 = m_w2[start:stop]
        for j in range(len(digitized_x)):
            i, k = digitized_x[j], digitized_y[j]
            content[n, i, k] += tmp_w[j]
            sumW2[n, i, k] += tmp_w2[j]
    return content.sum(axis=0), sumW2.sum(axis=0)


@jit(nopython=True, cache=True, nogil=True)
def is_none_zero(data):
    """
    short ciruit version for checking if the array is all zeros
    """
    for x in data:
        if x:
            return True
    return False


@jit(nopython=True, cache=True, parallel=True, nogil=True)
def parallel_nonzero_count(data):
    """
    parallel should be safe here
    """
    flattened = data.ravel()
    sum_ = 0
    for i in prange(flattened.size):
        sum_ += flattened[i] != 0
    return sum_
