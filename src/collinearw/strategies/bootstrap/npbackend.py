import copy
import numpy as np
from hashlib import md5
from numba import jit, prange, objmode
from ...core import Histogram, Histogram2D
from ... import jitfunc


def replica_weights(num_evnt, nreplica=10, tolist=True, dtype=np.intc, seed=42):
    rng = np.random.default_rng(seed)
    output = rng.poisson(1, (num_evnt, nreplica)).astype(dtype)
    return output.tolist() if tolist else output


@jit(nopython=True, cache=True)
def replica_weights_nb(num_evnt, nreplica=1000, seed=42):
    with objmode(output="int64[:,:]"):
        rng = np.random.default_rng(seed)
        output = rng.poisson(1, (num_evnt, nreplica))
    return output


# @jit(nopython=True, cache=True, parallel=True, nogil=True)
@jit(nopython=True, cache=False, parallel=True, nogil=True)
def fill_replica_1d_fast(bins, data, weight, nreplica, replica, sumw2, replica_w):
    """
    weight = [w1, w2, w3, ...], with size N
    replica_w = [r1, r1, r1, ...], with size N
    """
    nevent = len(data)
    assert replica.shape == sumw2.shape
    assert data.shape == weight.shape
    assert replica_w.shape == (nevent, nreplica)
    # TODO: actually, we may get the digitized bin from 'from_array'
    # so no need to redo the digitize again.
    digi = np.digitize(data, bins)
    # n_digi = len(digi)
    # max_bin = np.max(digi) + 1
    # for i in prange(nreplica):
    #     for j in prange(max_bin):
    #         for k in prange(n_digi):
    #             if j != digi[k]:
    #                 continue
    #             replica[i, j] += replica_w[i, k] * weight[k]
    #             sumw2[i, j] += replica_w[i, k] * (weight[k] ** 2)
    for i in prange(nreplica):
        for j in range(nevent):
            replica[i, digi[j]] += replica_w[j, i] * weight[j]
            sumw2[i, digi[j]] += replica_w[j, i] * (weight[j] ** 2)


@jit(nopython=True, cache=True, parallel=True, nogil=True)
def fill_replica_2d_fast(
    xbins, ybins, xdata, ydata, weight, nreplica, replica, sumw2, replica_w
):
    n_digi = len(xdata)
    assert xdata.shape == ydata.shape
    assert replica_w.shape == (n_digi, nreplica)
    x_digi = np.digitize(xdata, xbins)
    y_digi = np.digitize(ydata, ybins)
    for i in prange(nreplica):
        for j in range(n_digi):
            replica[i, x_digi[j], y_digi[j]] += replica_w[j, i] * weight[j]
            sumw2[i, x_digi[j], y_digi[j]] += replica_w[j, i] * (weight[j] ** 2)


@jit(nopython=True, cache=True, parallel=True, nogil=True)
def replica_1d_covariance_matrix(replica):
    nreplica, nbin = replica.shape
    cov = np.empty((nbin, nbin))
    # mean = np.mean(replica, axis=0) # numba does not support 'axis' kwargs
    mean = np.empty(nbin)
    for i in prange(nbin):
        mean_s = 0
        for toy in prange(nreplica):
            mean_s += replica[toy][i]
        mean[i] = mean_s / nreplica
    for i in prange(nbin):
        for j in range(i, nbin):
            s = 0
            for toy in prange(nreplica):
                s += (replica[toy][i] - mean[i]) * (replica[toy][j] - mean[j])
            cov[i, j] = cov[j, i] = s / nreplica
    return cov


@jit(nopython=True, cache=True, parallel=True, nogil=True)
def replica_1d_correlation_matrix(replica):
    nreplica, nbin = replica.shape
    cov = replica_1d_covariance_matrix(replica)
    diag = np.diag(cov)
    corr = np.empty((nbin, nbin))
    for i in prange(nbin):
        for j in prange(i + 1):
            norm = np.sqrt(diag[i] * diag[j])
            if norm == 0:
                corr[i, j] = corr[j, i] = 0
            else:
                corr[i, j] = corr[j, i] = cov[i, j] / norm
    return corr


# ==============================================================================
# ==============================================================================
class _ReplicaInterface:
    # __slots__ = ('nreplica', 'replica', 'replica_sumw2', 'replica_w')

    def __init__(self, nreplica):
        self.nreplica = nreplica
        self.replica = None
        self.replica_sumw2 = None
        self.replica_w = None

    @property
    def bootstrap(self):
        """
        mmmh, the ROOT version use bootstrap, so this is just to make it
        compatible with the BootstrapHistMaker
        """
        return self.replica, self.replica_sumw2

    @bootstrap.setter
    def bootstrap(self, rhs):
        """
        expecting rhs = (rhs.replica, rhs.replica_sumw2)
        """
        self.replica, self.replica_sumw2 = rhs

    def unique_name(self):
        return md5(f"{self.full_name}".encode("utf-8")).hexdigest()

    def replica_init(self, shape):
        self.replica = np.zeros(shape)
        self.replica_sumw2 = np.zeros(shape)

    def copy(self, shallow=False):
        return copy.copy(self) if shallow else copy.deepcopy(self)

    def add(self, rhs):
        super().add(rhs)
        jitfunc.hadd(self.replica, self.replica_sumw2, rhs.replica, rhs.replica_sumw2)

    def sub(self, rhs):
        super().sub(rhs)
        jitfunc.hsub(self.replica, self.replica_sumw2, rhs.replica, rhs.replica_sumw2)

    def mul(self, rhs):
        super().mul(rhs)
        jitfunc.hmul(self.replica, self.replica_sumw2, rhs.replica, rhs.replica_sumw2)

    def div(self, rhs):
        super().div(rhs)
        jitfunc.hdiv(self.replica, self.replica_sumw2, rhs.replica, rhs.replica_sumw2)

    def get_replica(self, i=None):
        if i is None:
            return self.bin_content
        return self.replica[i, :]

    def get_replica_sumw2(self, i=None):
        if i is None:
            return self.sumW2
        return self.replica_sumw2[i, :]

    def get_replica_root(self, i):
        """
        return ROOT histogram version of the i-th replica
        """
        c_h = self.to_histogram(copy=False)
        c_h._bin_content = self.get_replica(i)
        c_h._sumW2 = self.get_replica_sumw2(i)
        c_h.name += f"_replica_{i}"
        return c_h.root

    def get_replica_hist(self, i, copy=True):
        c_h = self.to_histogram(copy=False)
        if copy:
            c_h._bin_content = self.get_replica(i).copy()
            c_h._sumW2 = self.get_replica_sumw2(i).copy()
        else:
            c_h._bin_content = self.get_replica(i)
            c_h._sumW2 = self.get_replica_sumw2(i)
        return c_h

    def merge_replica(self, rhs):
        self.replica = np.vstack((self.replica, rhs.replica))
        self.replica_sumw2 = np.vstack((self.replica_sumw2, rhs.replica_sumw2))
        self.nreplica += rhs.nreplica


# ==============================================================================
# ==============================================================================


class HistogramBootstrap(_ReplicaInterface, Histogram):
    def __init__(self, *args, nreplica=100, **kwargs):
        _ReplicaInterface.__init__(self, nreplica)
        Histogram.__init__(self, *args, **kwargs)
        self.replica_init()

    def __copy__(self):
        c_self = super().__copy__()
        c_self.replica = copy.copy(self.replica)
        c_self.replica_sumw2 = copy.copy(self.replica_sumw2)
        c_self.nreplica = self.nreplica
        c_self.replica_w = self.replica_w
        return c_self

    def __deepcopy__(self, memo):
        c_self = super().__deepcopy__(memo)
        c_self.replica = copy.deepcopy(self.replica, memo)
        c_self.replica_sumw2 = copy.deepcopy(self.replica_sumw2, memo)
        c_self.nreplica = self.nreplica
        c_self.replica_w = self.replica_w
        return c_self

    @classmethod
    def from_histogram(cls, hist, nreplica=100, **kwargs):
        """
        Create bootstrap histogram from regular histogram
        """
        args = hist.name, hist.nbin, hist.xmin, hist.xmax
        new_hist = cls(*args, nreplica=nreplica, **kwargs)
        new_hist._bins = hist._bins
        new_hist._bin_content = hist._bin_content
        new_hist._sumW2 = hist._sumW2
        new_hist.replica_init()
        return new_hist

    def to_histogram(self, copy=True):
        """
        reduce to a more simplified Histogram type
        """
        kwargs = self.basic_info()
        new_hist = Histogram(**kwargs)
        if copy:
            new_hist.bins = copy.copy(self.bins)
            new_hist.bin_content = copy.copy(self.bin_content)
        else:
            new_hist.bins = self._bins
            new_hist.bin_content = self.bin_content
        return new_hist

    def replica_init(self):
        super().replica_init((self.nreplica, len(self.bin_content)))

    def from_array(self, data, w=None, w2=None):
        super().from_array(data, w, w2)
        if self.replica_w is not None:
            w = w if w is not None else np.ones(len(data), dtype=np.float64)
            fill_replica_1d_fast(
                self.bins,
                data.astype(np.float64),
                w.astype(np.float64),
                self.nreplica,
                self.replica,
                self.replica_sumw2,
                self.replica_w.astype(np.int64),
            )
            self.replica_w = None

    def collapse_to_underflow(self, shallow=False, inplace=False):
        c_self = super().collapse_to_underflow(shallow, inplace)
        # collapse replica
        c_self.replica[:, 0] = np.sum(c_self.replica, axis=1)
        c_self.replica[:, 1:].fill(0)
        c_self.replica_sumw2[:, 0] = np.sum(c_self.replica_sumw2, axis=1)
        c_self.replica_sumw2[:, 1:].fill(0)
        return c_self

    def rescale(self, scale):
        self.bin_content *= scale
        self.replica *= scale
        np.nan_to_num(self._bin_content, False, 0, 0, 0)
        np.nan_to_num(self.replica, False, 0, 0, 0)

    def normalise_replica(self):
        row_sum = np.sum(self.replica, axis=1)
        self.replica /= row_sum[:, np.newaxis]

    def replica_mean(self):
        return np.mean(self.replica, axis=0)

    def replica_var(self):
        return np.var(self.replica, axis=0)

    def covariance_matrix(self):
        return replica_1d_covariance_matrix(self.replica)

    def correlation_matrix(self):
        return replica_1d_correlation_matrix(self.replica)

    def scale_factor_covariance(self):
        return replica_1d_covariance_matrix(self.replica / self.bin_content)

    def merge_overflow(self):
        self._bin_content[[1, -2]] += self._bin_content[[0, -1]]
        self._sumW2[[1, -2]] += self._sumW2[[0, -1]]
        self._bin_content[[0, -1]] += 0.0
        self._sumW2[[0, -1]] += 0.0
        self.replica[:, [1, -2]] += self.replica[:, [0, -1]]
        self.replica_sumw2[:, [1, -2]] += self.replica_sumw2[:, [0, -1]]
        self.replica[:, [0, -1]] = 0.0
        self.replica_sumw2[:, [0, -1]] = 0.0


# ==============================================================================
# ==============================================================================


class Histogram2DBootstrap(_ReplicaInterface, Histogram2D):
    def __init__(self, *args, nreplica=100, **kwargs):
        _ReplicaInterface.__init__(self, nreplica)
        Histogram2D.__init__(self, *args, **kwargs)
        self.replica_init()

    def __copy__(self):
        c_self = super().__copy__()
        c_self.replica = copy.copy(self.replica)
        c_self.replica_sumw2 = copy.copy(self.replica_sumw2)
        c_self.nreplica = self.nreplica
        c_self.replica_w = self.replica_w
        return c_self

    def __deepcopy__(self, memo):
        c_self = super().__deepcopy__(memo)
        c_self.replica = copy.deepcopy(self.replica, memo)
        c_self.replica_sumw2 = copy.deepcopy(self.replica_sumw2, memo)
        c_self.nreplica = self.nreplica
        c_self.replica_w = self.replica_w
        return c_self

    @classmethod
    def from_histogram(cls, hist, nreplica=1000, **kwargs):
        """
        Create bootstrap histogram from regular histogram
        """
        bins = hist.bins
        name_args = hist.name, hist.xvar, hist.yvar
        xbins_args = len(bins[0]) - 1, bins[0][0], bins[0][-1]
        ybins_args = len(bins[1]) - 1, bins[1][0], bins[1][-1]
        args = (*name_args, *xbins_args, *ybins_args)
        new_hist = cls(*args, nreplica=nreplica, **kwargs)
        new_hist._bins = hist._bins
        new_hist._bin_content = hist._bin_content
        new_hist._sumW2 = hist._sumW2
        new_hist.replica_init()
        return new_hist

    def to_histogram(self, copy=True):
        """
        reduce to a more simplified Histogram type
        """
        kwargs = self.basic_info()
        new_hist = Histogram2D(**kwargs)
        if copy:
            new_hist.bins = copy.copy(self.bins)
            new_hist.bin_content = copy.copy(self.bin_content)
        else:
            new_hist.bins = self.bins
            new_hist.bin_content = self.bin_content
        return new_hist

    def replica_init(self):
        super().replica_init((self.nreplica, *self.bin_content.shape))

    def from_array(self, xdata, ydata, w=None, w2=None):
        super().from_array(xdata, ydata, w, w2)
        if self.replica_w is not None:
            w = w if w is not None else np.ones(len(xdata), dtype=np.float64)
            fill_replica_2d_fast(
                self.bins[0],
                self.bins[1],
                xdata.astype(np.float64),
                ydata.astype(np.float64),
                w.astype(np.float64),
                self.nreplica,
                self.replica,
                self.replica_sumw2,
                self.replica_w.astype(np.int64),
            )
            self.replica_w = None

    def collapse_to_underflow(self, axis=0, shallow=False, inplace=False):
        c_self = super().collapse_to_underflow(axis, shallow, inplace)
        if axis == 0:
            c_self.replica[:, :, 0] = np.sum(c_self.replica[:], axis=axis + 1)
            c_self.replica[:, :, 1:].fill(0)
            c_self.replica_sumw2[:, :, 0] = np.sum(
                c_self.replica_sumw2[:], axis=axis + 1
            )
            c_self.replica_sumw2[:, :, 1:].fill(0)
        elif axis == 1:
            c_self.replica[:, 0, :] = np.sum(c_self.replica[:], axis=axis + 1)
            c_self.replica[:, 1:, :].fill(0)
            c_self.replica_sumw2[:, 0, :] = np.sum(
                c_self.replica_sumw2[:], axis=axis + 1
            )
            c_self.replica_sumw2[:, 1:, :].fill(0)
        return c_self
