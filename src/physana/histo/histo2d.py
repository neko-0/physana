import logging

import numpy as np
import scipy

from . import jitfunc
from .histo_base import HistogramBase, _hadd, _hsub, _hdiv, _hmul
from .histo1d import Histogram
from ..backends.root import from_th2, to_th2

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class Histogram2D(HistogramBase):
    """
    class for 2D histograms
    """

    __slots__ = (
        "xvar",
        "yvar",
        "xbin",
        "xmin",
        "xmax",
        "ybin",
        "ymin",
        "ymax",
        "xtitle",
        "ytitle",
        "_observable",
        "_bins",
        "_data_array",
    )

    def __init__(
        self,
        name,
        xvar,
        yvar,
        xbin,
        xmin,
        xmax,
        ybin,
        ymin,
        ymax,
        xtitle=None,
        ytitle=None,
        dtype=None,
        filter_paths=None,
    ):
        super().__init__(name, dtype, filter_paths=filter_paths)
        self.xvar = xvar
        self.yvar = yvar
        self.xbin = xbin
        self.xmin = xmin
        self.xmax = xmax
        self.ybin = ybin
        self.ymin = ymin
        self.ymax = ymax
        self.xtitle = xtitle
        self.ytitle = ytitle
        self._observable = [xvar, yvar]

        self._bins = None
        self._bin_content = np.zeros((len(self.bins[0]) + 1, len(self.bins[1]) + 1))
        self._sumW2 = np.zeros((len(self.bins[0]) + 1, len(self.bins[1]) + 1))

        # use for holding the data in from_array
        self._data_array = None

    @property
    def bin_content(self):
        return self._bin_content

    @bin_content.setter
    def bin_content(self, rhs):
        if self._bin_content.shape != rhs.shape:
            raise ValueError("Shape does not match!")
        else:
            self._bin_content = rhs.astype(self._bin_content.dtype)

    @property
    def sumW2(self):
        return self._sumW2

    @sumW2.setter
    def sumW2(self, rhs):
        if self._sumW2.shape != rhs.shape:
            raise ValueError("Shape does not match!")
        else:
            self._sumW2 = rhs.astype(self._sumW2.dtype)

    @property
    def observable(self):
        return self._observable

    @observable.setter
    def observable(self, obs):
        self._observable = obs

    @classmethod
    def variable_bin(cls, name, xvar, yvar, xbins, ybins, *args, **kwargs):
        cls_obj = cls(
            name,
            xvar,
            yvar,
            len(xbins) - 1,
            xbins[0],
            xbins[-1],
            len(ybins) - 1,
            ybins[0],
            ybins[-1],
            *args,
            **kwargs,
        )
        # p_inf = np.array([np.inf])
        # n_inf = np.array([-np.inf])
        # xb = np.concatenate([n_inf, np.array(xbins), p_inf])
        # yb = np.concatenate([n_inf, np.array(ybins), p_inf])
        xb = np.array(xbins, dtype=np.single)
        yb = np.array(ybins, dtype=np.single)
        cls_obj.bins = [xb, yb]
        return cls_obj

    @property
    def bins(self):
        if self._bins is None:
            xwidth = (self.xmax - self.xmin) / self.xbin
            ywidth = (self.ymax - self.ymin) / self.ybin
            x = np.arange(self.xmin, self.xmax + xwidth, xwidth)
            y = np.arange(self.ymin, self.ymax + ywidth, ywidth)
            # p_inf = np.array([np.inf])
            # n_inf = np.array([-np.inf])
            # x = np.concatenate([n_inf, x, p_inf])
            # y = np.concatenate([n_inf, y, p_inf])
            self._bins = (x, y)
        return self._bins

    @bins.setter
    def bins(self, rhs):
        m_bins = self.bins
        old_shape = (len(m_bins[0]), len(m_bins[1]))
        rhs_shape = (len(rhs[0]), len(rhs[1]))
        if old_shape == rhs_shape:
            if all(m_bins[0] == rhs[0]) and all(m_bins[1] == rhs[1]):
                return
        self._bins = [np.asarray(rhs[0]), np.asarray(rhs[1])]
        content_shape = (len(self._bins[0]) + 1, len(self._bins[1]) + 1)
        self._bin_content = np.zeros(content_shape)
        self._sumW2 = np.zeros(content_shape)

    @property
    def bin_width(self):
        xbins, ybins = self.bins
        return np.diff(xbins), np.diff(ybins)

    @property
    def hist_type(self):
        return "2d"

    @property
    def ndim(self):
        return 2

    @property
    def underflow(self):
        return self._bin_content[0, :], self._bin_content[:, 0]

    @property
    def overflow(self):
        return self._bin_content[-1, :], self._bin_content[:, -1]

    def __getitem__(self, index):
        """
        only operate on the bin content
        """
        return self._bin_content[index]

    def __add__(self, rhs):
        c_self = self.copy()
        c_self.add(rhs)
        return c_self

    def __sub__(self, rhs):
        c_self = self.copy()
        c_self.sub(rhs)
        return c_self

    def __rsub__(self, lhs):
        c_self = self.copy()
        c_self.sub(lhs)
        c_self.mul(-1.0)
        return c_self

    def __truediv__(self, rhs):
        c_self = self.copy()
        c_self.div(rhs)
        return c_self

    def __rtruediv__(self, lhs):
        if isinstance(lhs, (int, float, np.floating)):
            c_self = self.copy()
            c_self._sumW2 /= 1.0 / self._bin_content**2
            c_self._bin_content = lhs / self._bin_content
            c_self._sumW2 *= c_self._bin_content**2
            return c_self
        else:
            raise TypeError(f"__rtruediv__ it not implemented for type {type(lhs)}")

    def __mul__(self, rhs):
        c_self = self.copy()
        c_self.mul(rhs)
        return c_self

    def __rmul__(self, lhs):
        return self.__mul__(lhs)

    def __abs__(self):
        c_self = self.copy()
        np.abs(c_self._bin_content, out=c_self._bin_content)
        return c_self

    def add(self, rhs):
        if isinstance(rhs, type(self)):
            _hadd(self._bin_content, self._sumW2, rhs._bin_content, rhs._sumW2)
            if rhs._data_array:
                if self._data_array is None:
                    self._data_array = []
                self._data_array += rhs._data_array
        elif isinstance(rhs, (int, float, np.floating)):
            self._bin_content += rhs
        else:
            raise TypeError(f"Invalid rhs type {type(rhs)}")

    def sub(self, rhs):
        if isinstance(rhs, type(self)):
            _hsub(self._bin_content, self._sumW2, rhs._bin_content, rhs._sumW2)
        elif isinstance(rhs, (int, float, np.floating)):
            self._bin_content -= rhs
        else:
            raise TypeError(f"Invalid rhs type {type(rhs)}")

    def div(self, rhs):
        if isinstance(rhs, type(self)):
            _hdiv(self._bin_content, self._sumW2, rhs._bin_content, rhs._sumW2)
        elif isinstance(rhs, (int, float, np.floating)):
            self.mul(1.0 / rhs)
        else:
            raise TypeError(f"Invalid type {type(rhs)}")

    def mul(self, rhs):
        if isinstance(rhs, type(self)):
            _hmul(self._bin_content, self._sumW2, rhs._bin_content, rhs._sumW2)
        elif isinstance(rhs, (int, float, np.floating)):
            self._sumW2 = self._sumW2 * rhs**2
            self._bin_content = self._bin_content * rhs
        else:
            raise TypeError(f"Invalid type {type(rhs)}")

    def basic_info(self):
        _basic_info = {
            "name",
            "xvar",
            "yvar",
            "xbin",
            "ybin",
            "xmin",
            "xmax",
            "ymin",
            "ymax",
            "xtitle",
            "ytitle",
        }
        return {name: getattr(self, name) for name in _basic_info}

    def integral(self, opt=""):
        return self._bin_content.sum()

    def project(self, axis):
        if axis == 0:
            bins = self.bins[1]
            name = f"{self.name}_proj_x"
            title = self.xtitle
        else:
            bins = self.bins[0]
            name = f"{self.name}_proj_y"
            title = self.ytitle
        proj = Histogram.variable_bin(name, bins, title)
        proj._bin_content = np.sum(self._bin_content, axis=axis)
        proj._sumW2 = np.sum(self._sumW2, axis=axis)
        return proj

    def project_x(self):
        return self.project(0)

    def project_y(self):
        return self.project(1)

    def profile(self, axis, weight=False):
        if axis == 0:
            bins = self.bins[1]
            name = f"{self.name}_prof_x"
            title = self.xtitle
        else:
            bins = self.bins[0]
            name = f"{self.name}_prof_y"
            title = self.ytitle
        prof = Histogram.variable_bin(name, bins, title)
        if weight:
            w = np.nan_to_num(1 / self._sumW2, posinf=0, neginf=0)
            prof._bin_content = np.sum(self._bin_content * w, axis=axis)
            prof._bin_content /= np.sum(w, axis=axis)
            np.nan_to_num(prof._bin_content, copy=False)
            prof._sumW2 = np.sum(w, axis=axis) / np.sum(w, axis=axis) ** 2
        else:
            prof._bin_content = np.mean(self._bin_content, axis=axis)
            prof._sumW2 = np.mean(self._sumW2, axis=axis)
        return prof

    def profile_x(self, weight=False):
        return self.profile(0, weight)

    def profile_y(self, weight=False):
        return self.profile(1, weight)

    def remove_negative_bin(self):
        self._sumW2[self._bin_content < 0] = 0.0
        self._bin_content[self._bin_content < 0] = 0.0

    def nan_to_num(self, nan=0.0, posinf=0, neginf=0):
        np.nan_to_num(self._bin_content, False, nan, posinf, neginf)
        np.nan_to_num(self._sumW2, False, nan, posinf, neginf)

    def get_bin_content(self, index):
        return self._bin_content[index]

    def get_bin_error(self, index):
        return np.sqrt(self._sumW2[index])

    def get_bin_content_list(self, index):
        return self.get_bin_content(tuple(np.transpose(index)))

    def get_bin_error_list(self, index):
        return self.get_bin_error(tuple(np.transpose(index)))

    def find_content_and_error(self, x, y):
        return jitfunc.find_content_and_error_2d(
            x, y, self.bins[0], self.bins[1], self._bin_content, self._sumW2
        )

    @staticmethod
    def _digitize(xdata, ydata, bins, weight=None, w2=None):
        """
        default implementation
        """
        m_w = weight if weight is not None else np.ones(len(xdata))
        # matching bins. inf is require for underflow/overflow bins
        # because values outside of ranges will not be included
        pos_inf = np.array([np.inf])
        neg_inf = -1 * pos_inf
        binx = np.concatenate((neg_inf, bins[0], pos_inf))
        biny = np.concatenate((neg_inf, bins[1], pos_inf))
        bins = [binx, biny]
        digitized2D_sum = scipy.stats.binned_statistic_2d(
            xdata,
            ydata,
            [m_w, m_w**2 if w2 is None else w2],
            "sum",
            bins=bins,
            expand_binnumbers=True,
        )

        '''
        for xb in range(0, len(bins[0]) + 1):
            for yb in range(0, len(bins[1]) + 1):
                """
                sel = (digitized2D.binnumber[0] == xb) & (
                    digitized2D.binnumber[1] == yb
                )
                """
                sel_x = digitized2D.binnumber[0] == xb
                sel_y = digitized2D.binnumber[1] == yb
                content_buffer[xb][yb] += np.sum(m_w[sel_x & sel_y])
                sumW2_buffer[xb][yb] += np.sum(m_w[sel_x & sel_y] ** 2)
        '''

        return digitized2D_sum.statistic

    @staticmethod
    def digitize(xdata, ydata, bins, weight=None, w2=None):
        """
        parallel with chuncks
        """
        m_w = weight if weight is not None else np.ones(len(xdata))
        return jitfunc.chuck_digitize_2d(
            np.asarray(xdata, np.double),
            np.asarray(ydata, np.double),
            np.asarray(bins[0], np.double),
            np.asarray(bins[1], np.double),
            m_w,
            w2,
        )

    def from_array(self, xdata, ydata, w=None, w2=None, accumulate=True):
        xdata = np.asarray(xdata)
        ydata = np.asarray(ydata)
        if w is not None:
            w = np.asarray(w)
        if w2 is not None:
            w2 = np.asarray(w2)
        if xdata.dtype == np.bool_:
            xdata = xdata.astype(int)
        if ydata.dtype == np.bool_:
            ydata = ydata.astype(int)
        content, sumW2 = self.digitize(xdata, ydata, self.bins, w, w2)
        if accumulate:
            self._bin_content += content
            self._sumW2 += sumW2
        else:
            self._bin_content = content
            self._sumW2 = sumW2
        if self._store_data:
            if self._data_array is None:
                self._data_array = []
            self._data_array.append((xdata, ydata, w))
        '''
        # temperoraly not using this, just disable
        # reduce disk space usage.
        if self.statistic is not None:
            self.statistic += dig2D.statistic
        else:
            self.statistic = dig2D.statistic.copy()
        '''

    def diagonal(self):
        """
        Returning the diagonal content.
        """
        return np.diagonal(self.bin_content)

    def purity(self, axis=0):
        """
        Return a purity 1D histogram along a axis.
        the purity along a given axis is defined as N(i==j)/sum(N(j)),
        where i and j is the index of the bin content matrix/array
        """
        # note the difference in axis for project and sum
        if axis == 0:
            var, title = self.yvar, self.ytitle
        else:
            var, title = self.xvar, self.xtitle
        bin = self.bins[axis]
        diag = np.diag(self.bin_content)
        sum = np.sum(self.bin_content, axis=axis)
        purity = diag / sum
        diag_error = np.sqrt(diag) / diag
        sum_error = np.sqrt(sum) / sum
        purity_error = (diag_error**2 + sum_error**2) * purity**2
        purity_histo = Histogram.variable_bin(
            var, bin, title, ytitle="Purity (N_i==j / sum(N_i))", type="purity"
        )
        purity_histo.bin_content = purity
        purity_histo.sumW2 = purity_error
        purity_histo.nan_to_num()
        return purity_histo

    @property
    def root(self):
        return to_th2(self)

    @root.setter
    def root(self, roothist):
        from_th2(self, roothist)

    def clear_content(self):
        self._bin_content.fill(0.0)
        self._sumW2.fill(0.0)

    def extend_bins(self, type="underflow", shallow=False, inplace=False):
        """
        Extending bins to under/overflow bins.
        """
        xbins, ybins = self.bins
        xbwidth, ybwidth = self.bin_width
        bin_content = self.bin_content
        sumW2 = self.sumW2
        c_hist = self if inplace else self.copy(shallow=shallow)
        if type == "underflow_x":
            new_bins = ([xbins[0] - xbwidth[0]], xbins)
            new_content = np.zeros((1, bin_content.shape[1]))
            new_sumW2 = np.zeros((1, sumW2.shape[1]))
            c_hist.bins = [np.concatenate(new_bins), ybins]
            c_hist._bin_content = np.concatenate((new_content, bin_content))
            c_hist._sumW2 = np.concatenate((new_sumW2, sumW2))
        elif type == "underflow_y":
            new_bins = ([ybins[0] - ybwidth[0]], ybins)
            new_content = np.zeros((bin_content.shape[0], 1))
            new_sumW2 = np.zeros((sumW2.shape[0], 1))
            c_hist.bins = [xbins, np.concatenate(new_bins)]
            c_hist._bin_content = np.concatenate((new_content, bin_content), axis=1)
            c_hist._sumW2 = np.concatenate((new_sumW2, sumW2), axis=1)
        elif type == "underflow":
            xbins = np.concatenate(([xbins[0] - xbwidth[0]], xbins))
            ybins = np.concatenate(([ybins[0] - ybwidth[0]], ybins))
            c_hist.bins = [xbins, ybins]
            shape = bin_content.shape
            new_content = np.zeros((shape[0] + 1, shape[1] + 1))
            new_sumW2 = np.zeros((shape[0] + 1, shape[1] + 1))
            new_content[1:, 1:] = bin_content if shallow else bin_content.copy()
            new_sumW2[1:, 1:] = sumW2 if shallow else sumW2.copy()
            c_hist._bin_content = new_content
            c_hist._sumW2 = new_sumW2
        elif type == "overflow_x":
            new_bins = (xbins, [xbins[-1] + xbwidth[-1]])
            new_content = np.zeros((1, bin_content.shape[1]))
            c_hist.bins = [np.concatenate(new_bins), ybins]
            c_hist.xbin = np.concatenate(new_bins)
            c_hist._bin_content = np.concatenate((bin_content, new_content))
            c_hist._sumW2 = np.concatenate((sumW2, new_sumW2))
        elif type == "overflow_y":
            new_bins = (ybins, [ybins[-1] + ybwidth[-1]])
            new_content = np.zeros((bin_content.shape[0], 1))
            new_sumW2 = np.zeros((sumW2.shape[0], 1))
            c_hist.bins = [xbins, np.concatenate(new_bins)]
            c_hist._bin_content = np.concatenate((bin_content, new_content), axis=1)
            c_hist._sumW2 = np.concatenate((sumW2, new_sumW2), axis=1)
        elif type == "overflow":
            ybins = np.concatenate((ybins, [ybins[-1] + ybwidth[-1]]))
            xbins = np.concatenate((xbins, [xbins[-1] + xbwidth[-1]]))
            c_hist.bins = [xbins, ybins]
            shape = bin_content.shape
            new_content = np.zeros((shape[0] + 1, shape[1] + 1))
            new_sumW2 = np.zeros((shape[0] + 1, shape[1] + 1))
            new_content[:-1, :-1] = bin_content if shallow else bin_content.copy()
            new_sumW2[:-1, :-1] = sumW2 if shallow else sumW2.copy()
            c_hist._bin_content = new_content
            c_hist._sumW2 = new_sumW2
        return c_hist

    def collapse_to_underflow(self, axis=0, shallow=False, inplace=False):
        c_self = self.copy(shallow=shallow) if not inplace else self
        if axis == 0:
            for i in range(c_self.bin_content.shape[0]):
                c_self.bin_content[i, 0] = c_self.bin_content[i, :].sum()
                c_self.bin_content[i, 1:].fill(0)
                c_self.sumW2[i, 0] = c_self.sumW2[i, :].sum()
                c_self.sumW2[i, 1:].fill(0)
        elif axis == 1:
            for i in range(c_self.bin_content.shape[1]):
                c_self.bin_content[0, i] = c_self.bin_content[:, i].sum()
                c_self.bin_content[1:, i].fill(0)
                c_self.sumW2[0, i] = c_self.sumW2[:, i].sum()
                c_self.sumW2[1:, i].fill(0)
        return c_self
