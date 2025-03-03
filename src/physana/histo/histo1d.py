import logging

import numpy as np

from . import jitfunc
from .histo_base import HistogramBase, _hadd, _hsub, _hdiv, _hmul
from ..backends.root import from_th1, to_th1, to_root_graph
from ..systematics.syst_band import SystematicsBand

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class Histogram(HistogramBase):
    __slots__ = (
        "nbin",
        "xmin",
        "xmax",
        "xtitle",
        "ytitle",
        "systematics",
        "_bins",
        "_observable",
        "_systematics_band",
        "_data_array",
    )

    def __init__(
        self,
        name,
        nbin,
        xmin,
        xmax,
        xtitle=None,
        ytitle="Number of events",
        observable=None,
        dtype=None,
        filter_paths=None,
    ):
        super().__init__(name, dtype, filter_paths=filter_paths)
        self.nbin = nbin
        self.xmin = xmin
        self.xmax = xmax
        self.xtitle = xtitle or name
        self.ytitle = ytitle
        self.systematics = None
        self._bins = None

        # observable is the lookup name in the TTree.
        # if observable is not specify, the name will be used as default.
        self._observable = observable

        self._bin_content = np.zeros(len(self.bins) + 1, dtype=np.single)
        self._sumW2 = np.zeros(len(self.bins) + 1, dtype=np.single)

        # storage for systematics bands in nominal histogram
        self._systematics_band = {} if self.systematics else None

        # use for holding the data in from_array
        self._data_array = None

    @classmethod
    def variable_bin(cls, name, bins, xtitle, *args, **kwargs):
        _obj = cls(name, len(bins) - 1, bins[0], bins[-1], xtitle, *args, **kwargs)
        _obj.bins = np.array(bins, dtype=np.single)
        return _obj

    @classmethod
    def array_to_hist(cls, a, w=None, name=None, bin_edge=None, *args, **kwargs):
        if name is None:
            name = str(a)
        if bin_edge is None:
            bmin = a.min()
            bmax = a.max()
            if bmin == bmax:
                bmax = bmin + 1
            bin_edge = np.linspace(bmin, bmax, 100)
        bins = (len(bin_edge) - 1, bin_edge[0], bin_edge[-1])
        _obj = cls(name, *bins, *args, **kwargs)
        _obj.from_array(a, w)
        return _obj

    @property
    def shape(self):
        return self._bin_content.shape

    @property
    def bin_content(self):
        return self._bin_content

    @bin_content.setter
    def bin_content(self, rhs):
        if self._bin_content.shape != rhs.shape:
            raise ValueError("Shape does not match!")
        self._bin_content = rhs.astype(self._bin_content.dtype)

    @property
    def sumW2(self):
        return self._sumW2

    @sumW2.setter
    def sumW2(self, rhs):
        if self._sumW2.shape != rhs.shape:
            raise ValueError("Shape does not match!")
        self._sumW2 = rhs.astype(self._sumW2.dtype)

    @property
    def observable(self):
        return (self._observable,) if self._observable else (self.name,)

    @observable.setter
    def observable(self, obs):
        self._observable = obs

    @property
    def is_nominal(self):
        return bool(self.systematics is None)

    @property
    def bins(self):
        if self._bins is None:
            xwidth = (self.xmax - self.xmin) / self.nbin
            self._bins = np.arange(self.xmin, self.xmax + xwidth, xwidth)
        return self._bins

    @bins.setter
    def bins(self, rhs):
        if not np.all(np.diff(rhs) > 0):
            raise ValueError(f"bins need to be sorted: {rhs}")
        m_bins = self.bins
        rhs = np.asarray(rhs)
        if m_bins.shape == rhs.shape:
            if all(m_bins == rhs):
                return
        self._bins, m_size = rhs, len(rhs)
        self._bin_content = np.zeros(m_size + 1, dtype=np.single)
        self._sumW2 = np.zeros(m_size + 1, dtype=np.single)

    @property
    def bin_width(self):
        return np.pad(np.diff(self.bins), (1, 1), mode='edge')

    @property
    def bin_index(self):
        return range(len(self._bin_content))

    @property
    def systematics_band(self):
        return self._systematics_band

    @property
    def hist_type(self):
        return "1d"

    @property
    def ndim(self):
        return 1

    @property
    def underflow(self):
        return self._bin_content[0]

    @property
    def overflow(self):
        return self._bin_content[-1]

    def __getitem__(self, index):
        """
        only operate on the bin content
        """
        return self._bin_content[index]

    def __setitem__(self, index, value):
        self._bin_content[index] = value

    def __add__(self, rhs):
        c_self = self.copy()
        c_self.add(rhs)
        return c_self

    def __radd__(self, rhs):
        return self.__add__(rhs)

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
        if not isinstance(lhs, (int, float, np.floating)):
            raise TypeError(f"__rtruediv__ it not implemented for type {type(lhs)}")
        c_self = self.copy()
        c_self._sumW2 /= 1.0 / self._bin_content**2
        c_self._bin_content = lhs / self._bin_content
        c_self._sumW2 *= c_self._bin_content**2
        return c_self

    def __mul__(self, rhs):
        c_self = self.copy()
        c_self.mul(rhs)
        return c_self

    def __abs__(self):
        c_self = self.copy()
        np.abs(c_self._bin_content, out=c_self._bin_content)
        return c_self

    def add(self, rhs):
        if isinstance(rhs, type(self)):
            _hadd(self._bin_content, self._sumW2, rhs._bin_content, rhs._sumW2)
            if not rhs._data_array:
                return
            if self._data_array is None:
                self._data_array = []
            self._data_array += rhs._data_array
        elif isinstance(rhs, (int, float, np.floating)):
            self._bin_content += np.single(rhs)
        else:
            raise TypeError(f"Invalid type {type(rhs)}")

    def sub(self, rhs):
        if isinstance(rhs, type(self)):
            # self._bin_content -= rhs._bin_content
            # self._sumW2 += rhs._sumW2
            _hsub(self._bin_content, self._sumW2, rhs._bin_content, rhs._sumW2)
        elif isinstance(rhs, (int, float, np.floating)):
            self._bin_content -= np.single(rhs)
        else:
            raise TypeError(f"Invalid type {type(rhs)}")

    def mul(self, rhs):
        if isinstance(rhs, type(self)):
            _hmul(self._bin_content, self._sumW2, rhs._bin_content, rhs._sumW2)
        elif isinstance(rhs, (int, float, np.floating)):
            # self._sumW2 = self._sumW2 * rhs ** 2
            self._sumW2 *= np.single(rhs) ** 2
            self._bin_content *= np.single(rhs)
        else:
            raise TypeError(f"Invalid type {type(rhs)}")

    def div(self, rhs):
        if isinstance(rhs, type(self)):
            _hdiv(self._bin_content, self._sumW2, rhs._bin_content, rhs._sumW2)
        elif isinstance(rhs, (int, float, np.floating)):
            self.mul(1.0 / rhs)
        else:
            raise TypeError(f"Invalid type {type(rhs)}")

    def clear_content(self):
        self._bin_content.fill(0.0)
        self._sumW2.fill(0.0)
        self._systematics_band = None

    def total_events(self):
        return np.sum(self._bin_content)

    def integral(self, opt=""):
        b_width = self.bin_width if "width" in opt else 1.0
        _integral = self._bin_content * b_width
        if "all" in opt:
            return _integral.sum()
        # skip underflow and overflow
        return _integral[1:-1].sum()

    def normalize(self):
        """
        normalize histogram bin contents and return the result.
        """
        c_self = self.copy()
        c_self.div(c_self.integral())
        return c_self

    def remove_negative_bin(self):
        """
        set all the negative bin content to zero
        """
        self._sumW2[self._bin_content < 0] = 0
        self._bin_content[self._bin_content < 0] = 0

    def nan_to_num(self, nan=0.0, posinf=0, neginf=0):
        np.nan_to_num(self._bin_content, False, nan, posinf, neginf)
        np.nan_to_num(self._sumW2, False, nan, posinf, neginf)

    def get_bin_content(self, index):
        return self._bin_content[index]

    def get_bin_error(self, index):
        return np.sqrt(self._sumW2[index])

    def find_content_and_error(self, x):
        return jitfunc.find_content_and_error_1d(
            x, self.bins, self._bin_content, self._sumW2
        )

    @staticmethod
    def digitize(data, bins, weight=None, w2=None):
        """
        jit default method
        """
        # return jitfunc.digitize_1d(
        #     np.asarray(data, np.double), np.asarray(bins, np.double), weight, w2
        # )
        """
        regualr approach
        """
        # content = [np.sum(m_w[digitized == i]) for i in bin_range]
        # sumW2 = [np.sum(m_w2[digitized == i]) for i in bin_range]
        # return np.asarray(content), np.asarray(sumW2)
        """
        jit parallel approach
        """
        return jitfunc.alt_chuck_digitize_1d(
            np.asarray(data, np.double), np.asarray(bins, np.double), weight, w2
        )

    def from_array(self, data, w=None, w2=None, accumulate=True):
        """
        load array of data into hitogram
        """
        bins = np.asarray(self.bins)
        data = np.asarray(data)
        if w is not None:
            w = np.asarray(w)
        if w2 is not None:
            w2 = np.asarray(w2)
        content, sumW2 = self.digitize(data, bins, w, w2)
        if accumulate:
            self._bin_content += content
            self._sumW2 += sumW2
        else:
            self._bin_content = content
            self._sumW2 = sumW2

        if self._store_data:
            if self._data_array is None:
                self._data_array = []
            self._data_array.append((data, w))

    def update_systematics_band(self, band: SystematicsBand):
        if self._systematics_band is None:
            self._systematics_band = {}
        # make sure the band has the same bin size
        assert band.shape == self.bin_content.shape
        if band.name in self._systematics_band:
            self._systematics_band[band.name].update(band)
        else:
            self._systematics_band[band.name] = band

    def scale_band(self, band_name):
        """
        Method directly invokes the scale_nominal() method within SystematicsBand
        class and scale the ratio difference with nominal bin content.

        Args:
            band_name : str
                name of the stored systematics band

        Return:
            {"up": np.array, "down":np.array}
        """
        try:
            return self.systematics_band[band_name].scale_nominal(self.bin_content)
        except KeyError:
            full_path = f"{self.parent.parent.name}/{self.parent.name}/{self.name}"
            raise KeyError(
                f"{full_path} has no syst. band '{band_name}' in {self.systematics_band}"
            )

    def total_band(
        self,
        exclude_types=None,
        exclude_names=None,
        include_stats=False,
        *,
        ext_band=None,
        update=False,
    ):
        bands = ext_band or self.systematics_band
        if not bands:
            logger.warning("No systematics band was found.")
            return None
        if "total" in bands:
            if not update:
                return bands["total"]
            del bands["total"]
        exclude_types = [] if exclude_types is None else exclude_types
        exclude_names = [] if exclude_names is None else exclude_names
        total = SystematicsBand("total", "total", self.shape)
        for band in bands.values():
            if band.name in exclude_names or band.type in exclude_types:
                continue
            total.update_sub_bands(band, exist_ok=False)
        if include_stats:
            stats_error = self.statistical_error(ratio=True)
            stats_band = SystematicsBand("stats", "stats", self.shape)
            stats_band.add_component("up", "stats", stats_error["up"])
            stats_band.add_component("down", "stats", stats_error["down"])
            total.update_sub_bands(stats_band, exist_ok=False)
        return total

    def statistical_error(self, scale_nominal=False, ratio=False):
        """
        generate and return poisson statistical up/down error
        """
        stats_error = np.sqrt(self._sumW2)
        if scale_nominal:
            up_error = self._bin_content + stats_error
            down_error = self._bin_content - stats_error
        elif ratio:
            stats_error /= self._bin_content
            np.nan_to_num(stats_error, copy=False)
            up_error = stats_error
            down_error = stats_error
        else:
            up_error = stats_error
            down_error = stats_error
        return {"up": up_error, "down": down_error}

    def merge_overflow(self):
        if self.systematics_band:
            norm_uf = self._bin_content[:2]
            norm_of = self._bin_content[-2:]
            merged_uf = norm_uf.sum()
            merged_of = norm_of.sum()
            for band in self.systematics_band.values():
                comp_names = band.component_names()
                band_comps = band.components
                up_comp = band_comps["up"]
                dn_comp = band_comps["down"]
                for name in comp_names:
                    up_under = np.sum(up_comp[name][:2] * norm_uf) / merged_uf
                    up_over = np.sum(up_comp[name][-2:] * norm_of) / merged_of
                    up_comp[name][[1, -2]] = [up_under, up_over]
                    up_comp[name][[0, -1]] = 0.0
                    dn_under = np.sum(dn_comp[name][:2] * norm_uf) / merged_uf
                    dn_over = np.sum(dn_comp[name][-2:] * norm_of) / merged_of
                    dn_comp[name][[1, -2]] = [dn_under, dn_over]
                    dn_comp[name][[0, -1]] = 0.0
            self._bin_content[[1, -2]] = [merged_uf, merged_of]
        else:
            self._bin_content[[1, -2]] += self._bin_content[[0, -1]]
        self._sumW2[[1, -2]] += self._sumW2[[0, -1]]
        # set under/overflow bins to zeros.
        self._bin_content[[0, -1]] = 0.0
        self._sumW2[[0, -1]] = 0.0

    def basic_info(self):
        _basic_info = {"name", "nbin", "xmin", "xmax", "xtitle", "ytitle"}
        return {name: getattr(self, name) for name in _basic_info}

    def extend_bins(self, type="underflow", shallow=False, inplace=False):
        """
        Extending bins to under/overflow bins.
        """
        bins = self.bins
        bwidth = self.bin_width
        bin_content = self.bin_content
        sumW2 = self.sumW2
        if type == "underflow":
            new_bins = ([bins[0] - bwidth[0]], bins)
            new_content = ([0], bin_content)
            new_sumW2 = ([0], sumW2)
        elif type == "overflow":
            new_bins = (bins, [bins[-1] + bwidth[-1]])
            new_content = (bin_content, [0])
            new_sumW2 = (sumW2, [0])
        else:
            new_bins = ([bins[0] - bwidth[0]], bins, [bins[-1] + bwidth[-1]])
            new_content = ([0], bin_content, [0])
            new_sumW2 = ([0], sumW2, [0])
        c_hist = self if inplace else self.copy(shallow=shallow)
        c_hist.bins = np.concatenate(new_bins)
        c_hist.bin_content = np.concatenate(new_content)
        c_hist.sumW2 = np.concatenate(new_sumW2)
        return c_hist

    def collapse_to_underflow(self, shallow=False, inplace=False):
        c_self = self.copy(shallow=shallow) if not inplace else self
        c_self.bin_content[0] = c_self.bin_content.sum()
        c_self.bin_content[1:].fill(0)
        c_self.sumW2[0] = c_self.sumW2.sum()
        c_self.sumW2[1:].fill(0)
        return c_self

    def loc(self, value):
        return np.digitize(value, self._bins)

    def root_graph(self, *args, **kwargs):
        return to_root_graph(self, *args, **kwargs)

    @property
    def root(self):
        return to_th1(self)

    @root.setter
    def root(self, roothist):
        from_th1(self, roothist)
