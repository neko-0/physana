import logging

import numpy as np
from typing import Optional, List, Dict, Any, Union, TYPE_CHECKING

if TYPE_CHECKING:
    import uproot
    import ROOT

from .jitfunc import USE_JIT
from .jitfunc import find_content_and_error_1d as jit_content_err_1d
from .jitfunc import alt_chuck_digitize_1d as jit_digitize_1d
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
        "disable_weights",
        "_bins",
        "_observable",
        "_systematics_band",
        "_data_array",
    )

    def __init__(
        self,
        name: str,
        nbin: int,
        xmin: float,
        xmax: float,
        xtitle: str = None,
        ytitle: str = "Number of events",
        observable: str = None,
        dtype: str = None,
        filter_paths: list = None,
        np_dtype: np.dtype = np.double,
    ) -> None:
        super().__init__(name, dtype, filter_paths=filter_paths, np_dtype=np_dtype)
        self.nbin: int = nbin
        self.xmin: float = xmin
        self.xmax: float = xmax
        self.xtitle: str = xtitle or name
        self.ytitle: str = ytitle
        self.systematics: dict = None
        self._bins: np.ndarray = None

        # observable is the lookup name in the TTree.
        # if observable is not specify, the name will be used as default.
        self._observable: str = observable

        self._bin_content: np.ndarray = np.zeros(len(self.bins) + 1, dtype=np_dtype)
        self._sumW2: np.ndarray = np.zeros(len(self.bins) + 1, dtype=np_dtype)

        # storage for systematics bands in nominal histogram
        self._systematics_band: dict = {} if self.systematics else None

        # option for turning of weights when filling
        self.disable_weights: bool = False

        # use for holding the data in from_array
        self._data_array: np.ndarray = None

    @classmethod
    def variable_bin(
        cls, name: str, bins: np.ndarray, xtitle: str, *args, **kwargs
    ) -> "Histogram":
        _obj = cls(name, len(bins) - 1, bins[0], bins[-1], xtitle, *args, **kwargs)
        _obj.bins = np.array(bins, _obj.np_dtype)
        return _obj

    @classmethod
    def array_to_hist(
        cls,
        a: np.ndarray,
        w: np.ndarray = None,
        name: str = None,
        bin_edge: np.ndarray = None,
        *args,
        **kwargs,
    ) -> "Histogram":
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
    def shape(self) -> tuple:
        return self._bin_content.shape

    @property
    def bin_content(self) -> np.ndarray:
        return self._bin_content

    @bin_content.setter
    def bin_content(self, rhs: np.ndarray) -> None:
        if self._bin_content.shape != rhs.shape:
            raise ValueError("Shape does not match!")
        self._bin_content = rhs.astype(self._bin_content.dtype)

    @property
    def sumW2(self) -> np.ndarray:
        return self._sumW2

    @sumW2.setter
    def sumW2(self, rhs: np.ndarray) -> None:
        if self._sumW2.shape != rhs.shape:
            raise ValueError("Shape does not match!")
        self._sumW2 = rhs.astype(self._sumW2.dtype)

    @property
    def observable(self) -> tuple:
        return (self._observable,) if self._observable else (self.name,)

    @observable.setter
    def observable(self, obs: str) -> None:
        self._observable = obs

    @property
    def is_nominal(self) -> bool:
        return bool(self.systematics is None)

    @property
    def bins(self) -> np.ndarray:
        if self._bins is None:
            xwidth = (self.xmax - self.xmin) / self.nbin
            self._bins = np.arange(self.xmin, self.xmax + xwidth, xwidth)
        return self._bins

    @bins.setter
    def bins(self, rhs: Union[list, np.ndarray]):
        if not np.all(np.diff(rhs) > 0):
            raise ValueError(f"bins need to be sorted: {rhs}")
        m_bins = self.bins
        rhs = np.asarray(rhs)
        if m_bins.shape == rhs.shape:
            if all(m_bins == rhs):
                return
        self._bins, m_size = rhs, len(rhs)
        self._bin_content = np.zeros(m_size + 1, dtype=self.np_dtype)
        self._sumW2 = np.zeros(m_size + 1, dtype=self.np_dtype)

    @property
    def bin_width(self) -> np.ndarray:
        return np.pad(np.diff(self.bins), (1, 1), mode='edge')

    @property
    def bin_index(self) -> range:
        return range(len(self._bin_content))

    @property
    def systematics_band(self) -> dict:
        return self._systematics_band

    @property
    def hist_type(self) -> str:
        return "1d"

    @property
    def ndim(self) -> int:
        return 1

    @property
    def underflow(self) -> float:
        return self._bin_content[0]

    @property
    def overflow(self) -> float:
        return self._bin_content[-1]

    def __getitem__(self, index) -> np.ndarray:
        """
        only operate on the bin content
        """
        return self._bin_content[index]

    def __setitem__(self, index, value: np.ndarray) -> None:
        self._bin_content[index] = value

    def __add__(self, rhs: "Histogram") -> "Histogram":
        c_self = self.copy()
        c_self.add(rhs)
        c_self.parent = None
        return c_self

    def __radd__(self, rhs: "Histogram") -> "Histogram":
        return self.__add__(rhs)

    def __sub__(self, rhs: "Histogram") -> "Histogram":
        c_self = self.copy()
        c_self.sub(rhs)
        c_self.parent = None
        return c_self

    def __rsub__(self, lhs: "Histogram") -> "Histogram":
        c_self = self.copy()
        c_self.sub(lhs)
        c_self.mul(-1.0)
        c_self.parent = None
        return c_self

    def __truediv__(self, rhs: Union[float, "Histogram"]) -> "Histogram":
        c_self = self.copy()
        c_self.div(rhs)
        c_self.parent = None
        return c_self

    def __rtruediv__(self, lhs: Union[int, float, np.floating]) -> "Histogram":
        c_self = self.copy()
        c_self._sumW2 /= 1.0 / self._bin_content**2
        c_self._bin_content = lhs / self._bin_content
        c_self._sumW2 *= c_self._bin_content**2
        c_self.parent = None
        return c_self

    def __mul__(self, rhs: Union["Histogram", int, float, np.floating]) -> "Histogram":
        c_self = self.copy()
        c_self.mul(rhs)
        c_self.parent = None
        return c_self

    def __rmul__(self, lhs: Union[int, float, np.floating]) -> "Histogram":
        return self.__mul__(lhs)

    def __abs__(self) -> "Histogram":
        c_self = self.copy()
        np.abs(c_self._bin_content, out=c_self._bin_content)
        c_self.parent = None
        return c_self

    def add(self, rhs: Union["Histogram", int, float, np.floating]) -> None:
        if isinstance(rhs, type(self)):
            _hadd(self._bin_content, self._sumW2, rhs._bin_content, rhs._sumW2)
            if not rhs._data_array:
                return
            if self._data_array is None:
                self._data_array = []
            self._data_array += rhs._data_array
        elif isinstance(rhs, (int, float, np.floating)):
            self._bin_content += self.np_dtype(rhs)
        else:
            raise TypeError(f"Invalid type {type(rhs)}")

    def sub(self, rhs: Union["Histogram", int, float, np.floating]) -> None:
        if isinstance(rhs, type(self)):
            # self._bin_content -= rhs._bin_content
            # self._sumW2 += rhs._sumW2
            _hsub(self._bin_content, self._sumW2, rhs._bin_content, rhs._sumW2)
        elif isinstance(rhs, (int, float, np.floating)):
            self._bin_content -= self.np_dtype(rhs)
        else:
            raise TypeError(f"Invalid type {type(rhs)}")

    def mul(self, rhs: Union["Histogram", int, float, np.floating]) -> None:
        if isinstance(rhs, type(self)):
            _hmul(self._bin_content, self._sumW2, rhs._bin_content, rhs._sumW2)
        elif isinstance(rhs, (int, float, np.floating)):
            # self._sumW2 = self._sumW2 * rhs ** 2
            self._sumW2 *= self.np_dtype(rhs) ** 2
            self._bin_content *= self.np_dtype(rhs)
        else:
            raise TypeError(f"Invalid type {type(rhs)}")

    def div(self, rhs: Union["Histogram", int, float, np.floating]) -> None:
        if isinstance(rhs, type(self)):
            _hdiv(self._bin_content, self._sumW2, rhs._bin_content, rhs._sumW2)
        elif isinstance(rhs, (int, float, np.floating)):
            self.mul(1.0 / rhs)
        else:
            raise TypeError(f"Invalid type {type(rhs)}")

    def clear_content(self) -> None:
        self._bin_content.fill(0.0)
        self._sumW2.fill(0.0)
        self._systematics_band = None

    def total_events(self) -> float:
        return np.sum(self._bin_content)

    def integral(self, opt: str = "") -> float:
        b_width = self.bin_width if "width" in opt else 1.0
        _integral = self._bin_content * b_width
        if "all" in opt:
            return _integral.sum()
        return _integral[1:-1].sum()

    def normalize(self) -> "Histogram":
        c_self = self.copy()
        c_self.div(c_self.integral())
        return c_self

    def remove_negative_bin(self) -> None:
        mask = self._bin_content < 0
        self._sumW2[mask] = 0
        self._bin_content[mask] = 0

    def nan_to_num(
        self, nan: float = 0.0, posinf: float = 0.0, neginf: float = 0.0
    ) -> None:
        np.nan_to_num(self._bin_content, False, nan, posinf, neginf)
        np.nan_to_num(self._sumW2, False, nan, posinf, neginf)

    def get_bin_content(self, index: int) -> float:
        return self._bin_content[index]

    def get_bin_error(self, index: int) -> float:
        return np.sqrt(self._sumW2[index])

    def find_content_and_error(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return jit_content_err_1d(x, self.bins, self._bin_content, self._sumW2)

    @staticmethod
    def digitize(
        data: np.ndarray,
        bins: np.ndarray,
        weight: np.ndarray = None,
        w2: np.ndarray = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        jit default method
        """
        # return jitfunc.digitize_1d(
        #     np.asarray(data, np.double), np.asarray(bins, np.double), weight, w2
        # )
        if USE_JIT:
            return jit_digitize_1d(
                np.asarray(data, np.double), np.asarray(bins, np.double), weight, w2
            )
        else:
            digitized = np.digitize(data, bins)
            nbins = len(bins) + 1
            content = np.bincount(digitized, weights=weight, minlength=nbins)
            if w2 is None and weight is not None:
                w2 = weight**2
            sumW2 = np.bincount(digitized, weights=w2, minlength=nbins)
            return content, sumW2

    def from_array(
        self,
        data: np.ndarray,
        w: np.ndarray = None,
        w2: np.ndarray = None,
        accumulate: bool = True,
    ) -> None:
        """
        load array of data into hitogram
        """
        bins = np.asarray(self.bins)
        data = np.asarray(data)
        if self.disable_weights:
            w, w2 = None, None
        else:
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

    def update_systematics_band(self, band: SystematicsBand) -> None:
        if self._systematics_band is None:
            self._systematics_band = {}
        assert band.shape == self.bin_content.shape
        if band.name in self._systematics_band:
            self._systematics_band[band.name].update(band)
        else:
            self._systematics_band[band.name] = band

    def scale_band(self, band_name: str) -> dict:
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
        exclude_types: Optional[List[str]] = None,
        exclude_names: Optional[List[str]] = None,
        include_stats: bool = False,
        *,
        ext_band: Optional[Dict[str, SystematicsBand]] = None,
        update: bool = False,
    ) -> Optional[SystematicsBand]:
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

    def statistical_error(
        self, scale_nominal: bool = False, ratio: bool = False
    ) -> Dict[str, np.ndarray]:
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

    def merge_overflow(self) -> None:
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
        # set under/overflow bins to zero after merging
        self._bin_content[[0, -1]] = 0.0
        self._sumW2[[0, -1]] = 0.0

    def basic_info(self) -> Dict[str, Any]:
        _basic_info = {"name", "nbin", "xmin", "xmax", "xtitle", "ytitle"}
        return {name: getattr(self, name) for name in _basic_info}

    def extend_bins(
        self, type: str = "underflow", shallow: bool = False, inplace: bool = False
    ) -> "Histogram":
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

    def collapse_to_underflow(
        self, shallow: bool = False, inplace: bool = False
    ) -> "Histogram":
        c_self = self.copy(shallow=shallow) if not inplace else self
        c_self.bin_content[0] = c_self.bin_content.sum()
        c_self.bin_content[1:].fill(0)
        c_self.sumW2[0] = c_self.sumW2.sum()
        c_self.sumW2[1:].fill(0)
        return c_self

    def loc(self, value: Union[float, np.ndarray]) -> Union[int, np.ndarray]:
        return np.digitize(value, self._bins)

    def root_graph(self, *args, **kwargs) -> "ROOT.TGraphAsymmErrors":
        return to_root_graph(self, *args, **kwargs)

    @property
    def root(self) -> "ROOT.TH1":
        """Returns this histogram as a ROOT TH1"""
        return to_th1(self)

    @root.setter
    def root(self, roothist: "ROOT.TH1") -> None:
        """Sets this histogram from a ROOT TH1"""
        from_th1(self, roothist)


def from_uproot_histo(ihisto: "uproot.models.TH.Model_TH1") -> "Histogram":
    name = ihisto.name
    content, bins = ihisto.to_numpy()
    title = ihisto.title
    ohisto = Histogram.variable_bin(name, bins, title)
    # it seems uproot doesn't include underflow/overflow
    ohisto.bin_content[1:-1] = content
    ohisto.sumW2[1:-1] = ihisto.variances()
    return ohisto
