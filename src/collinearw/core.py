from copy import deepcopy, copy
import contextlib
import logging
import re, fnmatch
import collections
import sys
import hashlib
import scipy.stats
import numpy as np
from .utils import to_numexpr
from .backends.root import from_th1, to_th1, from_th2, to_th2, to_root_graph
from . import jitfunc


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


_all_slots_cache = {}


def get_all_slots(obj):
    obj_class = type(obj).__name__
    attr_keys = _all_slots_cache.get(obj_class, None)
    if attr_keys:
        return attr_keys
    attr_keys = set()
    for cls in obj.__class__.__mro__:
        attr_keys.update(getattr(cls, '__slots__', []))
    if not attr_keys:
        # fall back to __dict__ if no __slots__
        attr_keys = obj.__dict__.keys()
    _all_slots_cache.update({obj_class: attr_keys})
    return attr_keys


class Filter:
    """
    Filter objects based on the full_name paths.

    This allows one to filter on multiple paths simultaneously using the
    full_name attribute of the specified object.

    Example:
        >>> import collinearw
        >>> r = collinearw.core.Region('region', None, None)
        >>> r_ex = Region('regionExclude', None, None)
        >>> filt = Filter(['/regionExclude'])
        >>> filt.accept(r)
        True
        >>> filt.accept(r_ex)
        False
    """

    __slots__ = ['_pattern', '_key']

    def __init__(self, values=None, key='full_name'):
        """
        Construct a Filter object.

        Args:
            values (:obj:`list` of :obj:`str`): A list of values to filter out.

        Returns:
            filter (:class:`~collinearw.core.Filter`): The Filter instance.
        """
        values = values or []
        self._pattern = (
            re.compile('|'.join(fnmatch.translate(value) for value in values))
            if values
            else None
        )
        self._key = key

    def match(self, value):
        """
        Match the excluded values against the provided value.

        Args:
            value (:obj:`str`): The value to check against the filtered values.

        Returns:
            None or matched substring.
        """
        return self._pattern.match(value) if self._pattern is not None else False

    def accept(self, obj):
        """
        Whether the provided object's key is allowed or not based on specified excluded values.

        Args:
            obj (:obj:`object`): An object with Filter.key attribute.

        Returns:
            Bool: ``True`` if the value is accepted or ``False`` if the value is not accepted.
        """
        if self._pattern is None:
            return True
        try:
            value = getattr(obj, self.key)
            if value is None:
                return False
        except AttributeError:
            raise ValueError(f'{obj} does not have a {self.key} attribute')
        return bool(self.match(value))

    def filter(self, obj):
        return not self.accept(obj)

    @property
    def key(self):
        return self._key

    @key.setter
    def key(self, value):
        self._key = value

    @property
    def pattern(self):
        return self._pattern

    @pattern.setter
    def pattern(self, values):
        if not isinstance(values, list):
            raise TypeError(f"Invalid type {type(values)}")
        self._pattern = re.compile(
            '|'.join(fnmatch.translate(value) for value in values)
        )


class SystematicBand:
    def __init__(self, name, type, shape):
        self.name = name
        self.type = type
        self.shape = shape
        # The _components_up/down hold every single components
        self._components_up = {}
        self._components_down = {}
        # the _sub_bands is a sub-structure that contains other
        # SystematicBand object, it's compoents are all covered in _up/_down
        # it's convinent for exploring a group of components.
        self._sub_bands = {}
        self._up = None
        self._down = None
        self._cache_total = False

    def __sizeof__(self):
        size = 0
        for comp in self.components.values():
            value = next(iter(comp.values()))
            size += sys.getsizeof(value)
            size *= len(comp)
        for sub_band in self._sub_bands.values():
            size += sys.getsizeof(sub_band)
        return size

    def __getitem__(self, band_type):
        return self.get_band(band_type)

    def __add__(self, rhs):
        new_obj = SystematicBand(self.name, self.type, self.shape)
        new_obj.combine(self)
        new_obj.combine(rhs)
        return new_obj

    def combine(self, other):
        """
        combining with other systematic band in quadrature.

        Args:
            other : SystematicBand
                SystematicBand object for combining.

        Return:
            no return
        """

        new_components = {"up": {}, "down": {}}
        # combining same components first
        for side in ["up", "down"]:
            self_components = getattr(self, f"_components_{side}")
            other_components = getattr(other, f"_components_{side}")
            for name, comp in other_components.items():
                if name in self_components:
                    self_components[name] = np.sqrt(
                        self_components[name] ** 2 + comp**2
                    )
                else:
                    new_components[side].update({name: deepcopy(comp)})

        # combining sub bands.
        for name, sub_band in other._sub_bands.items():
            if name not in self._sub_bands:
                self.update_sub_bands(sub_band)

        # check if any missing top level components after sub-band merged.
        for side in ["up", "down"]:
            self_components = getattr(self, f"_components_{side}")
            for name, comp in new_components[side].items():
                if name not in self_components:
                    self_components.update({name: comp})

    def add_component(self, type, name, band: np.array):
        if type not in ["up", "down"]:
            raise ValueError(f"Invalid {type}. Use up/down")
        assert band.shape == self.shape
        getattr(self, f"_components_{type}")[name] = band

    def update(self, other, copy=True):
        """
        Update band information from another band
        """
        other = deepcopy(other) if copy else other
        for sub_band in other._sub_bands.values():
            self.update_sub_bands(sub_band, copy=False)
        self.update_components(other, copy=False)

    def update_sub_bands(self, band, exist_ok=True, copy=True):
        """
        method for update sub-band structure from another band.

        Args:
            band : SystematicBand
                a SystematicBand instance to be stored in sub-bands dict.

            exist_ok : bool, default=False
                if it's True, then just updating/overwrite
                sub-band components name collides.
                if it's False, expand sub-band name into it's components
        """
        if band.name not in self._sub_bands:
            self._sub_bands[band.name] = deepcopy(band) if copy else band
        else:
            self._sub_bands[band.name].update_components(band, exist_ok, copy)
        # update the top level components from it's sub-band components
        # no need to make copy here
        for sub_band in self._sub_bands.values():
            self.update_components(sub_band, exist_ok, copy=False)

    def update_components(self, band, exist_ok=True, copy=True):
        if exist_ok:
            _up = band._components_up
            _down = band._components_down
        else:
            bname = band.name
            _up = {f"{bname}/{x}": y for x, y in band._components_up.items()}
            _down = {f"{bname}/{x}": y for x, y in band._components_down.items()}
        if copy:
            _up = deepcopy(_up)
            _down = deepcopy(_down)
        self._components_up.update(_up)
        self._components_down.update(_down)

    def get_band(self, type):
        _band = np.zeros(self.shape)
        for component in getattr(self, f"_components_{type}").values():
            _band += component * component
        return np.sqrt(_band)

    def remove_sub_band(self, name):
        band = self._sub_bands[name]
        for key in band._components_up.keys():
            del self._components_up[key]
        for key in band._components_down.keys():
            del self._components_down[key]
        del self._sub_bands[name]

    @property
    def up(self):
        """
        return the total up band
        """
        if not self._cache_total or self._up is None:
            self._up = self.get_band("up")
        return self._up

    @up.setter
    def up(self, value):
        self._cache_total = True
        self._up = value

    @property
    def down(self):
        """
        return the total down band
        """
        if not self._cache_total or self._down is None:
            self._down = self.get_band("down")
        return self._down

    @down.setter
    def down(self, value):
        self._cache_total = True
        self._down = value

    @property
    def components(self):
        return {"up": self._components_up, "down": self._components_down}

    @property
    def sub_bands(self):
        return self._sub_bands

    def use_cache_total(self, value):
        self._cache_total = value
        if not value:
            self._up = None
            self._down = None

    def scale_nominal(self, nominal: np.array):
        """
        convert ratio band to actual band of difference with respect to bin content
        """
        _up = self.get_band("up") * nominal
        _down = self.get_band("down") * nominal
        return {"up": _up, "down": _down}

    def scale_components(self, scale):
        for value in self._components_up.values():
            value *= scale
        for value in self._components_down.values():
            value *= scale

    def list_sub_bands(self):
        return set(self._sub_bands.keys())

    def flatten(self):
        """
        flatten the band structure and dump it into dict format
        """
        output = {}
        for name in {"name", "type", "shape"}:
            output[name] = getattr(self, name)
        output["components"] = self.components
        return output

    def average(self):
        """
        return the average of the up+down band
        """
        return (self.up + self.down) / 2.0

    def component_names(self):
        return self._components_up.keys()

    def components_as_bands(self, filter_zero=True):
        comp_names = self._components_up.keys()
        for name in comp_names:
            _up = self._components_up[name]
            _dn = self._components_down[name]
            # if filter_zero and np.all(_up == 0.0) and np.all(_dn == 0.0):
            #     continue
            if filter_zero and np.all(_up == 0.0):
                _up[_up == 0.0] = 1e-9
            if filter_zero and np.all(_dn == 0.0):
                _dn[_dn == 0.0] = 1e-9
            if filter_zero and np.abs(np.sum(_up)) < 1e-5:
                _up[_up == 0.0] = 1e-9
            if filter_zero and np.abs(np.sum(_dn)) < 1e-5:
                _dn[_dn == 0.0] = 1e-9
            comp_band = SystematicBand(name, self.type, self.shape)
            comp_band.add_component("up", name, _up)
            comp_band.add_component("down", name, _dn)
            yield comp_band

    def clear(self):
        self._components_up = {}
        self._components_down = {}
        self._sub_bands = {}
        self._up = None
        self._down = None
        self._cache_total = False

    @classmethod
    def loads(cls, band_data):
        """
        loading flatten data from the `flatten` method output
        """
        band = cls(band_data["name"], band_data["type"], band_data["shape"])
        band._components_up.update(band_data["components"]["up"])
        band._components_down.update(band_data["components"]["down"])
        return band


# ==============================================================================
# ==============================================================================


def _hadd(lcontent, lsumW2, rcontent, rsumW2):
    lcontent += rcontent
    lsumW2 += rsumW2


def _hsub(lcontent, lsumW2, rcontent, rsumW2):
    lcontent -= rcontent
    lsumW2 += rsumW2


def _hmul(lcontent, lsumW2, rcontent, rsumW2):
    lsumW2 *= rcontent**2
    lsumW2 += rsumW2 * lcontent**2
    lcontent *= rcontent


def _hdiv(lcontent, lsumW2, rcontent, rsumW2):
    # treating error as un-correlated
    # y = a/b -> sigma_y**2 / y**2 = sigma_a**2/a**2 + sigma_b**2/b**2
    lsumW2 /= lcontent**2
    lsumW2 += rsumW2 / rcontent**2
    lcontent /= rcontent
    lsumW2 *= lcontent**2


class HistogramBase:
    __slots__ = (
        "name",
        "type",
        "color",
        "alpha",
        "backend",
        "linestyle",
        "linewidth",
        "markerstyle",
        "markersize",
        "binerror",
        "fillstyle",
        "filter",
        "weights",
        "_parent",
        "_bin_content",
        "_sumW2",
        "_store_data",
        "_skip_deepcopy",
    )

    def __init__(
        self, name, type=None, color="Black", alpha=1.0, backend=None, filter_paths=None
    ):
        self.name = name
        self.type = type
        self.color = color
        self.alpha = alpha
        self.backend = backend
        self.linestyle = None
        self.linewidth = None
        self.markersize = None
        self.markerstyle = None
        self.binerror = None
        self.fillstyle = None
        self.filter = Filter(filter_paths)
        self.weights = None  # string expression to be evaluated for weights
        self._parent = None
        self._bin_content = None
        self._sumW2 = None
        self._store_data = False
        self._skip_deepcopy = {"_parent"}

    def __sizeof__(self):
        size = 0
        for key in get_all_slots(self):
            # _parent is actually reference to an object(region,process) that
            # also has __sizeof__ to look into child object. Need to skip this
            # to avoid recursive size checking.
            if key == "_parent":
                continue
            size += sys.getsizeof(getattr(self, key))
        return size

    def __copy__(self):
        cls = self.__class__
        copy_self = cls.__new__(cls)
        keys = get_all_slots(self)
        skip_deepcopy_keys = keys & self._skip_deepcopy  # intersection
        copy_keys = keys - skip_deepcopy_keys  # complementary
        for key in skip_deepcopy_keys:
            setattr(copy_self, key, getattr(self, key))
        for key in copy_keys:
            setattr(copy_self, key, copy(getattr(self, key)))
        return copy_self

    def __deepcopy__(self, memo):
        cls = self.__class__
        copy_self = cls.__new__(cls)
        memo[id(self)] = copy_self
        keys = get_all_slots(self)
        skip_deepcopy_keys = keys & self._skip_deepcopy  # intersection
        deepcopy_keys = keys - skip_deepcopy_keys  # complementary
        for key in skip_deepcopy_keys:
            setattr(copy_self, key, getattr(self, key))
        for key in deepcopy_keys:
            setattr(copy_self, key, deepcopy(getattr(self, key), memo))
        return copy_self

    def __getitem__(self):
        raise NotImplementedError("__getitem__ is not implemented")

    @classmethod
    def variable_bin(cls):
        raise NotImplementedError("Not implemented")

    @property
    def bin_content(self):
        raise NotImplementedError("Not implemented")

    @property
    def sumW2(self):
        raise NotImplementedError("Not implemented")

    @property
    def bins(self):
        raise AttributeError("Not implemented")

    @property
    def bin_width(self):
        raise AttributeError("Not implemented")

    @property
    def bin_index(self):
        raise AttributeError("Not implemented")

    @property
    def root(self):
        raise AttributeError("Not implemented")

    @root.setter
    def root(self, roothist):
        raise AttributeError("Not implemented")

    @property
    def overflow(self):
        raise AttributeError("overflow is not implemented")

    @property
    def underflow(self):
        raise AttributeError("underflow is not implemented")

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, parent):
        self._parent = parent

    @property
    def full_name(self):
        try:
            full_name = self.parent.full_name if self.parent else ''
        except AttributeError:
            full_name = self.parent or ''
        return '/' + '/'.join([full_name, self.name]).lstrip('/')

    @property
    def hist_type(self):
        return "base"

    @property
    def ndim(self):
        return None

    def unique_name(self):
        return hashlib.md5(
            f'{self.full_name}_{id(self)}_{self._bin_content}'.encode('utf-8')
        ).hexdigest()

    def rebin(self):
        raise AttributeError("Not implemented")

    def copy(self, *, shallow=False, noparent=False):
        c_self = copy(self) if shallow else deepcopy(self)
        if noparent:
            c_self._parent = None
        return c_self

    def save(self, *args, **kwargs):
        raise AttributeError("Not implemented")

    def clear_content(self):
        raise NotImplementedError("Not implemented")

    def basic_info(self):
        """
        Return the basic information that are required for reconstructing
        the objects
        """
        raise NotImplementedError("Not implemented")

    def extend_bins(self):
        """
        extend the under/overflow bin edge 1 bin width below/above
        """
        raise NotImplementedError("Not implemented")


# ==============================================================================
# ==============================================================================


class Histogram(HistogramBase):
    __slots__ = (
        "nbin",
        "xmin",
        "xmax",
        "xtitle",
        "ytitle",
        "systematic",
        "_bins",
        "_observable",
        "_systematic_band",
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
        color="Black",
        alpha=1.0,
        backend="np",
        observable=None,
        type=None,
        filter_paths=None,
    ):
        super(Histogram, self).__init__(
            name, type, color, alpha, backend, filter_paths=filter_paths
        )
        self.nbin = nbin
        self.xmin = xmin
        self.xmax = xmax
        self.xtitle = xtitle or name
        self.ytitle = ytitle
        self.backend = backend
        self.systematic = None
        self._bins = None

        # observable is the lookup name in the TTree.
        # if observable is not specify, the name will be used as default.
        self._observable = observable

        self._bin_content = np.zeros(len(self.bins) + 1, dtype=np.single)
        self._sumW2 = np.zeros(len(self.bins) + 1, dtype=np.single)

        # storage for systematic bands in nominal histogram
        self._systematic_band = {} if self.systematic else None

        # use for holding the data in from_array
        self._data_array = None

    @classmethod
    def variable_bin(
        cls,
        name,
        bins,
        xtitle,
        *args,
        **kwargs,
    ):
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
        return bool(self.systematic is None)

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
    def systematic_band(self):
        return self._systematic_band

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
        # self._bin_content = np.zeros(len(self.bins) + 1)
        # self._sumW2 = np.zeros(len(self.bins) + 1)
        # use ndarray.fill to avoid new allocation?
        self._bin_content.fill(0.0)
        self._sumW2.fill(0.0)
        self._systematic_band = None

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

    def update_systematic_band(self, band: SystematicBand):
        if self._systematic_band is None:
            self._systematic_band = {}
        # make sure the band has the same bin size
        assert band.shape == self.bin_content.shape
        if band.name in self._systematic_band:
            self._systematic_band[band.name].update(band)
        else:
            self._systematic_band[band.name] = band

    def scale_band(self, band_name):
        """
        Method directly invokes the scale_nominal() method within SystematicBand
        class and scale the ratio difference with nominal bin content.

        Args:
            band_name : str
                name of the stored systematic band

        Return:
            {"up": np.array, "down":np.array}
        """
        try:
            return self.systematic_band[band_name].scale_nominal(self.bin_content)
        except KeyError:
            full_path = f"{self.parent.parent.name}/{self.parent.name}/{self.name}"
            raise KeyError(
                f"{full_path} has no syst. band '{band_name}' in {self.systematic_band}"
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
        bands = ext_band or self.systematic_band
        if not bands:
            logger.warning("No systematic band was found.")
            return None
        if "total" in bands:
            if not update:
                return bands["total"]
            del bands["total"]
        exclude_types = [] if exclude_types is None else exclude_types
        exclude_names = [] if exclude_names is None else exclude_names
        total = SystematicBand("total", "total", self.shape)
        for band in bands.values():
            if band.name in exclude_names or band.type in exclude_types:
                continue
            total.update_sub_bands(band, exist_ok=False)
        if include_stats:
            stats_error = self.statistical_error(ratio=True)
            stats_band = SystematicBand("stats", "stats", self.shape)
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
        if self.systematic_band:
            norm_uf = self._bin_content[:2]
            norm_of = self._bin_content[-2:]
            merged_uf = norm_uf.sum()
            merged_of = norm_of.sum()
            for band in self.systematic_band.values():
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


# ==============================================================================
# ==============================================================================


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
        color="Black",
        alpha=1.0,
        backend="np",
        type=None,
        filter_paths=None,
    ):
        super(Histogram2D, self).__init__(
            name, type, color, alpha, backend, filter_paths=filter_paths
        )
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
        # shape = (len(self.bins[0]) + 1, len(self.bins[1]) + 1)
        # self._bin_content = np.zeros(shape)
        # self._sumW2 = np.zeros(shape)
        # use fill to avoid new allocation?
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


# ==============================================================================
# ==============================================================================


class BaseMixin:
    __slots__ = (
        "name",
        "weights",
        "type",
        "selection_numexpr",
        "_selection",
        "_parent",
        "_ntuple_branches",
        "_skip_deepcopy",
    )

    def __init__(self, name, weights, selection, kind):
        self.name = name
        self.weights = weights
        self._selection = selection
        self.selection_numexpr = to_numexpr(selection)
        self.type = kind
        self._parent = None
        self._ntuple_branches = set()
        self._skip_deepcopy = {"_parent", "_ntuple_branches", "_skip_deepcopy"}

    def __str__(self):
        return f"name:{self.name} selection:{self.selection} type:{self.type}"

    def __deepcopy__(self, memo):
        cls = self.__class__
        copy_self = cls.__new__(cls)
        memo[id(self)] = copy_self
        keys = get_all_slots(self)
        skip_deepcopy_keys = keys & self._skip_deepcopy  # intersection
        deepcopy_keys = keys - skip_deepcopy_keys  # complementary
        for key in skip_deepcopy_keys:
            value = getattr(self, key)
            setattr(copy_self, key, value)
        for key in deepcopy_keys:
            value = deepcopy(getattr(self, key), memo)
            setattr(copy_self, key, value)
        return copy_self

    @property
    def selection(self):
        return self._selection

    @selection.setter
    def selection(self, value):
        self._selection = value
        self.selection_numexpr = to_numexpr(value)

    @property
    def ntuple_branches(self):
        return self._ntuple_branches

    @ntuple_branches.setter
    def ntuple_branches(self, rhs):
        self._ntuple_branches = rhs

    def copy(self, *, shallow=False, noparent=False):
        """
        temp_parent = self.parent
        self.parent = None
        c_self = copy(self) if shallow else deepcopy(self)
        c_self.parent = temp_parent
        self.parent = temp_parent
        return c_self
        """
        self.cache_clear()
        if shallow:
            copy_self = copy(self)
        else:
            # m_parent = self.parent
            # self.clear_children_parent()
            copy_self = deepcopy(self)
            # self.parent = m_parent  # ressign parent after copy
            # self.update_children_parent()
        copy_self.update_children_parent()
        if noparent:
            copy_self._parent = None
        return copy_self

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, parent):
        self._parent = parent

    @property
    def full_name(self):
        try:
            return '/' + '/'.join(
                [self.parent.full_name if self.parent else '', self.name]
            ).lstrip('/')
        except AttributeError:
            return '/' + '/'.join([self.parent or '', self.name]).lstrip('/')

    def update_children_parent(self):
        pass

    def clear_children_parent(self):
        pass

    def add_selection(self, rhs):
        new_selection = " && ".join([self._selection, rhs])
        self.selection = new_selection

    def cache_clear(self):
        pass


class Region(BaseMixin):
    """
    This class handles region (ie. different cuts/selections for a given process)

    def __init__

        paramaters:

            name (str) := name of region for later reference.

            weight (str) := weight, something like (genWeight*eventWeight)

            selection (str) := specific cuts/selections, e.g. jet1Pt>500&&met>200
                this is different from the selection in class Process.

            study_type (str) := study type for later reference

            corr_type (str) := keyword specifying the name of correction type

            processes (list) := processes from class Process

        return:
            None
    """

    __slots__ = (
        "corr_type",
        "histograms",
        "_iter_counter",
        "event_count",
        "effective_event_count",
        "sumW2",
        "total_event_count",
        "branch_reserved",
        "_use_cache",
        "_full_selection",
        "_cached_histogram_dict",
        "hist_type_filter",
    )

    def __init__(
        self,
        name,
        weights,
        selection,
        study_type='plot',
        corr_type='None',
        filter_hist_types=None,
    ):
        super(Region, self).__init__(name, weights, selection, study_type)
        self.corr_type = corr_type
        self.histograms = []
        # self.histograms2D = []
        self._iter_counter = 0
        self.event_count = 0  # pure number event filled into this region
        self.effective_event_count = 0  # effective number of event (with weights)
        self.sumW2 = 0
        self.total_event_count = 0  # number of weighted event before selection.
        self._full_selection = None
        self.branch_reserved = False
        # cached histogram dict
        self._use_cache = False
        self._cached_histogram_dict = None
        self._skip_deepcopy.add("_cached_histogram_dict")
        self.hist_type_filter = Filter(filter_hist_types, key='type')

    @property
    def use_cache(self):
        return self._use_cache

    @use_cache.setter
    def use_cache(self, value):
        self._use_cache = value
        if not self._use_cache:
            self.cache_clear()
        else:
            # just run it once to update cache
            _ = self._histograms_dict

    @property
    def _histograms_dict(self):
        if not self.use_cache:
            return {histogram.name: histogram for histogram in self.histograms}
        if self._cached_histogram_dict is None:
            h_dict = {histogram.name: histogram for histogram in self.histograms}
            self._cached_histogram_dict = h_dict
        return self._cached_histogram_dict

    def __str__(self):
        header = (
            f"Region: {self.name}\n selection: {self.selection}\n type: {self.type}"
        )
        body = ""
        for index, histo in enumerate(self.histograms):
            body += f"[{index}]observable: {histo.name}\n"
        return f"{header}\n{body}"

    def __sizeof__(self):
        size = 0
        for key in get_all_slots(self):
            if key == "_parent":
                continue
            size += sys.getsizeof(getattr(self, key))
        for h in self.histograms:
            size += sys.getsizeof(h)
        return size

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

    def __mul__(self, rhs):
        c_self = self.copy()
        c_self.mul(rhs)
        return c_self

    def __truediv__(self, rhs):
        c_self = self.copy()
        c_self.div(rhs)
        return c_self

    def __iter__(self):
        return self

    def __next__(self):
        if self._iter_counter >= len(self.histograms):
            self._iter_counter = 0
            raise StopIteration
        hist = self.histograms[self._iter_counter]
        self._iter_counter += 1
        return hist

    def __getitem__(self, key):
        return self.histograms[key]

    def __setitem__(self, key, value):
        self.histograms[key] = value

    def cache_clear(self):
        self._cached_histogram_dict = None

    def add(self, rhs):
        try:
            if isinstance(rhs, (int, float, np.floating)):
                self.effective_event_count += rhs
                for hist in self.histograms:
                    hist.add(rhs)
            else:
                if self.name != rhs.name:
                    self.name = f"{self.name}+{rhs.name}"
                self.effective_event_count += rhs.effective_event_count
                # get the dict of self histograms first,
                self_h_dict = self._histograms_dict
                # do the operation and combine
                _cache = []
                for key, value in rhs.items():
                    if key in self_h_dict:
                        self_h_dict[key].add(value)
                    else:
                        _cache.append(value)
                self.add_histograms(_cache, skip_check=True)
        except Exception as _error:
            raise Exception(f"error:add({type(self)},{type(rhs)})") from _error

    def sub(self, rhs):
        try:
            if isinstance(rhs, (int, float, np.floating)):
                self.effective_event_count -= rhs
                for hist in self.histograms:
                    hist.sub(rhs)
            else:
                if self.name != rhs.name:
                    self.name = f"{self.name}-{rhs.name}"
                self.effective_event_count -= rhs.effective_event_count
                # get the dict of self histograms first,
                self_h_dict = self._histograms_dict
                # do the operation and combine
                _cache = []
                for key, value in rhs.items():
                    if key in self_h_dict:
                        self_h_dict[key].sub(value)
                    else:
                        _cache.append(value)
                self.add_histograms(_cache, skip_check=True)
        except Exception as _error:
            raise Exception(f"error:sub({type(self)},{type(rhs)})") from _error

    def mul(self, rhs):
        try:
            if isinstance(rhs, (int, float, np.floating)):
                self.effective_event_count *= rhs
                for hist in self.histograms:
                    hist.mul(rhs)
            else:
                if self.name != rhs.name:
                    self.name = f"{self.name}*{rhs.name}"
                self.effective_event_count *= rhs.effective_event_count
                # get the dict of self histograms first,
                self_h_dict = self._histograms_dict
                # do the operation and combine
                _cache = []
                for key, value in rhs.items():
                    if key in self_h_dict:
                        self_h_dict[key].mul(value)
                    else:
                        _cache.append(value)
                self.add_histograms(_cache, skip_check=True)
        except Exception as _error:
            raise Exception(f"error:mul({type(self)},{type(rhs)})") from _error

    def div(self, rhs):
        try:
            if isinstance(rhs, (int, float, np.floating)):
                rhs_effective_event_count = rhs
                for hist in self.histograms:
                    hist.div(rhs)
            else:
                if self.name != rhs.name:
                    self.name = f"{self.name}/{rhs.name}"
                rhs_effective_event_count = rhs.effective_event_count
                # get the dict of self histograms first,
                self_h_dict = self._histograms_dict
                # do the operation and combine
                _cache = []
                for key, value in rhs.items():
                    if key in self_h_dict:
                        self_h_dict[key].div(value)
                    else:
                        _cache.append(value)
                self.add_histograms(_cache, skip_check=True)
            try:
                self.effective_event_count /= rhs_effective_event_count
            except ZeroDivisionError:
                self.effective_event_count = 0
        except Exception as _error:
            raise Exception(f"error:div({type(self)},{type(rhs)})") from _error

    def add_histogram(self, histo, enable_filter=False, skip_check=False, copy=True):
        """
        Adding histogram into region.
        If there's histogram with same name. Skip if histogram is existed.

        Args:
            histo (obj:Histogram) : Histogram class instance

            enable_filter: bool, default=False
                option for enabling histogram filtering.
                NOTE: this is needed just for backward compability for now, since
                there's some cases this method is used and doesn't care about
                histogram filtering.

        Returns:
            None
        """
        # check if histogram exists first
        if not skip_check and histo.name in self.list_observables():
            logger.debug(f"{histo.name} already exists in {self.full_name}")
            return

        if enable_filter:
            if not histo.filter.accept(self):
                logger.debug(f"{histo.name} rejected {self.full_name}")
                return

        c_histo = histo.copy() if copy else histo
        # update name based on region
        c_histo.parent = self
        self.histograms.append(c_histo)
        # clear method cache
        self.cache_clear()

    def add_histograms(self, histos, enable_filter=False, skip_check=False):
        if not skip_check:
            exist_histo = self.list_observables()
            histos = [histo.copy() for histo in histos if histo.name not in exist_histo]
        else:
            histos = [histo.copy() for histo in histos]
        if not histos:
            return
        # update parent
        for histo in histos:
            histo.parent = self
        self.histograms += histos
        # clear method cache
        self.cache_clear()

    def remove_histogram(self, hist):
        if isinstance(hist, str):
            self.histograms.remove(self.get_histogram(hist))
        else:
            self.histograms.remove(hist)
        # clear method cache
        self.cache_clear()

    def get(self, name):
        try:
            if self.use_cache:
                return self._histograms_dict[name]
            for hist in self.histograms:
                if hist.name != name:
                    continue
                return hist
            else:
                raise KeyError
        except KeyError as _error:
            raise KeyError(f"No {name} in region {self.name}") from _error

    def histograms_names(self):
        for histo in self.histograms:
            yield histo.name

    def list_histograms(self, pattern=None, backend=None):
        histos = self.histograms_names()
        if pattern is None:
            return list(histos)
        if backend != "re":
            return fnmatch.filter(histos, pattern)
        reg_exp = re.compile(pattern)
        return list(filter(reg_exp.search, histos))

    def has_histgoram(self, name):
        return True if name in self._histograms_dict else False

    def items(self):
        """
        Similar to _histogram_dict, but using yield
        """
        for hist in self.histograms:
            yield hist.name, hist

    def clear(self):
        self.event_count = 0
        self.effective_event_count = 0
        self.sumW2 = 0
        self.total_event_count = 0
        self.histograms = []

    def clear_content(self):
        self.cache_clear()
        self.event_count = 0
        self.effective_event_count = 0
        self.sumW2 = 0
        self.total_event_count = 0
        for hist in self.histograms:
            hist.clear_content()

    def scale(self, value):
        self.effective_event_count *= value
        for histogram in self.histograms:
            histogram.mul(value)

    def update_children_parent(self):
        """
        updating the parent information in it's histograms
        """
        for h in self.histograms:
            h.parent = self

    # method aliases for backward compatibility
    get_histogram = get
    get_observable = get
    add_observable = add_histogram
    remove_observable = remove_histogram
    list_observables = list_histograms
    has_observable = has_histgoram


# ==============================================================================
# ==============================================================================


class Process(BaseMixin):
    """
    This class defines the physics process and observables for anayslis

    def __init__

        paramaters:

            name (str) := name of the physics process (eg. ttbar, diboson, ZZ->llll, etc).
                This is stored in base class 'BaseMixin'

            treename (str) := name of the ttree, should be the name of top level process.
                eg. ttbar, diboson etc. If only ZZ->llll is considered, you still use
                diboson as treename. The final lookup name will be treename+systematics,
                eg. ttbar_Nosys.

            selection (str) := this is the top level selection,
                eg. 'DataSetNumber' for selecting subprocess within a given process

            process_type (str) := See Process.types. This is translated to mc/data/fakes in base class 'BaseMixin'

            systematics (list) := list of systematics(detector effect, etc)

        return:
            None
    """

    __slots__ = (
        "treename",
        "systematic",
        "regions",
        "lumi",
        "process_type",
        "combine_tree",
        "color",
        "alpha",
        "title",
        "linestyle",
        "linewidth",
        "markerstyle",
        "markersize",
        "binerror",
        "fillstyle",
        "file_record",
        "_iter_counter",
        "_filename",
        "_use_cache",
        "_cached_region_dict",
    )

    types = {
        "bkg": "mc",
        "data": "data",
        "fakes": "fakes",
        "mc": "mc",
        "mc_syst": "mc",
        "signal": "mc",
        "signal_syst": "mc",
        "signal_alt": "mc",
    }

    def __init__(
        self,
        name,
        treename=None,
        selection="",
        weights=[],
        process_type='mc',
        filename=set(),
        lumi=None,
        combine_tree=None,
        # styles
        color=None,
        alpha=None,
        legendname=None,
        linestyle=1,  # ROOT.kSolid
        linewidth=1,
        markerstyle=8,  # ROOT.kFullDotLarge
        markersize=1,
        binerror=0,  # ROOT.TH1.kNormal
        fillstyle=0,  # Hollow
    ):
        assert (
            process_type in self.types
        ), f"{process_type} must be a valid process: {list(self.types.keys())}"
        super(Process, self).__init__(
            name, weights, selection, self.types.get(process_type)
        )
        self.treename = treename if treename else name
        self.systematic = None
        self.regions = []
        self.lumi = lumi
        self._iter_counter = 0
        self.process_type = process_type
        self.combine_tree = combine_tree
        self.file_record = set()  # keep a record of input files
        self.filename = filename  # setting filename via property
        # styles
        self.color = color
        self.alpha = alpha
        self.title = legendname or name
        self.linestyle = linestyle
        self.linewidth = linewidth
        self.markerstyle = markerstyle
        self.markersize = markersize
        self.binerror = binerror
        self.fillstyle = fillstyle
        # cached region dict
        self._use_cache = False
        self._cached_region_dict = None
        self._skip_deepcopy.add("_cached_region_dict")

    @property
    def use_cache(self):
        return self._use_cache

    @use_cache.setter
    def use_cache(self, value):
        self._use_cache = value
        if not self._use_cache:
            self.cache_clear()
        else:
            # just run it once to update cache
            _ = self._regions_dict

    @property
    def _regions_dict(self):
        if not self.use_cache:
            return {region.name: region for region in self.regions}
        if self._cached_region_dict is None:
            r_dict = {region.name: region for region in self.regions}
            self._cached_region_dict = r_dict
        return self._cached_region_dict

    @property
    def filename(self):
        return self._filename

    @filename.setter
    def filename(self, value):
        if isinstance(value, set):
            self._filename = value
        elif isinstance(value, str):
            self._filename = {value}
        elif isinstance(value, list):
            self._filename = set(value)
        else:
            raise ValueError(f"Invalid filename type {type(value)}.")
        # check if record has been set.
        if not self.file_record:
            self.file_record = copy(self._filename)

    def __str__(self):
        header = f"{super(Process, self).__str__()} Tree:{self.treename}"
        body = ""
        for index, r in enumerate(self.regions):
            body += f"[{index}] Region: {r.name} \n {' '*4}Sel:{r.selection}\n"
        return f"{header}\n{body}"

    def __sizeof__(self):
        size = 0
        for key in get_all_slots(self):
            if key == "_parent":
                continue
            size += sys.getsizeof(getattr(self, key))
        for r in self.regions:
            size += sys.getsizeof(r)
        return size

    def __add__(self, rhs):
        c_self = self.copy()
        c_self.add(rhs)
        return c_self

    def __radd__(self, lhs):
        c_self = self.copy()
        c_self.add(lhs)
        return c_self

    def __sub__(self, rhs):
        c_self = self.copy()
        c_self.sub(rhs)
        return c_self

    def __rsub__(self, lhs):
        c_self = self.copy()
        c_self.sub(lhs)
        c_self.mul(-1)
        return c_self

    def __mul__(self, rhs):
        c_self = self.copy()
        c_self.mul(rhs)
        return c_self

    def __truediv__(self, rhs):
        c_self = self.copy()
        c_self.div(rhs)
        return c_self

    def __iter__(self):
        return self

    def __next__(self):
        if self._iter_counter >= len(self.regions):
            self._iter_counter = 0
            raise StopIteration
        else:
            region = self.regions[self._iter_counter]
            self._iter_counter += 1
            return region

    def __getitem__(self, key):
        return self.regions[key]

    def __setitem__(self, key, value):
        self.regions[key] = value

    def update_filename(self, rhs):
        self._filename |= rhs._filename

    def cache_clear(self):
        self._cached_region_dict = None

    def add(self, rhs):
        try:
            if isinstance(rhs, (int, float, np.floating)):
                for region in self.regions:
                    region.add(rhs)
            else:
                if self.name != rhs.name:
                    self.name = f"{self.name}+{rhs.name}"
                # get the dict of self region first,
                self_r_dict = self._regions_dict
                # do the operation and combine
                _cache = []
                for key, value in rhs.items():
                    if key in self_r_dict:
                        self_r_dict[key].add(value)
                    else:
                        _cache.append(value)
                self.add_regions(_cache, skip_check=True)
        except Exception as _error:
            raise TypeError(f"invalid type {type(rhs)}") from _error

    def sub(self, rhs):
        try:
            if isinstance(rhs, (int, float, np.floating)):
                for region in self.regions:
                    region.sub(rhs)
            else:
                if self.name != rhs.name:
                    self.name = f"{self.name}-{rhs.name}"
                # get the dict of self region first,
                self_r_dict = self._regions_dict
                # do the operation and combine
                _cache = []
                for key, value in rhs.items():
                    if key in self_r_dict:
                        self_r_dict[key].sub(value)
                    else:
                        _cache.append(value)
                self.add_regions(_cache, skip_check=True)
        except Exception as _error:
            raise TypeError(f"invalid type {type(rhs)}") from _error

    def mul(self, rhs):
        try:
            if isinstance(rhs, (int, float, np.floating)):
                for region in self.regions:
                    region.mul(rhs)
            else:
                if self.name != rhs.name:
                    self.name = f"{self.name}*{rhs.name}"
                # get the dict of region first,
                # get the dict of self region first,
                self_r_dict = self._regions_dict
                # do the operation and combine
                _cache = []
                for key, value in rhs.items():
                    if key in self_r_dict:
                        self_r_dict[key].mul(value)
                    else:
                        _cache.append(value)
                self.add_regions(_cache, skip_check=True)
        except Exception as _error:
            raise TypeError(f"invalid type {type(rhs)}") from _error

    def div(self, rhs):
        try:
            if isinstance(rhs, (int, float, np.floating)):
                for region in self.regions:
                    region.div(rhs)
            else:
                if self.name != rhs.name:
                    self.name = f"{self.name}/{rhs.name}"
                # get the dict of self region first,
                self_r_dict = self._regions_dict
                # do the operation and combine
                _cache = []
                for key, value in rhs.items():
                    if key in self_r_dict:
                        self_r_dict[key].div(value)
                    else:
                        _cache.append(value)
                self.add_regions(_cache, skip_check=True)
        except Exception as _error:
            raise TypeError(f"invalid type {type(rhs)}") from _error

    def add_region(self, region):
        if region.name in self.regions_names():
            return
        c_region = region.copy()
        c_region.parent = self
        self.regions.append(c_region)
        # clear cache
        self.cache_clear()

    def add_regions(self, regions, skip_check=False):
        if not skip_check:
            s_regions = self.regions_names()
            regions = [r.copy() for r in regions if r.name not in s_regions]
        else:
            regions = [region.copy() for region in regions]
        if not regions:
            return
        # update parent and combine
        for region in regions:
            region.parent = self
        self.regions += regions
        # clear cache
        self.cache_clear()

    def remove_region(self, region):
        self.regions.remove(self.get_region(region))
        # clear cache
        self.cache_clear()

    def get(self, name):
        try:
            if self.use_cache:
                return self._regions_dict[name]
            for r in self.regions:
                if r.name != name:
                    continue
                return r
            else:
                raise KeyError
        except KeyError:
            syst = self.systematic.full_name if self.systematic else ""
            raise KeyError(f"{self.name}{syst} does not has {name}")

    def pop_region(self, region):
        m_region = self.get_region(region)
        self.regions.remove(m_region)
        return m_region

    def regions_names(self):
        for r in self.regions:
            yield r.name

    def list_regions(self, pattern=None, backend=None):
        regions = [r.name for r in self.regions]
        if pattern is None:
            return regions
        if backend != "re":
            return fnmatch.filter(regions, pattern)
        reg_exp = re.compile(pattern)
        return list(filter(reg_exp.search, regions))

    def scale(self, value, *, skip_type=None):
        logger.info(f"{self.name} rescaling with {value}. Skip type {skip_type}")
        for region in self.regions:
            if skip_type:
                if skip_type in region.type:
                    continue
            region.scale(value)

    def rename(self, new_name):
        self.name = new_name

    def items(self):
        for region in self.regions:
            yield region.name, region

    def clear(self):
        self.regions = []

    def clear_content(self):
        self.cache_clear()
        for r in self.regions:
            r.clear_content()

    @classmethod
    def fakes(cls, fake_name):
        return cls(fake_name, "dummytreename", process_type="fakes")

    @property
    def full_name(self):
        return '/' + '/'.join(
            [
                self.parent.full_name if self.parent else '',
                self.name,
                '_'.join(self.systematic.full_name) if self.systematic else "nominal",
            ]
        ).lstrip('/')

    def update_children_parent(self):
        """
        updating the parent information in it's regions/histograms
        """
        for r in self.regions:
            r.parent = self
            for h in r.histograms:
                h.parent = r

    def clear_children_parent(self):
        """
        In fact, better make it to string instead of object reference.
        The problem is unpickle does not recover the 'weak' ref of self and adding
        too much overhead and duplication.
        """
        self_full_name = self.full_name
        for r in self.regions:
            r_full_name = r.full_name
            for h in r.histograms:
                h.parent = r_full_name
            r.parent = self_full_name

    def copy_metadata(self, rhs, metadata_list=None):
        if not isinstance(rhs, type(self)):
            raise TypeError(f"Invalid type {type(rhs)}")
        if metadata_list is None:
            metadata_list = {
                "process_type",
                "color",
                "alpha",
                "alpha",
                "title",
                "linestyle",
                "linewidth",
                "markerstyle",
                "markersize",
                "fillstyle",
            }
        for metadata in metadata_list:
            setattr(self, metadata, getattr(rhs, metadata))

    # method aliases for backward compatibility
    get_region = get


# ==============================================================================
# ==============================================================================
class ProcessSet:
    """
    Wrapper class for processes to handle systematics.
    """

    __slots__ = (
        "_name",
        "nominal",
        "systematics",
        "computed_systematics",
        "_systematic_types",
        "_use_cache",
        "_name_cache",
        "_full_name_cache",
    )

    def __init__(self, name, *args, **kwargs):
        self._name = name
        self.nominal = None
        self.systematics = []
        self.computed_systematics = {}

        self._systematic_types = None
        self._use_cache = False
        self._name_cache = None
        self._full_name_cache = None

    def __sizeof__(self):
        size = 0
        for key in get_all_slots(self):
            if key == "_parent":
                continue
            size += sys.getsizeof(getattr(self, key))
        size += sys.getsizeof(self.nominal)
        for s in self.systematics:
            size += sys.getsizeof(s)
        return size

    def __iter__(self):
        if self.nominal is not None:
            yield self.nominal
        for systematic in self.systematics:
            yield systematic

    def __add__(self, rhs):
        c_self = self.copy()
        c_self.add(rhs)
        return c_self

    def __sub__(self, rhs):
        c_self = self.copy()
        c_self.sub(rhs)
        return c_self

    def copy(self, *, shallow=False):
        self.cache_clear()
        if shallow:
            return copy(self)
        else:
            return deepcopy(self)

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, rhs):
        self._name = rhs
        for p in self:
            p.name = rhs

    @property
    def title(self):
        # grab the legend title from process.
        for p in self:
            return p.title

    @property
    def systematic_names(self):
        return self._systematics_name_dict.keys()

    @property
    def process_type(self):
        """
        return process level type information, e.g. data, mc, signal etc
        """
        try:
            return self.nominal.process_type
        except:
            try:
                return self.systematics[0].process_type
            except:
                return None

    @property
    def type(self):
        """
        return process level type information, e.g. data, mc, signal etc
        """
        try:
            return self.nominal.type
        except:
            try:
                return self.systematics[0].type
            except:
                return None

    @property
    def _systematics_name_dict(self):
        if self._name_cache is not None:
            return self._name_cache
        else:
            sys_dict = collections.defaultdict(list)
            for sys_p in self.systematics:
                sys_dict[sys_p.systematic.name].append(sys_p)
            if self._use_cache:
                self._name_cache = sys_dict
            return sys_dict

    @property
    def _systematics_full_name_dict(self):
        if self._use_cache and self._full_name_cache is not None:
            return self._full_name_cache
        sys_dict = {sys_p.systematic.full_name: sys_p for sys_p in self.systematics}
        if self._use_cache:
            self._full_name_cache = sys_dict
        return sys_dict

    @property
    def use_cache(self):
        return self._use_cache

    @use_cache.setter
    def use_cache(self, value):
        self._use_cache = value
        if not self._use_cache:
            self.cache_clear()

    def cache_clear(self):
        self._name_cache = None
        self._full_name_cache = None

    def add(self, rhs, copy=True):
        if self.name != rhs.name:
            self.name = f"{self.name}+{rhs.name}"
        if isinstance(rhs, type(self)):
            if self.nominal is None:
                self.nominal = deepcopy(rhs.nominal) if copy else rhs.nominal
            else:
                if rhs.nominal is not None:
                    self.nominal.add(rhs.nominal)
            non_exist = []
            for rhs_sys in rhs.systematics:
                exist = False
                for self_sys in self.systematics:
                    if self_sys.systematic == rhs_sys.systematic:
                        self_sys.add(rhs_sys)
                        exist = True
                        break
                if not exist:
                    non_exist.append(rhs_sys.copy() if copy else rhs_sys)
            self.systematics += non_exist
        elif isinstance(rhs, Process):
            if rhs.systematic is None:
                if self.nominal is None:
                    self.nominal = deepcopy(rhs) if copy else rhs
                else:
                    self.nominal.add(rhs)
            else:
                exist = False
                for self_sys in self.systematics:
                    if self_sys.systematic == rhs.systematic:
                        self_sys.add(rhs)
                        exist = True
                        break
                if not exist:
                    self.systematics.append(rhs.copy() if copy else rhs)
        else:
            raise TypeError(f"ProcessSet cannnot add type {type(rhs)}")

    def sub(self, rhs, copy=True):
        if self.name != rhs.name:
            self.name = f"{self.name}+{rhs.name}"
        if isinstance(rhs, type(self)):
            if self.nominal is None:
                self.nominal = deepcopy(rhs.nominal) if copy else rhs.nominal
            else:
                if rhs.nominal is not None:
                    self.nominal.sub(rhs.nominal)
            non_exist = []
            for rhs_sys in rhs.systematics:
                exist = False
                for self_sys in self.systematics:
                    if self_sys.systematic == rhs_sys.systematic:
                        self_sys.sub(rhs_sys)
                        exist = True
                        break
                if not exist:
                    non_exist.append(rhs_sys.copy() if copy else rhs_sys)
            self.systematics += non_exist
        elif isinstance(rhs, Process):
            if rhs.systematic is None:
                if self.nominal is None:
                    self.nominal = deepcopy(rhs) if copy else rhs
                else:
                    self.nominal.sub(rhs)
            else:
                exist = False
                for self_sys in self.systematics:
                    if self_sys.systematic == rhs.systematic:
                        self_sys.sub(rhs)
                        exist = True
                        break
                if not exist:
                    self.systematics.append(rhs.copy() if copy else rhs)
        else:
            raise TypeError(f"ProcessSet cannnot add type {type(rhs)}")

    def add_systematic(self, systematic):
        if systematic.full_name in self._systematics_full_name_dict:
            raise ValueError(f"Found duplicated systematics {systematic.full_name}")
        self.systematics.append(self.nominal.copy())
        self.systematics[-1].systematic = systematic

    def get(self, syst_name=None):
        """
        Getter method for either nominal or systematic pocess

        parameters:

            syst_name( tuple(str,str,str) ) : systematic lookup name.
                e.g.(name, sys treename, sys weight)

        return:

            Process object
        """
        # just return nominal if None
        if syst_name is None:
            return self.nominal

        # check with systematic full name
        if self._use_cache:
            m_proc = self._systematics_full_name_dict.get(syst_name, None)
        else:
            for syst in self.systematics:
                if syst.systematic.full_name != syst_name:
                    continue
                m_proc = syst
                break
            else:
                m_proc = None
        if m_proc is not None:
            return m_proc

        # check with systematic gropu name
        m_proc = self._systematics_name_dict.get(syst_name, None)
        if m_proc is not None:
            return m_proc

        # finally check with computed systematic name
        return self.computed_systematics.get(syst_name, self.nominal)

    @classmethod
    def create_nominal(cls, name, *args, **kwargs):
        p_set = cls(name)
        p_set.nominal = Process(name, *args, **kwargs)
        return p_set

    @classmethod
    def from_process(cls, process):
        """
        create a ProcessSet from a Process object, and set it to nominal
        """
        p_set = cls(process.name)
        if process.systematic is None:
            p_set.nominal = process
        else:
            p_set.systematics.append(process)
        return p_set

    def systematic_type(self, *, update=False):
        if getattr(self, '_systematic_types', None) and not update:
            return self._systematic_types
        sys_type = []
        for sys_process in self.systematics:
            sys_type.append(
                (
                    sys_process.systematic.name,
                    sys_process.systematic.full_name,
                    sys_process.systematic.sys_type,
                    sys_process.systematic.handle,
                )
            )
        self._systematic_type = sys_type
        return self._systematic_type

    def list_systematic_names(self):
        return self.systematic_names

    def list_systematic_full_name(self):
        return self._systematics_full_name_dict.keys()

    def list_computed_systematics(self):
        return self.computed_systematics.keys()

    def get_computed_systematic(self, name):
        return self.computed_systematics.get(name, self.nominal)

    def reset(self):
        self.nominal = None
        self.systematics = []
        self.computed_systematics = {}

    def generate_systematic_group(self, name, lookup):
        output = []
        syst_list = []
        syst_list += list(self.list_systematic_full_name())
        syst_list += list(self.list_computed_systematics())
        syst_list = set(syst_list)
        for syst_tuple in syst_list:
            if syst_tuple is None:
                continue
            if all([fnmatch.fnmatch(*x) for x in zip(syst_tuple, lookup)]):
                output.append(syst_tuple)
        return {name: output}

    def num_processes(self):
        n_syst = len(self.systematics)
        return n_syst if self.nominal is None else 1 + n_syst

    def remove_systematic(self, name=None, full_name=None):
        if name is None and full_name is None:
            return
        filtered = self.systematics
        if name:
            filtered = [x for x in filtered if x.systematic.name != name]
        if full_name:
            filtered = [x for x in filtered if x.systematic.full_name != full_name]

        self.systematics = filtered


# ==============================================================================


@contextlib.contextmanager
def temporary_cache(obj):
    """
    function that turn on the cache temperorarily.
    """
    if not isinstance(obj, (Region, Process, ProcessSet)):
        raise
    obj.use_cache = True
    yield
    obj.use_cache = False


# ==============================================================================
# ==============================================================================
class SystematicBase:
    __slots__ = ("name", "full_name", "source", "sys_type", "handle", "symmetrize")

    def __init__(
        self, name, full_name, source, type="dummy", handle=None, symmetrize=False
    ):
        self.name = name
        self.full_name = full_name
        self.source = source
        self.sys_type = type
        self.handle = handle
        self.symmetrize = symmetrize

    def __str__(self):
        return f"<Systematic {self.full_name}>"

    def __eq__(self, rhs):
        return (
            (self.full_name == rhs.full_name)
            and self.sys_type == rhs.sys_type
            and self.source == rhs.source
        )

    def copy(self, *, shallow=False):
        return copy(self) if shallow else deepcopy(self)


class Systematics(SystematicBase):
    """
    Class for handling systematics.
    """

    __slots__ = ("treename", "weight", "normalize", "swap_weight")

    def __init__(
        self,
        name,
        treename,
        weight,
        source,
        *,
        sys_type=None,
        handle=None,
        normalize=False,
        symmetrize=False,
        swap_weight=None,
    ):
        """
        name (str) : name of the systematics (this is the top level name, can be generic)

        treename (str) : name of the TTree that identify the kinamatic/tree based systematics

        weight (str) : name of the weight inside the TTree.

        source (str) : source of the systematics (tree/weight based)

        sys_type (str) : tells us how to combine systematics downsstream

        source : come from tree or weight
        """
        full_name = (name, treename, weight)
        super(Systematics, self).__init__(
            name, full_name, source, sys_type, handle, symmetrize
        )
        self.treename = treename
        self.weight = weight
        self.normalize = normalize
        self.swap_weight = swap_weight

    def __eq__(self, rhs):
        return (
            (self.full_name == rhs.full_name)
            and self.sys_type == rhs.sys_type
            and self.source == rhs.source
        )

    def __repr__(self):
        if self.source == 'weight':
            return f'<Systematic ({self.weight})>'
        else:
            return f'<Systematic ({self.treename})>'


# ==============================================================================
# ==============================================================================


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
