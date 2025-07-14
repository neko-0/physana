import hashlib

from .container import BaseAnalysisContainer
from .tools import Filter


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


class HistogramBase(BaseAnalysisContainer):
    __slots__ = (
        "dtype",
        "np_dtype",
        "filter",
        "weights",
        "parent",
        "_bin_content",
        "_sumW2",
        "_store_data",
    )

    def __init__(self, name, dtype=None, filter_paths=None, np_dtype=None):
        super().__init__(name)
        self.name = name
        self.dtype = dtype
        self.np_dtype = np_dtype
        self.filter = Filter(filter_paths)
        self.weights = None  # string expression to be evaluated for weights
        self.parent = None
        self._bin_content = None
        self._sumW2 = None
        self._store_data = False

    @classmethod
    def variable_bin(cls):
        raise NotImplementedError

    @property
    def bin_content(self):
        raise NotImplementedError

    @property
    def sumW2(self):
        raise NotImplementedError

    @property
    def bins(self):
        raise NotImplementedError

    @property
    def bin_width(self):
        raise NotImplementedError

    @property
    def bin_index(self):
        raise NotImplementedError

    @property
    def root(self):
        raise NotImplementedError

    @root.setter
    def root(self, roothist):
        raise NotImplementedError

    @property
    def overflow(self):
        raise NotImplementedError

    @property
    def underflow(self):
        raise NotImplementedError

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
        raise NotImplementedError

    def save(self, *args, **kwargs):
        raise NotImplementedError

    def basic_info(self):
        """
        Return the basic information that are required for reconstructing
        the objects
        """
        raise NotImplementedError

    def extend_bins(self):
        """
        extend the under/overflow bin edge 1 bin width below/above
        """
        raise NotImplementedError
