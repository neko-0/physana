import logging
import re
import fnmatch

import numpy as np

from .container import AnalysisContainer
from .tools import Filter

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class Region(AnalysisContainer):
    """
    This class handles region (ie. different cuts/selections for a given process)
    """

    __slots__ = (
        "hist_type_filter",
        "full_selection",
        "correction_type",
        "total_nevents",
        "filled_nevents",
        "effective_nevents",
        "sumW2",
        "branch_reserved",
        "histo_branches",
    )

    def __init__(
        self,
        name,
        weights,
        selection,
        dtype="reco",
        correction_type='None',
        filter_hist_types=None,
    ):
        super().__init__(name, weights, selection, dtype)
        self.hist_type_filter = Filter(filter_hist_types, key='dtype')
        self.full_selection = None
        self.correction_type = correction_type
        self.total_nevents = 0  # number of weighted event before selection.
        self.filled_nevents = 0  # pure number event filled into this region
        self.effective_nevents = 0  # effective number of event (with weights)
        self.sumW2 = 0
        self.branch_reserved = False
        self.histo_branches = set()

    def __str__(self):
        space = " " * 3
        header = "\n".join(
            [
                f"Region: {self.name}",
                f"├──Selection: {self.selection}",
                f"├──Type: {self.dtype}",
            ]
        )
        body = "\n".join(
            ["├──Observables:"]
            + [
                f"{space}├──[{index}]: {histogram.name}"
                for index, histogram in enumerate(self)
            ]
        )
        return f"{header}\n{body}"

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

    def get_counts(self):
        return self.total_nevents, self.filled_nevents, self.effective_nevents

    def nevents_add(self, rhs_total, rhs_filled, rhs_effective):
        self.total_nevents += rhs_total
        self.filled_nevents += rhs_filled
        self.effective_nevents += rhs_effective

    def nevents_sub(self, rhs_total, rhs_filled, rhs_effective):
        self.total_nevents -= rhs_total
        self.filled_nevents -= rhs_filled
        self.effective_nevents -= rhs_effective

    def nevents_mul(self, rhs_total, rhs_filled, rhs_effective):
        self.total_nevents *= rhs_total
        self.filled_nevents *= rhs_filled
        self.effective_nevents *= rhs_effective

    def nevents_div(self, rhs_total, rhs_filled, rhs_effective):
        self.total_nevents /= rhs_total
        self.filled_nevents /= rhs_filled
        self.effective_nevents /= rhs_effective

    def _name_operation(self, rhs, op):
        if self.name != rhs.name:
            if op == "add":
                self.name = f"{self.name}+{rhs.name}"
            elif op == "sub":
                self.name = f"{self.name}-{rhs.name}"
            elif op == "mul":
                self.name = f"{self.name}*{rhs.name}"

    def _apply_operation(self, rhs, operation):
        """Apply a mathematical operation to this object and another object."""
        if isinstance(rhs, (int, float, np.floating)):
            getattr(self, f"nevents_{operation}")(rhs, rhs, rhs)
            for histogram in self:
                getattr(histogram, operation)(rhs)
        elif not isinstance(rhs, self.__class__):
            raise TypeError(f"Invalid {operation=} with {type(rhs)=}")
        else:
            self._name_operation(rhs, operation)
            getattr(self, f"nevents_{operation}")(*rhs.get_counts())
            for key, value in rhs._data.items():
                if key in self._data:
                    getattr(self._data[key], operation)(value)
                else:
                    self._data[key] = value

    def add(self, rhs):
        self._apply_operation(rhs, "add")

    def sub(self, rhs):
        self._apply_operation(rhs, "sub")

    def mul(self, rhs):
        self._apply_operation(rhs, "mul")

    def div(self, rhs):
        if isinstance(rhs, (int, float, np.floating)):
            rhs_totle_nevents = rhs
            rhs_fille_nevents = rhs
            rhs_effective_nevents = rhs
            for histogram in self:
                histogram.div(rhs)
        elif not isinstance(rhs, self.__class__):
            raise TypeError(f"Invalid division with {type(rhs)=}")
        else:
            if self.name != rhs.name:
                self.name = f"{self.name}/{rhs.name}"
            rhs_totle_nevents = rhs.total_nevents
            rhs_fille_nevents = rhs.filled_nevents
            rhs_effective_nevents = rhs.effective_nevents
            for key, value in rhs._data.items():
                if key in self._data:
                    self._data[key].div(value)
                else:
                    self._data[key] = value
        try:
            self.total_nevents /= rhs_totle_nevents
            self.filled_nevents /= rhs_fille_nevents
            self.effective_nevents /= rhs_effective_nevents
        except ZeroDivisionError:
            self.total_nevents = 0
            self.filled_nevents = 0
            self.effective_nevents = 0

    @property
    def histograms(self):
        """Accessor for histograms stored in the underlying data dictionary."""
        return list(self)

    def _add_histogram(self, histogram, enable_filter=False, copy=True):
        if histogram.name in self._data:
            return

        if enable_filter and not histogram.filter.accept(self):
            return

        if copy:
            copied_histogram = histogram.copy()
            copied_histogram.parent = self
        else:
            copied_histogram = histogram
        self._data[copied_histogram.name] = copied_histogram

    def _add_histograms(self, histograms, enable_filter=False, copy=True):
        for histogram in histograms:
            if histogram.name in self._data:
                continue
            if enable_filter and not histogram.filter.accept(self):
                continue
            if copy:
                copied_histogram = histogram.copy()
                copied_histogram.parent = self
            else:
                copied_histogram = histogram
            self._data[copied_histogram.name] = copied_histogram

    def append(self, histogram, enable_filter=False, copy=True):
        """
        Add a histogram into the region.

        If the histogram already exists in the region, this method does nothing.

        Args:
            histogram: The histogram to be added.
            enable_filter: If True, filter histograms based on their filter functions.
            copy: If True, copy the histogram before adding it to the region.
        """
        if isinstance(histogram, list):
            self._add_histograms(histogram, enable_filter, copy)
        else:
            self._add_histogram(histogram, enable_filter, copy)

    def update(self, histogram):
        self._data[histogram.name] = histogram

    def get(self, name):
        return self._data[name]

    def remove(self, histogram_or_name):
        """
        Remove a histogram from the region by name or the histogram object itself.

        Args:
            histogram_or_name: The histogram object or its name to be removed.
        """
        if isinstance(histogram_or_name, str):
            self._data.pop(histogram_or_name, None)
        else:
            self._data.pop(histogram_or_name.name, None)

    def list_histograms(self, pattern=None, backend=None):
        histos = self._data.keys()
        if pattern is None:
            return list(histos)
        if backend != "re":
            return fnmatch.filter(histos, pattern)
        reg_exp = re.compile(pattern)
        return list(filter(reg_exp.search, histos))

    def clear(self):
        super().clear()

    def clear_content(self):
        self.total_nevents = 0
        self.filled_nevents = 0
        self.effective_nevents = 0
        self.sumW2 = 0
        for hist in self.histograms:
            hist.clear_content()

    def scale(self, value):
        self.effective_nevents *= value
        for histogram in self:
            histogram.mul(value)

    def update_children_parent(self):
        """Update parent information in child histograms."""
        for histogram in self:
            histogram.parent = self

    def clear_children_parent(self):
        for histogram in self:
            histogram.parent = self.full_name
