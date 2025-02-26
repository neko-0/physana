import logging
import re
import fnmatch
from copy import copy

import numpy as np

from .container import AnalysisContainer

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class Process(AnalysisContainer):

    __slots__ = (
        "treename",
        "combine_tree",
        "systematics",
        "title",
        "file_record",
        "_input_files",
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
        dtype="mc",
        input_files=set(),
        title=None,
    ):
        """Construct a Process object.

        Parameters
        ----------
        name : str
            Name of the physics process (e.g. ttbar, diboson, ZZ->llll).
        treename : str or iterator(str), optional
            Name of the TTree, should be the name of top level process.
        selection : str, optional
            Top level selection, e.g. 'DataSetNumber' for selecting subprocess within a given process.
        weights : list, optional
            List of weights to apply.
        dtype : str, optional
            Type of process, e.g. 'mc', 'data', 'fakes'.
        input_files : set, optional
            Set of input files.

        Notes
        -----
        The final lookup name will be treename+systematics, e.g. ttbar_Nosys.
        """
        assert dtype in self.types, f"only accept type: {list(self.types)}"

        super().__init__(name, weights, selection, self.types.get(dtype))
        self.treename = treename or "reco"
        self.combine_tree = None
        self.systematics = None
        self._input_files = None
        self.file_record = set()  # keep a record of input files
        self.input_files = input_files  # setting filename via property
        self.title = title or name

    def __str__(self):
        header = f"{super().__str__()} Tree:{self.treename}"
        body = ""
        for index, r in enumerate(self.regions):
            body += f"[{index}] Region: {r.name} \n {' '*4}Sel:{r.selection}\n"
        return f"{header}\n{body}"

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

    @property
    def input_files(self):
        return self._input_files

    @input_files.setter
    def input_files(self, value):
        """Set input files of the process.

        Parameters
        ----------
        value : set | str | list
            Set of input files.
        """
        input_files = set()
        if isinstance(value, set):
            input_files = value
        elif isinstance(value, str):
            input_files = {value}
        elif isinstance(value, list):
            input_files = set(value)
        else:
            raise ValueError(f"Invalid input file type {type(value)}.")
        self._input_files = input_files
        # check if record has been set.
        if not self.file_record:
            self.file_record = copy(input_files)

    def update_input_files(self, rhs):
        self._input_files |= rhs._input_files

    def _name_operation(self, rhs, op):
        if self.name != rhs.name:
            if op == "add":
                self.name = f"{self.name}+{rhs.name}"
            elif op == "sub":
                self.name = f"{self.name}-{rhs.name}"
            elif op == "mul":
                self.name = f"{self.name}*{rhs.name}"
            elif op == "div":
                self.name = f"{self.name}/{rhs.name}"

    def _apply_operation(self, rhs, operation):
        if isinstance(rhs, (int, float, np.floating)):
            for region in self:
                getattr(region, operation)(rhs)
        elif not isinstance(rhs, self.__class__):
            raise TypeError(f"Invalid {operation=} with {type(rhs)=}")
        else:
            self._name_operation(rhs, operation)
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
        self._apply_operation(rhs, "div")

    @property
    def regions(self):
        """Accessor for regions stored in the underlying data dictionary."""
        return list(self)

    def _add_region(self, region, copy=True):
        if region.name in self._data:
            return
        if copy:
            copied_region = region.copy()
            copied_region.parent = self
        else:
            copied_region = region
        self._data[copied_region.name] = copied_region

    def _add_regions(self, regions, copy=True):
        for region in regions:
            if region.name in self._data:
                continue
            if copy:
                copied_region = region.copy()
                copied_region.parent = self
            else:
                copied_region = region
            self._data[copied_region.name] = copied_region

    def append(self, region, copy=True):
        if isinstance(region, list):
            self._add_regions(region, copy)
        else:
            self._add_region(region, copy)

    def update(self, region):
        self._data[region.name] = region

    def get(self, name):
        return self._data[name]

    def remove(self, region_or_name):
        if isinstance(region_or_name, str):
            self._data.pop(region_or_name)
        else:
            self._data.pop(region_or_name.name)

    def regions_names(self):
        yield from self._data.keys()

    def list_regions(self, pattern=None, backend=None):
        regions = [x for x in self.regions_names()]
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
                if skip_type in region.dtype:
                    continue
            region.scale(value)

    def clear(self):
        super().clear()

    def clear_content(self):
        for region in self:
            region.clear_content()

    @property
    def full_name(self):
        parts = [
            self.parent.full_name if self.parent else '',
            self.name,
            str(self.systematics) if self.systematics else 'nominal',
        ]
        return '/'.join(filter(None, parts))

    def update_children_parent(self):
        """
        updating the parent information in it's regions/histograms
        """
        for region in self.regions:
            region.parent = self
            for histogram in region.histograms:
                histogram.parent = region

    def clear_children_parent(self):
        """
        In fact, better make it to string instead of object reference.
        The problem is unpickle does not recover the 'weak' ref of self and adding
        too much overhead and duplication.
        """
        self_full_name = self.full_name
        for region in self.regions:
            r_full_name = region.full_name
            for histogram in region.histograms:
                histogram.parent = r_full_name
            region.parent = self_full_name

    def copy_metadata(self, rhs, metadata_list=None):
        if not isinstance(rhs, type(self)):
            raise TypeError(f"Invalid type {type(rhs)}")
        if metadata_list is None:
            metadata_list = {"dtype", "metadata"}
        for metadata in metadata_list:
            setattr(self, metadata, getattr(rhs, metadata))


# ==============================================================================
# ==============================================================================
