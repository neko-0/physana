import logging
import sys
import fnmatch
import collections
from copy import copy, deepcopy

from .container import BaseAnalysisContainer
from .process import Process

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class ProcessSet(BaseAnalysisContainer):

    __slots__ = (
        "computed_systematics",
        "_nominal",
        "_use_cached_map",
        "_syst_name_map",
    )

    def __init__(self, name, *args, **kwargs):
        super().__init__(name)
        self.computed_systematics = {}
        self._nominal = None
        self._use_cached_map = False
        self._syst_name_map = None

    def __sizeof__(self):
        total_size = super().__sizeof__()
        total_size += sys.getsizeof(self.nominal)
        return total_size

    def __iter__(self):
        if self.nominal is not None:
            yield self.nominal
        yield from self._data.values()

    def __add__(self, rhs):
        c_self = self.copy()
        c_self.add(rhs)
        return c_self

    def __sub__(self, rhs):
        c_self = self.copy()
        c_self.sub(rhs)
        return c_self

    def copy(self, *, shallow=False):
        self.clear_cache()
        if shallow:
            return copy(self)
        else:
            return deepcopy(self)

    @property
    def systematics(self):
        """Acccesor for systematics stored in the underlying data dictionary"""
        return list(self._data.values())

    @property
    def nominal(self):
        if self._nominal is None:
            self._nominal = self._data.pop(None, None)
        return self._nominal

    @nominal.setter
    def nominal(self, rhs):
        self._nominal = rhs

    @BaseAnalysisContainer.name.setter
    def name(self, rhs):
        super().name = rhs
        for process in self:
            process.name = rhs
        # need to clear the cache to update names.
        self._name_cache = None

    @property
    def title(self):
        # grab the legend title from process.
        for process in self:
            return process.title

    @property
    def systematics_names(self):
        return self.syst_name_map.keys()

    @property
    def process_type(self):
        """
        return process level type information, e.g. data, mc, signal etc
        """
        for process in self:
            return process.dtype

    @property
    def type(self):
        """
        return process level type information, e.g. data, mc, signal etc
        """
        return self.process_type

    @property
    def syst_name_map(self):
        if self._use_cached_map and self._syst_name_map is not None:
            return self._syst_name_map

        syst_name_map = collections.defaultdict(list)
        for process in self._data.values():
            syst_name_map[process.systematics.name].append(process)

        if self._use_cached_map:
            self._syst_name_map = syst_name_map

        return syst_name_map

    @property
    def use_cache(self):
        return self._use_cached_map

    @use_cache.setter
    def use_cache(self, value):
        self._use_cached_map = value
        if not self._use_cached_map:
            self.clear_cache()

    def clear_cache(self):
        self._syst_name_map = None

    def append(self, process, copy=True):
        if process.systematics is None and self.nominal is None:
            self.nominal = process.copy() if copy else process
        else:
            self[process.systematics.full_name] = process.copy() if copy else process

    def update_nominal(self, other_process, copy):
        if self.nominal is None:
            self.nominal = other_process.copy() if copy else other_process
        else:
            if other_process is not None:
                self.nominal.add(other_process)

    def update_systematics(self, other_process_set, copy):
        for other_key, other_syst in other_process_set.items():
            if other_key in self:
                self[other_key].add(other_syst)
            else:
                self[other_key] = other_syst.copy() if copy else other_syst

    def add(self, other_process_set, copy=True):
        if self.name != other_process_set.name:
            self.name = f"{self.name}+{other_process_set.name}"
        if isinstance(other_process_set, type(self)):
            self.update_nominal(other_process_set.nominal, copy)
            self.update_systematics(other_process_set, copy)
        elif isinstance(other_process_set, Process):
            if other_process_set.systematic is None:
                self.update_nominal(other_process_set, copy)
            else:
                other_key = other_process_set.systematic.full_name
                if other_key in self:
                    self[other_key].add(other_process_set)
                else:
                    self[other_key] = (
                        other_process_set.copy() if copy else other_process_set
                    )
        else:
            raise TypeError(f"ProcessSet cannot add type {type(other_process_set)}")

    def sub(self, other_process_set, copy=True):
        if self.name != other_process_set.name:
            self.name = f"{self.name}-{other_process_set.name}"
        if isinstance(other_process_set, type(self)):
            self.update_nominal(other_process_set.nominal, copy)
            self.update_systematics(other_process_set, copy)
        elif isinstance(other_process_set, Process):
            if other_process_set.systematic is None:
                self.update_nominal(other_process_set, copy)
            else:
                other_key = other_process_set.systematic.full_name
                if other_key in self:
                    self[other_key].sub(other_process_set)
                else:
                    self[other_key] = (
                        other_process_set.copy() if copy else other_process_set
                    )
        else:
            raise TypeError(f"ProcessSet cannot sub type {type(other_process_set)}")

    def add_systematics(self, systematics):
        syst_full_name = systematics.full_name
        if syst_full_name in self:
            raise ValueError(f"Found duplicated systematics {syst_full_name}")
        syst_process = self.nominal.copy()
        syst_process.systematics = systematics
        self[syst_full_name] = syst_process

    def get(self, syst_name=None):
        """
        Getter method for either nominal or systematic pocess

        parameters:

            syst_name( tuple(str,str,str) ) : systematic lookup name.
                e.g.(name, systematics type, suffix, tag)

        return:

            Process object
        """
        # just return nominal if None
        if syst_name is None:
            return self.nominal

        # check with systematics full name
        m_proc = self._data.get(syst_name, None)
        if m_proc is not None:
            return m_proc

        # check with systematics group name
        m_proc = self.syst_name_map.get(syst_name, None)
        if m_proc is not None:
            return m_proc

        # finally check with computed systematics name
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
        if process.systematics is None:
            p_set.nominal = process
        else:
            p_set.append(process, copy=False)
        return p_set

    def list_systematics_full_name(self):
        return self.keys()

    def list_computed_systematics(self):
        return self.computed_systematics.keys()

    def get_computed_systematics(self, name):
        return self.computed_systematics.get(name, self.nominal)

    def clear(self):
        super().clear()
        self.computed_systematics = {}

    def generate_systematics_group(self, name, lookup):
        output = []
        syst_list = []
        syst_list += list(self.list_systematics_full_name())
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

    def remove(self, name=None, full_name=None):
        if name is None and full_name is None:
            return

        self_data = self._data
        for syst_full_name in list(self_data.keys()):
            if syst_full_name[0] == name:
                del self_data[syst_full_name]
            elif syst_full_name == full_name:
                del self_data[syst_full_name]
