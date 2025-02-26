from sys import getsizeof
from copy import copy, deepcopy

from .tools import get_all_slots, to_numexpr


class BaseAnalysisContainer:
    __slots__ = (
        "parent",
        "metadata",
        "_name",
        "_data",
        "_deepcopy_exclusions",
        "_metadata_deepcopy_exclusions",
    )

    def __init__(self, name):
        self._name = name
        self.parent = None
        self.metadata = {}
        self._data = {}
        self._deepcopy_exclusions = {
            "parent",
            "_deepcopy_exclusions",
            "_metadata_deepcopy_exclusions",
        }
        self._metadata_deepcopy_exclusions = set()

    def __str__(self):
        return f"{type(self).__name__}:{self.name}"

    def __iter__(self):
        return iter(self._data.values())

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

    def __contains__(self, key):
        return key in self._data

    def __copy__(self):
        return self._copy()

    def __deepcopy__(self, memo):
        return self._copy(memo)

    def __sizeof__(self):
        total_size = 0
        for attribute in get_all_slots(self):
            if attribute == "parent":
                continue
            total_size += getsizeof(getattr(self, attribute))
        for data_item in self:
            total_size += getsizeof(data_item)
        return total_size

    def _copy(self, memo=None):
        """
        Make a copy of this container.

        This method is used by the built-in functions :func:`copy.copy` and
        :func:`copy.deepcopy`. If `memo` is not `None`, it is a dictionary that maps
        objects to their copies. The method first creates a new instance of the same
        class, and then copies the attributes. If `memo` is not `None`, it is used to
        store the mapping of the original object to its copy.

        The method does not copy the `parent` attribute. The `metadata` attribute is
        copied using the `deepcopy` function, unless keys in
        `_metadata_deepcopy_exclusions` are found, in which case the values are
        copied using the `copy` function.

        The method returns the copied object.

        Parameters
        ----------
        memo : dict or None, optional
            A dictionary that maps objects to their copies. If `None`, a new
            dictionary is created.

        Returns
        -------
        BaseAnalysisContainer
            A copy of this container.
        """
        cls = self.__class__
        copied_self = cls.__new__(cls)
        if memo is not None:
            memo[id(self)] = copied_self

        if self._metadata_deepcopy_exclusions:
            copied_self.metadata = {
                key: (
                    value
                    if key in self._metadata_deepcopy_exclusions
                    else (deepcopy(value, memo) if memo else copy(value))
                )
                for key, value in self.metadata.items()
            }
        else:
            copied_self.metadata = (
                deepcopy(self.metadata, memo) if memo else copy(self.metadata)
            )

        for key in get_all_slots(self):
            if key == "metadata":
                continue
            if key not in self._deepcopy_exclusions:
                value = (
                    deepcopy(getattr(self, key), memo)
                    if memo
                    else copy(getattr(self, key))
                )
            else:
                value = getattr(self, key)
            setattr(copied_self, key, value)

        copied_self.update_children_parent()

        return copied_self

    def copy(self, *, shallow=False, no_parent=False):
        copied_instance = copy(self) if shallow else deepcopy(self)
        if no_parent:
            copied_instance.parent = None
        return copied_instance

    def items(self):
        return self._data.items()

    def keys(self):
        return self._data.keys()

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, rhs):
        if isinstance(self.parent, BaseAnalysisContainer):
            parent_data = self.parent._data
            parent_data[rhs] = parent_data.pop(self.name)
        self._name = rhs

    @property
    def deepcopy_exclusions(self):
        return self._deepcopy_exclusions

    @property
    def deepcopy_metadata_exclusions(self):
        return self._metadata_deepcopy_exclusions

    @property
    def full_name(self):
        if self.parent is None:
            return self.name
        else:
            return f"{self.parent.full_name}/{self.name}"

    def add_deepcopy_exclusion(self, name):
        self._deepcopy_exclusions.add(name)

    def discard_deepcopy_exclusion(self, name):
        if name not in {
            "parent",
            "_deepcopy_exclusions",
            "_metadata_deepcopy_exclusions",
        }:
            self._deepcopy_exclusions.discard(name)

    def add_metadata_deepcopy_exclusion(self, name):
        self._metadata_deepcopy_exclusions.add(name)

    def discard_metadata_deepcopy_exclusion(self, name):
        self._metadata_deepcopy_exclusions.discard(name)

    def replace_data(self, other):
        self._data = other._data
        self.update_children_parent()

    def clear(self):
        self._data = {}
        self.metadata = {}

    def clear_content(self, *args, **kwargs):
        raise NotImplementedError

    def append(self, *args, **kwargs):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def get(self, *args, **kwargs):
        raise NotImplementedError

    def remove(self, *args, **kwargs):
        raise NotImplementedError

    def update_children_parent(self, *args, **kwargs):
        pass

    def clear_children_parent(self, *args, **kwargs):
        raise NotImplementedError


class AnalysisContainer(BaseAnalysisContainer):

    __slots__ = (
        "dtype",
        "weights",
        "cstyle_selection",
        "ntuple_branches",
        "_selection",
    )

    def __init__(self, name, weights=None, selection=None, dtype=None):
        super().__init__(name)
        self.dtype = dtype
        self.weights = weights
        self.cstyle_selection = selection
        self._selection = to_numexpr(selection)
        self.ntuple_branches = set()

    @property
    def selection(self):
        return self._selection

    @selection.setter
    def selection(self, value):
        self.cstyle_selection = value
        self._selection = to_numexpr(value)

    def add_selection(self, rhs):
        new_selection = " && ".join([self.cstyle_selection, rhs])
        self.selection = new_selection
