from sys import getsizeof
from copy import copy, deepcopy
from typing import Dict, Any, Optional, Union, List

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

    def __init__(self, name: str):
        self._name = name
        self.parent: Optional[BaseAnalysisContainer] = None
        self.metadata: Dict[str, Any] = {}
        self._data: Dict[str, Any] = {}
        self._deepcopy_exclusions: set = {
            "parent",
            "_deepcopy_exclusions",
            "_metadata_deepcopy_exclusions",
        }
        self._metadata_deepcopy_exclusions: set = set()

    def __str__(self) -> str:
        return f"{type(self).__name__}:{self.name}"

    def __iter__(self) -> iter:
        return iter(self._data.values())

    def __getitem__(self, key: str) -> Any:
        try:
            return self._data[key]
        except KeyError:
            raise KeyError(f"{key} not found. Possible keys: {', '.join(self._data)}")

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __copy__(self) -> "BaseAnalysisContainer":
        return self._copy()

    def __deepcopy__(
        self, memo: Dict[int, "BaseAnalysisContainer"]
    ) -> "BaseAnalysisContainer":
        return self._copy(memo)

    def __sizeof__(self) -> int:
        total_size = 0
        for attribute in get_all_slots(self):
            if attribute == "parent":
                continue
            total_size += getsizeof(getattr(self, attribute))
        for data_item in self:
            total_size += getsizeof(data_item)
        return total_size

    def _copy(self, memo: Optional[Dict[int, Any]] = None) -> "BaseAnalysisContainer":
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

    def copy(
        self, *, shallow: bool = False, no_parent: bool = False
    ) -> "BaseAnalysisContainer":
        copied_instance = copy(self) if shallow else deepcopy(self)
        if no_parent:
            copied_instance.parent = None
        return copied_instance

    def items(self) -> iter:
        return self._data.items()

    def keys(self) -> iter:
        return self._data.keys()

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, rhs: str) -> None:
        if isinstance(self.parent, BaseAnalysisContainer):
            parent_data = self.parent._data
            parent_data[rhs] = parent_data.pop(self.name)
        self._name = rhs

    @property
    def deepcopy_exclusions(self) -> set:
        return self._deepcopy_exclusions

    @property
    def deepcopy_metadata_exclusions(self) -> set:
        return self._metadata_deepcopy_exclusions

    @property
    def full_name(self) -> str:
        if self.parent is None:
            return self.name
        else:
            return f"{self.parent.full_name}/{self.name}"

    def add_deepcopy_exclusion(self, name: str) -> None:
        self._deepcopy_exclusions.add(name)

    def discard_deepcopy_exclusion(self, name: str) -> None:
        if name not in {
            "parent",
            "_deepcopy_exclusions",
            "_metadata_deepcopy_exclusions",
        }:
            self._deepcopy_exclusions.discard(name)

    def add_metadata_deepcopy_exclusion(self, name: str) -> None:
        self._metadata_deepcopy_exclusions.add(name)

    def discard_metadata_deepcopy_exclusion(self, name: str) -> None:
        self._metadata_deepcopy_exclusions.discard(name)

    def replace_data(self, other: "BaseAnalysisContainer") -> None:
        self._data = other._data
        self.update_children_parent()

    def clear(self) -> None:
        self._data = {}
        self.metadata = {}

    def clear_content(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def append(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def update(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def get(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    def remove(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def update_children_parent(self, *args, **kwargs) -> None:
        pass

    def clear_children_parent(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def root_parent(self) -> Optional["BaseAnalysisContainer"]:
        parent: Optional[BaseAnalysisContainer] = self.parent
        while parent and parent.parent:
            parent = parent.parent
        return parent


class AnalysisContainer(BaseAnalysisContainer):
    __slots__ = (
        "dtype",
        "weights",
        "cstyle_selection",
        "ntuple_branches",
        "_selection",
    )

    def __init__(
        self,
        name: str,
        weights: Optional[str] = None,
        selection: Optional[str] = None,
        dtype: Optional[str] = None,
    ) -> None:
        super().__init__(name)
        self.dtype: Optional[str] = dtype
        self.weights: Optional[str] = weights
        self.cstyle_selection: Optional[str] = selection
        self._selection: str = to_numexpr(selection)
        self.ntuple_branches: set = set()

    @property
    def selection(self) -> str:
        return self._selection

    @selection.setter
    def selection(self, value: Union[str, List[str]]) -> None:
        if isinstance(value, list):
            self.cstyle_selection = " && ".join(value)
        else:
            self.cstyle_selection = value
        self._selection = to_numexpr(value)

    def add_selection(self, rhs: str) -> None:
        new_selection = " && ".join([self.cstyle_selection, rhs])
        self.selection = new_selection
