from copy import copy, deepcopy


class SystematicsBase:
    """
    Base class for systematics.

    Parameters
    ----------
    name : str
        The name of the systematic.
    syst_type : str
        The type of the systematic.
    """

    __slots__ = ("name", "syst_type", "is_dummy", "_full_name")

    def __init__(self, name: str, syst_type: str):
        self.name = name
        self.syst_type = syst_type
        self.is_dummy = False
        self._full_name = (name, syst_type)

    def __str__(self) -> str:
        return "_".join(self._full_name)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SystematicsBase):
            raise TypeError(
                "Comparison is only supported between SystematicsBase instances."
            )

        return self._full_name == other._full_name

    @property
    def full_name(self) -> tuple:
        return self._full_name

    def copy(self, shallow: bool = True) -> object:
        """
        Return a copy of the current object.

        Parameters
        ----------
        shallow : bool, optional
            Whether to return a shallow copy or a deep copy. Defaults to True.

        Returns
        -------
        object
            A copy of the current object.
        """

        return copy(self) if shallow else deepcopy(self)


class Systematics(SystematicsBase):
    """
    Parameters
    ----------
    name : str
        The name of the systematic.
    syst_type : str
        The type of the systematic.
    tag : str, optional
        The tag of the systematic. Defaults to "NOSYS".
    """

    __slots__ = ("tag", "metadata")

    def __init__(self, name: str, syst_type: str, tag: str = "NOSYS"):
        super().__init__(name, syst_type)
        self.tag = tag
        self.metadata = {}
        self._full_name = (name, syst_type, tag)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Systematics):
            raise TypeError(
                "Comparison is only supported between Systematics instances."
            )

        return self._full_name == other._full_name
