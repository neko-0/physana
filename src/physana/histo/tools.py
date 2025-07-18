import sys
import re
from fnmatch import translate
from functools import lru_cache
from typing import Any, Set, Callable, List, Union, Optional
import formulate

if formulate.__version__ >= "1.0.0":
    from formulate.AST import Symbol


_all_slots_cache = {}


def get_all_slots(obj: Any) -> Set[str]:
    obj_class = type(obj)
    if obj_class not in _all_slots_cache:
        slots = set()
        for cls in obj_class.__mro__:
            slots.update(getattr(cls, '__slots__', ()))
        if not slots:
            slots = set(obj.__dict__.keys())
        _all_slots_cache[obj_class] = slots
    return _all_slots_cache[obj_class]


class RecursionLimit:
    __slots__ = ("old_limit", "limit")

    def __init__(self, limit: int) -> None:
        self.old_limit: int = sys.getrecursionlimit()
        self.limit: int = limit

    def __enter__(self) -> None:
        sys.setrecursionlimit(self.limit)

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        sys.setrecursionlimit(self.old_limit)


@lru_cache(maxsize=None)
def from_root(expr: str, *, rlimit: int = 5000) -> "formulate.AST":
    try:
        with RecursionLimit(rlimit):  # For long expression
            return formulate.from_root(expr)
    except Exception as _error:
        raise Exception(f"unable to parse root {expr}") from _error


@lru_cache(maxsize=None)
def from_numexpr(expr: str, rlimit: int = 5000) -> "formulate.AST":
    try:
        with RecursionLimit(rlimit):
            return formulate.from_numexpr(expr)
    except Exception as _error:
        raise Exception(f"unable to parse numexpr {expr}") from _error


@lru_cache(maxsize=None)
def to_numexpr(expr: str) -> str:
    if expr:
        return from_root(expr).to_numexpr()
    return expr


# Temperary workaround for older version
if formulate.__version__ < "1.0.0":
    get_variables = lambda x: x.variables
else:

    def get_variables(expr: "formulate.AST") -> Set[str]:
        variables_keeper = set()

        def traverse(node):
            if node is None:
                return
            if isinstance(node, Symbol):
                variables_keeper.add(node.symbol)
            if hasattr(node, "left"):
                traverse(node.left)
            if hasattr(node, "right"):
                traverse(node.right)

        traverse(expr)

        return variables_keeper


@lru_cache(maxsize=None)
def get_expression_variables(
    expr: str, parser: Callable[[str], Any] = from_root
) -> Set[str]:
    """
    Get variables from an expression
    """
    return get_variables(parser(expr))


class Filter:
    """
    Filter objects based on the full_name paths.

    This allows one to filter on multiple paths simultaneously using the
    full_name attribute of the specified object.

    Example:
        >>> import physana
        >>> r = physana.core.Region('region', None, None)
        >>> r_ex = Region('regionExclude', None, None)
        >>> filt = Filter(['/regionExclude'])
        >>> filt.accept(r)
        True
        >>> filt.accept(r_ex)
        False
    """

    __slots__ = ['_pattern', '_key']

    def __init__(
        self, values: Optional[List[str]] = None, key: str = 'full_name'
    ) -> None:
        """
        Construct a Filter object.

        Args:
            values: A list of values to filter out.
            key: The attribute name to use for filtering.

        Returns:
            None
        """
        values = values or []
        self._pattern = (
            re.compile('|'.join(translate(value) for value in values))
            if values
            else None
        )
        self._key = key

    def match(self, value: str) -> Union[None, str, bool]:
        """
        Match the excluded values against the provided value.

        Args:
            value (str): The value to check against the filtered values.

        Returns:
            Union[None, str, bool]: The matched substring if a match is found,
            False if no pattern is defined, or None if no match occurs.
        """
        return self._pattern.match(value) if self._pattern is not None else False

    def accept(self, obj: object) -> bool:
        """
        Whether the provided object's key is allowed or not based on specified excluded values.

        Args:
            obj (object): An object with Filter.key attribute.

        Returns:
            bool: ``True`` if the value is accepted or ``False`` if the value is not accepted.
        """
        if self._pattern is None:  # type: ignore
            return True
        try:
            value = getattr(obj, self.key)  # type: ignore
            if value is None:
                return False
        except AttributeError:
            raise ValueError(f'{obj} does not have a {self.key} attribute')
        return bool(self.match(value))

    def filter(self, obj: object) -> bool:
        """
        Whether the provided object's key is excluded or not based on specified excluded values.

        Args:
            obj (object): An object with Filter.key attribute.

        Returns:
            bool: ``True`` if the value is excluded or ``False`` if the value is not excluded.
        """
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
    def pattern(self, values: List[str]) -> None:
        """
        Set the pattern value of the Filter object.

        Args:
            values (List[str]): A list of string values to use as the pattern.

        Raises:
            TypeError: if the input is not a list.
        """
        if not isinstance(values, list):
            raise TypeError(f"Invalid type {type(values)}")
        self._pattern = re.compile('|'.join(translate(value) for value in values))
