import sys
import re
import fnmatch
from functools import lru_cache
from typing import Any, Set, Callable
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

    def __init__(self, values=None, key='full_name'):
        """
        Construct a Filter object.

        Args:
            values (:obj:`list` of :obj:`str`): A list of values to filter out.

        Returns:
            filter (:class:`~physana.core.Filter`): The Filter instance.
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
