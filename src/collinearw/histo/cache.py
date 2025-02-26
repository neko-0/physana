import contextlib

from .process_set import ProcessSet


@contextlib.contextmanager
def temporary_cache(obj):
    """
    Context manager to temporarily enable caching for the given object.
    """
    if not isinstance(obj, ProcessSet):
        raise TypeError("Object must be an instance of ProcessSet")

    obj.use_cache = True
    try:
        yield
    finally:
        obj.use_cache = False
