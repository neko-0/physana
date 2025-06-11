from typing import Type, Any, Dict, Tuple, FrozenSet, Callable


def singleton(cls: Type[Any]) -> Callable[..., Any]:
    """Decorator to make a class a singleton."""
    instances: Dict[Type[Any], Any] = {}
    cache: Dict[Tuple[Any, Tuple[Any, ...], FrozenSet[Tuple[str, Any]]], Any] = {}

    def get_instance(*args: Any, disable_singleton: bool = False, **kwargs: Any) -> Any:
        if disable_singleton:
            return cls(*args, **kwargs)
        key = (cls, args, frozenset(kwargs.items()))
        if key not in cache:
            if cls not in instances:
                instances[cls] = cls(*args, **kwargs)
            cache[key] = instances[cls]
        return cache[key]

    return get_instance
