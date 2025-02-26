'''
see official doc :
https://docs.python.org/3/library/importlib.html#implementing-lazy-imports
'''

import importlib.util
import sys


def lazy_import(name, package=None):
    spec = importlib.util.find_spec(name, package)
    loader = importlib.util.LazyLoader(spec.loader)
    spec.loader = loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    loader.exec_module(module)
    return module
