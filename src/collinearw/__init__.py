# try:
#     # get the RooUnfold loaded up first, before ROOT
#     import RooUnfold
# except ImportError:
#     RooUnfold = None
#     pass

# from .version import __version__
# from .core import Process, Region, Histogram, Histogram2D
# from .core import ProcessSet, Systematics
# from .configMgr import ConfigMgr
# from .tableMaker import TableMaker
# from .plotMaker import PlotMaker, PlotJob
# from .histManipulate import HistManipulate
# from .histMaker import HistMaker
# from . import strategies
#
# __all__ = [
#     '__version__',
#     'ConfigMgr',
#     'Systematics',
#     'Region',
#     'Process',
#     'ProcessSet',
#     'Histogram',
#     'Histogram2D',
#     'ConfigMgr',
#     'PlotMaker',
#     'TableMaker',
#     'PlotJob',
#     'HistManipulate',
#     'HistMaker',
#     'run_ABCD_TF',
#     'run_ABCD_Fakes',
#     'strategies',
# ]

# del RooUnfold

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=[
        'strategies',
        'histManipulate',
        'run_HistMaker',
        'run_PlotMaker',
        'tools',
    ],
    submod_attrs={
        'version': ['__version__'],
        'core': [
            'ProcessSet',
            'Systematics',
            'Process',
            'Region',
            'Histogram',
            'Histogram2D',
        ],
        'configMgr': ['ConfigMgr', 'XSecSumEvtW'],
        'tableMaker': ['TableMaker'],
        'plotMaker': ['PlotMaker', 'PlotJob'],
        'histManipulate': ['HistManipulate', 'run_ABCD_Fakes', 'run_ABCD_TF'],
        'histMaker': ['HistMaker'],
        'run_HistMaker': ['histmaker_generic_interface'],
    },
)
