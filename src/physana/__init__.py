import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=[
        'strategies',
        'histManipulate',
        'algorithm',
        'run_PlotMaker',
        'tools',
        'systematics',
    ],
    submod_attrs={
        'version': ['__version__'],
        'histo': [
            'ProcessSet',
            'Process',
            'Region',
            'Histogram',
            'Histogram2D',
        ],
        'serialization': ['Serialization', 'to_root'],
        'configs': ['ConfigMgr', 'XSecSumEvtW'],
        'tableMaker': ['TableMaker'],
        'plotMaker': ['PlotMaker', 'PlotJob'],
        'histManipulate': ['HistManipulate', 'run_ABCD_Fakes', 'run_ABCD_TF'],
        'algorithm': ['BaseAlgorithm', 'HistMaker', 'run_algorithm', 'CutFlowReader'],
    },
)
