import logging
from .rootbackend import RootTH1DBootstrap, RootTH2DBootstrap
from .npbackend import HistogramBootstrap, Histogram2DBootstrap
from ..unfolding import metadata

log = logging.getLogger(__name__)


"""
TODO: ensure same seed is used for a given process?
"""


def add_ROOT_bootstrap_1d(config, *args, **kwargs):
    try:
        bootstrap = RootTH1DBootstrap(*args, **kwargs)
    except TypeError:
        bootstrap = RootTH1DBootstrap.variable_bin(*args, **kwargs)
    config.append_histogram_1d(bootstrap)


def add_ROOT_bootstrap_2d(config, *args, **kwargs):
    try:
        bootstrap = RootTH2DBootstrap(*args, **kwargs)
    except TypeError:
        bootstrap = RootTH2DBootstrap.variable_bin(*args, **kwargs)
    config.append_histogram_2d(bootstrap)


def add_numpy_bootstrap_1d(config, *args, nreplica=100, **kwargs):
    config.add_observable(*args, **kwargs)
    hist = config.histograms.pop(-1)
    rhist = HistogramBootstrap.from_histogram(hist, nreplica=nreplica, **kwargs)
    config.append_histogram_1d(rhist)


def add_numpy_bootstrap_2d(config, *args, nreplica=100, **kwargs):
    config.add_histogram2D(*args, **kwargs)
    hist = config.histograms2D.pop(-1)
    rhist = Histogram2DBootstrap.from_histogram(hist, nreplica=nreplica, **kwargs)
    config.append_histogram_2d(rhist)


def add_observables(
    config,
    name,
    unfold_to,
    bins,
    xmin,
    xmax,
    xtitle,
    backend='root',
    hist_name=None,
    **kwargs,
):
    """
    Creates two 1-D histograms, and 1 2-D histogram.
    """
    log.info(
        f"Defining two observables reco={name}, truth={unfold_to}; and a 2D response=response_matrix_{name}"
    )

    if metadata.has(config, "observables", hist_name or name):
        return False

    observables = {
        "reco": name,
        "truth": unfold_to,
        "response": f"response_matrix_{name}",
    }
    if hist_name:
        hist_name = {
            "reco": hist_name,
            "truth": f"truth_{hist_name}",
            "response": f"response_matrix_{hist_name}",
        }
    else:
        hist_name = observables

    if backend == "root":
        add_bootstrap_1d = add_ROOT_bootstrap_1d
        add_bootstrap_2d = add_ROOT_bootstrap_2d
    elif backend == "numpy":
        add_bootstrap_1d = add_numpy_bootstrap_1d
        add_bootstrap_2d = add_numpy_bootstrap_2d

    add_bootstrap_1d(
        config,
        hist_name["reco"],
        bins,
        xmin,
        xmax,
        xtitle,
        type="reco",
        observable=observables["reco"],
        **kwargs,
    )

    add_bootstrap_1d(
        config,
        hist_name["truth"],
        bins,
        xmin,
        xmax,
        f"Particle level {xtitle}",
        type="truth",
        observable=observables["truth"],
        **kwargs,
    )

    add_bootstrap_2d(
        config,
        hist_name["response"],
        observables["reco"],
        observables["truth"],
        bins,
        xmin,
        xmax,
        bins,
        xmin,
        xmax,
        xtitle,
        f"Particle level {xtitle}",
        type="response",
        **kwargs,
    )

    metadata.save(config, "observables", hist_name["reco"], hist_name)
