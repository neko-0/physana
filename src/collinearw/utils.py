import sys
import os
import warnings
from contextlib import contextmanager
import packaging.version
import numpy as np

HAS_YODA = False
try:
    import yoda

    yoda_version = packaging.version.parse(yoda.__version__)
    yoda_min_version = packaging.version.parse("2.0.0")
    if yoda_version >= yoda_min_version:
        HAS_YODA = True
    else:
        warnings.warn(
            f"YODA version is {yoda_version}, but at least {yoda_min_version} is required."
        )
except ImportError:
    warnings.warn("Cannot import yoda module!")

try:
    import ROOT
except ImportError:
    warnings.warn("Cannot import ROOT module!")


@contextmanager
def stdout_redirected(new_stdout=None):
    new_stdout = new_stdout or open(os.devnull, 'w')
    save_stdout = sys.stdout
    sys.stdout = new_stdout
    try:
        yield new_stdout
    finally:
        sys.stdout = save_stdout


@contextmanager
def root_redirected(new_path, mode="a"):
    ROOT.gSystem.RedirectOutput(new_path, mode)
    try:
        yield None
    finally:
        ROOT.gROOT.ProcessLine("gSystem->RedirectOutput(0);")


@contextmanager
def all_redirected(new_path=None, mode="a"):
    try:
        with stdout_redirected(new_path) as f:
            with root_redirected(f.name, mode):
                yield None
    finally:
        pass


def root2yoda(roothist, path="", *args, **kwargs):
    if not HAS_YODA:
        raise ImportError("Cannot import yoda module!")
    yodahist = yoda.root.to_yoda(roothist, *args, **kwargs)
    if path:
        yodahist.setPath(path)
    return yodahist


def hist2yoda(hist, path="", use_bin_width=True, include_systematic_error=True):
    """
    Convert a Histogram to a YODA Estimate1D.

    Parameters
    ----------
    hist : Histogram
        The histogram to convert.
    path : str
        The path of the YODA histogram.
    use_bin_width : bool
        Multiply the bin contents by the bin widths.
    include_systematic_error : bool
        Include the systematic error in the YODA histogram.

    Returns
    -------
    yodahist : Estimate1D
        The converted YODA histogram.
    """
    if not HAS_YODA:
        raise ImportError("Cannot import yoda module!")

    bin_edges = np.asarray(hist.bins, dtype="float")
    bin_contents = hist.bin_content
    sumw2 = hist.sumW2
    bin_widths = hist.bin_width
    if include_systematic_error:
        total_band = hist.total_band(include_stats=True)
        if total_band is None:
            errors = np.sqrt(sumw2)
        else:
            errors = (total_band.up + total_band.down) * 0.5 * bin_contents
    else:
        errors = np.sqrt(sumw2)

    yodahist = yoda.Estimate1D(bin_edges, path=path, title=hist.name)
    for i in range(len(bin_edges) + 1):
        width_factor = bin_widths[i] if use_bin_width else 1
        yodahist.bin(i).setVal(bin_contents[i] * width_factor)
        yodahist.bin(i).setErr(errors[i] * width_factor)
    yodahist.setAnnotation("XLabel", hist.xtitle)
    yodahist.setAnnotation("YLabel", hist.ytitle)
    return yodahist


def yoda_write(*args, **kwargs):
    if not HAS_YODA:
        raise ImportError("Cannot import yoda module!")
    yoda.write(*args, **kwargs)
