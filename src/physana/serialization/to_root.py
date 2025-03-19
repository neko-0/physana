import warnings
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from ..configs import ConfigMgr
    from ..histo import HistogramBase

FOUND_ROOT = True
try:
    import ROOT
except ImportError:
    warnings.warn("Cannot import ROOT module!")
    FOUND_ROOT = False


def dump_config_histograms(config: ConfigMgr, filename: str) -> None:  # noqa
    """
    Dump histograms from a ConfigMgr object to a ROOT file.

    This function iterates over the process sets in the given ConfigMgr object,
    extracts histograms from each process, and writes them to a specified ROOT file.
    Each histogram is named using a combination of the process name, region name,
    histogram name, and a systematics tag.

    Parameters
    ----------
    config : ConfigMgr
        The configuration object containing process sets with histograms.
    filename : str
        The name of the ROOT file where histograms will be written.

    Raises
    ------
    UserWarning
        If the ROOT module cannot be imported.

    Notes
    -----
    The ROOT file is opened in "RECREATE" mode, which overwrites any existing file
    with the same name. Each histogram is assigned a name and written to the ROOT file.
    If a process does not have systematics, a "NOSYS" tag is used in the name.
    """
    if not FOUND_ROOT:
        warnings.warn("ROOT module is not found!")
        return

    with ROOT.TFile.Open(filename, "RECREATE") as tfile:
        tfile.cd()
        for process_set in config.process_sets:
            for process in process_set:
                tag = process.systematics.tag if process.systematics else "NOSYS"
                for region in process:
                    for histo in region:
                        name = f"{process.name}_{histo.parent.name}_{histo.name}_{tag}"
                        root_histo = histo.root
                        root_histo.SetName(name)
                        root_histo.Write()


def dump_histograms(histograms: List[HistogramBase], output_filename: str) -> None:
    """
    Dump histograms to a ROOT file.

    Parameters
    ----------
    histograms : list or tuple or iterable
        List of Histogram objects
    output_filename : str
        Name of the output ROOT file

    Notes
    -----
    This function requires the ROOT module to be installed.
    """
    if not FOUND_ROOT:
        raise ImportError("ROOT module not found!")

    with ROOT.TFile.Open(output_filename, "RECREATE") as output_file:
        output_file.cd()
        for histogram in histograms:
            root_histogram = histogram.root
            root_histogram.SetName(histogram.name)
            root_histogram.Write()

            HistogramBase
