import logging
import uproot
from typing import List

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DATA_YEAR: dict[str, tuple[int, int]] = {
    "2018": (63002, 6367686831),
    "2017": (50062, 5629238599),
    "2016": (44680, 5383448881),
    "2015": (10216, 1694555330),
}

uproot_open = uproot.open


class FileMetaData:
    """
    A class to extract metadata from a ROOT file.

    Parameters
    ----------
    filename : str
        The path to the ROOT file.

    Attributes
    ----------
    dtype : str
        The type of the file, e.g., "mc21" or "data".
    campaign : str
        The campaign of the file, e.g., "mc16a" or "2015".
    dsid : int
        The dataset ID of the file.
    tag : str
        The e-tag of the file.
    num_executed_files : int
        The number of executed files.
    nevents : int
        The number of events in the file.
    """

    __slots__ = ("dtype", "campaign", "dsid", "tag", "num_executed_files", "nevents")

    def __init__(self, filename: str) -> None:
        """
        Initialize the FileMetaData object.

        Parameters
        ----------
        filename : str
            The path to the ROOT file.
        """
        with uproot_open(filename) as f:
            labels = f['metadata'].axis().labels()
            self.dtype: str = labels[0]
            self.campaign: str = labels[1]
            self.dsid: int = labels[2]
            self.tag: str = labels[3]
            self.num_executed_files: int = f['EventLoop_FileExecuted'].num_entries
            self.nevents: int = int(f['EventLoop_EventCount'].values()[0])


def check_data_completeness(list_of_files: List[str]) -> None:
    """
    Check the completeness of data files.

    Parameters
    ----------
    list_of_files : list of str
        List of file paths to check for completeness.
    """
    nfiles = {year: 0 for year in DATA_YEAR}
    nevents = {year: 0 for year in DATA_YEAR}

    for file in list_of_files:
        fmd = FileMetaData(file)
        if fmd.campaign not in DATA_YEAR:
            continue
        nfiles[fmd.campaign] += fmd.num_executed_files
        nevents[fmd.campaign] += fmd.nevents

    for year, (expected_files, expected_events) in DATA_YEAR.items():
        if nfiles[year] != expected_files:
            logger.warning(
                f"Incomplete files for {year}: {nfiles[year]} != {expected_files} ({nfiles[year]-expected_files})"
            )
        elif nevents[year] != expected_events:
            logger.warning(
                f"Incomplete events for {year}: {nevents[year]} != {expected_events} ({nevents[year]-expected_events})"
            )
        else:
            logger.info(f"Complete data for {year}")
