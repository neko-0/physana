import logging
from typing import List, Optional

from .file_metadata import FileMetaData

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DATA_YEAR: dict[str, tuple[int, int]] = {
    "2024": (176481, 13605714880),
    "2023": (30164, 2859125391),
    "2022": (33064, 3469785536),
    "2018": (115188, 6367686831),
    "2017": (92724, 5629238599),
    "2016": (73570, 5383448881),
    "2015": (29757, 1694555330),
}


def check_data_completeness(
    list_of_files: List[str],
    metadata_key: str = "campaign",
    data_year: Optional[dict[str, tuple[int, int]]] = None,
) -> None:
    """
    Evaluate the completeness of data files by comparing the number of files and events
    found in the provided list against expected values for each year.

    Parameters
    ----------
    list_of_files : List[str]
        List of file paths to check for completeness.
    metadata_key : str, optional
        Attribute key to access metadata for each file. Defaults to "campaign".
    data_year : dict[str, tuple[int, int]], optional
        Dictionary of expected number of files and events for each year.
        If not provided, uses a built-in dictionary.
    """

    if data_year is None:
        data_year = DATA_YEAR

    nfiles: dict[str, int] = {year: 0 for year in data_year}
    nevents: dict[str, int] = {year: 0 for year in data_year}

    for file in list_of_files:
        fmd = FileMetaData(file)
        lookup = str(getattr(fmd, metadata_key))
        if lookup not in data_year:
            continue
        nfiles[lookup] += fmd.num_executed_files
        nevents[lookup] += fmd.num_events

    for year, (expected_files, expected_events) in data_year.items():
        nfile_fail = nfiles[year] != expected_files
        nevent_fail = nevents[year] != expected_events
        if nfile_fail or nevent_fail:
            file_percent = (nfiles[year] - expected_files) / expected_files * 100
            event_percent = (nevents[year] - expected_events) / expected_events * 100
            logger.warning(f"Year {year}")
            if nfile_fail:
                logger.warning(
                    f"\t files found {nfiles[year]}, expected {expected_files} ({file_percent:.2f}%)"
                )
            if nevent_fail:
                logger.warning(
                    f"\t nevents found {nevents[year]}, expected {expected_events} ({event_percent:.2f}%)"
                )
        else:
            logger.info(f"Complete data for {year}")
