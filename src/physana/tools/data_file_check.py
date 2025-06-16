import logging
from typing import List

from .file_metadata import FileMetaData

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DATA_YEAR: dict[str, tuple[int, int]] = {
    "2018": (115188, 6367686831),
    "2017": (92724, 5629238599),
    "2016": (73570, 5383448881),
    "2015": (29757, 1694555330),
}


def check_data_completeness(list_of_files: List[str]) -> None:
    """
    Check the completeness of data files.

    Parameters
    ----------
    list_of_files : list of str
        List of file paths to check for completeness.
    """
    nfiles: dict[str, int] = {year: 0 for year in DATA_YEAR}
    nevents: dict[str, int] = {year: 0 for year in DATA_YEAR}

    for file in list_of_files:
        fmd = FileMetaData(file)
        if fmd.campaign not in DATA_YEAR:
            continue
        nfiles[fmd.campaign] += fmd.num_executed_files
        nevents[fmd.campaign] += fmd.num_events

    for year, (expected_files, expected_events) in DATA_YEAR.items():
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
