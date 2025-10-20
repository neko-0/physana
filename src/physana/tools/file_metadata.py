from pathlib import Path
import re

import numpy as np
import uproot

# Precompile regex patterns once
VERSION_PATTERN = re.compile(r'\bAB\d+-TCPT\d+-Vjj\d+\b')
NUMBER_PATTERN = re.compile(r'\d+')


class FileMetaData:
    """
    A class to extract metadata from a ROOT file.

    Parameters
    ----------
    ifile : str or uproot.reading.ReadOnlyDirectory
        The path to the ROOT file.

    Attributes
    ----------
    data_type : str
        The type of the file, e.g., "mc21" or "data".
    campaign : str
        The campaign of the file, e.g., "mc16a" or "2015".
    dataset_id : int
        The dataset ID of the file.
    e_tag : str
        The e-tag of the file.
    num_executed_files : int
        The number of executed files.
    num_input_events : int
    num_events : int
        The number of events in the file.
    cutbookkeeper : dict[tuple[int, int, str], numpy.ndarray]
        A dictionary where each key is a tuple of (dataset ID, run number, system name)
        and each value is a numpy array containing the total number of events,
        the sum of weights, and the sum of squared weights.
    file_path : str
        The path to the file.
    reco_tree_entries : int
        The number of reco tree entries.
    particle_tree_entries : int
        The number of particle tree entries.
    num_trees : int
        The number of trees.
    ab_version : int
        The version of the AnalysisBase package.
    tcpt_version : int
        The version of the TCPT package. master branch is 1
    vjj_version : int
        The version of the Vjj package. master branch is 1
    """

    __slots__ = (
        "data_type",
        "campaign",
        "dataset_id",
        "e_tag",
        "num_executed_files",
        "num_input_events",
        "file_path",
        "cutbookkeeper",
        "reco_tree_entries",
        "particle_tree_entries",
        "num_trees",
        "ab_version",
        "tcpt_version",
        "vjj_version",
    )

    def __init__(self, ifile: str | uproot.ReadOnlyDirectory) -> None:
        """
        Initialize the FileMetaData object.

        Parameters
        ----------
        ifile : str or uproot.reading.ReadOnlyDirectory
            The path to the ROOT file.
        """
        self.data_type: str = ""
        self.campaign: str = ""
        self.dataset_id: int = 0
        self.e_tag: str = ""
        self.num_executed_files: int = 0
        self.num_input_events: int = 0
        self.file_path: str = ""
        self.cutbookkeeper: dict = {}
        self.reco_tree_entries: int = 0
        self.particle_tree_entries: int = 0
        self.num_trees: int = 0
        self.ab_version: int = 0
        self.tcpt_version: int = 0  # 1 is master branch
        self.vjj_version: int = 0  # 1 is master branch

        if isinstance(ifile, str):
            with uproot.open(ifile) as root_file:
                self._load_metadata(root_file)
        else:
            self._load_metadata(ifile)

    def _load_metadata(self, tfile: uproot.ReadOnlyDirectory) -> None:
        try:
            labels = tfile['metadata'].axis().labels()
        except KeyError as _error:
            raise KeyError(f"{tfile.file_path} does not contain metadata") from _error
        self.file_path = str(Path(tfile.file_path).resolve())
        self.data_type = labels[0]
        self.campaign = labels[1]
        self.dataset_id = int(labels[2])
        self.e_tag = labels[3]
        self.num_executed_files = tfile['EventLoop_FileExecuted'].num_entries

        # the number of events per algorithm
        # the first entry is the total number of input events
        self.num_input_events = int(tfile['EventLoop_EventCount'].values()[0])

        # get reco and particle tree entries
        if reco_tree := tfile.get("reco"):
            self.reco_tree_entries = reco_tree.num_entries
            self.num_trees += 1
        if particle_tree := tfile.get("particleLevel"):
            self.particle_tree_entries = particle_tree.num_entries
            self.num_trees += 1

        # get cutbookkeeper
        if self.data_type == "data":
            self.cutbookkeeper[(self.campaign, 0, "NOSYS")] = (
                self.num_input_events,
                self.num_input_events,
                self.num_input_events,
            )
        else:
            # what is "CutBookkeeper_Updated"?
            cutbook_keepers = (
                x for x in tfile if "CutBookkeeper" in x and "Updated" not in x
            )
            for obj_name in cutbook_keepers:
                _, dsid, run_number, syst = obj_name.split("_")

                # The systematics is of the form "NOSYS;1"
                syst = syst.split(";")[0]

                lookup = (int(dsid), int(run_number), syst)

                # The cutbookkeeper is a histogram with 3 bins
                # bin 0: the total number of events
                # bin 1: the sum of weights
                # bin 2: the sum of squared weights
                cutbookkeeper_histo = tfile[obj_name].values()
                self.cutbookkeeper[lookup] = (
                    np.double(cutbookkeeper_histo[0]),
                    np.double(cutbookkeeper_histo[1]),
                    np.double(cutbookkeeper_histo[2]),
                )

        # get package version from file path
        pkg_versions = VERSION_PATTERN.search(self.file_path)
        if pkg_versions:
            versions = NUMBER_PATTERN.findall(pkg_versions.group())
            self.ab_version = versions[0]
            self.tcpt_version = versions[1]
            self.vjj_version = versions[2]
