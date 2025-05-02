import uproot


class FileMetaData:
    """
    A class to extract metadata from a ROOT file.

    Parameters
    ----------
    filename : str or uproot.reading.ReadOnlyDirectory
        The path to the ROOT file.

    Attributes
    ----------
    data_type : str
        The type of the file, e.g., "mc21" or "data".
    campaign : str
        The campaign of the file, e.g., "mc16a" or "2015".
    dataset_id : int
        The dataset ID of the file.
    tag : str
        The e-tag of the file.
    num_executed_files : int
        The number of executed files.
    num_events : int
        The number of events in the file.
    """

    __slots__ = (
        "data_type",
        "campaign",
        "dataset_id",
        "tag",
        "num_executed_files",
        "num_events",
    )

    def __init__(self, ifile: str | uproot.ReadOnlyDirectory) -> None:
        """
        Initialize the FileMetaData object.

        Parameters
        ----------
        ifile : str or uproot.reading.ReadOnlyDirectory
            The path to the ROOT file.
        """
        self.data_type: str = None
        self.campaign: str = None
        self.dataset_id: int = None
        self.tag: str = None
        self.num_executed_files: int = None
        self.num_events: int = None

        if isinstance(ifile, str):
            with uproot.open(ifile) as root_file:
                self._load_metadata(root_file)
        else:
            self._load_metadata(ifile)

    def _load_metadata(self, tfile: uproot.ReadOnlyDirectory) -> None:
        if 'metadata' not in tfile:
            raise ValueError("File does not contain metadata")
        labels = tfile['metadata'].axis().labels()
        self.data_type = labels[0]
        self.campaign = labels[1]
        self.dataset_id = labels[2]
        self.tag = labels[3]
        self.num_executed_files = tfile['EventLoop_FileExecuted'].num_entries
        self.num_events = int(tfile['EventLoop_EventCount'].values()[0])
