import numpy as np
from functools import lru_cache

from .singleton import singleton


class PMGSampleInfo:
    __slots__ = (
        "dsid",
        "container_name",
        "ami_xsec",
        "filter_eff",
        "k_factor",
        "xsec_unc_up",
        "xsec_unc_down",
        "effective_xsec",
    )

    def __init__(
        self,
        dsid,
        container_name,
        ami_xsec,
        filter_eff,
        k_factor,
        xsec_unc_up,
        xsec_unc_down,
    ):
        self.dsid = dsid
        self.container_name = container_name
        self.ami_xsec = ami_xsec
        self.filter_eff = filter_eff
        self.k_factor = k_factor
        self.xsec_unc_up = xsec_unc_up
        self.xsec_unc_down = xsec_unc_down

        self.effective_xsec = ami_xsec * filter_eff * k_factor


@singleton
class PMGXsec:
    def __init__(self, xsec_file):
        self.xsec_file = xsec_file
        self.sample_info = {}

    def __call__(self, dsid):
        return self.get_xsec(dsid)

    def __getstate__(self):
        # only store the minimum amount of info needed to reconstruct
        return {"xsec_file": self.xsec_file, "sample_info": {}}

    @lru_cache(maxsize=None)
    def calculate_xsec(self, dsid):
        sample = self.sample_info.get(dsid)
        if sample is None:
            return 1.0
        return sample.effective_xsec

    def reset(self):
        self.sample_info = {}
        self.calculate_xsec.cache_clear()

    def parse_pmg_xsec_file(self, xsec_file: str) -> dict:
        """
        Parse a PMG xsec file to create a mapping of dataset IDs to cross section information.

        Parameters
        ----------
        xsec_file : str
            The file path to the PMG cross section data.

        Returns
        -------
        dict
            A dictionary where each key is a dataset ID (int) and each value is a PMGSampleInfo
            object containing cross section details.
        """
        column_types = [
            (0, int),  # dsid
            (2, str),  # container_name
            (4, float),  # ami_xsec
            (6, float),  # filter_eff
            (8, float),  # k_factor
            (10, float),  # xsec_unc_up
            (12, float),  # xsec_unc_down
        ]

        data = np.loadtxt(xsec_file, delimiter="\t", skiprows=1, dtype=str)
        sample_info = {}
        for row in data:
            sample = PMGSampleInfo(*[func(row[col]) for col, func in column_types])
            sample_info[sample.dsid] = sample

        return sample_info

    def get_xsec(self, dsid):
        """
        Get the cross section value(s) for the specified dataset ID(s).

        Parameters
        ----------
        dsid : int, list, tuple, or np.ndarray
            The dataset ID(s) for which to retrieve cross section values.

        Returns
        -------
        float or np.ndarray
            A single cross section value if `dsid` is an int, or an array of values
            if `dsid` is a collection. Returns 1.0 for any `dsid` not found.

        Raises
        ------
        ValueError
            If `dsid` is not of type int, list, tuple, or np.ndarray.
        """
        if not self.sample_info:
            self.sample_info = self.parse_pmg_xsec_file(self.xsec_file)
        if isinstance(dsid, (int, np.integer)):
            return self.calculate_xsec(dsid)
        elif isinstance(dsid, (list, tuple, np.ndarray)):
            calc_xsec = self.calculate_xsec
            dsid_array = np.asarray(dsid)
            unique_dsid, inverse_mask = np.unique(dsid_array, return_inverse=True)
            return np.array([calc_xsec(d) for d in unique_dsid])[inverse_mask]
        else:
            raise ValueError(f'Invalid input type: {type(dsid)}')
