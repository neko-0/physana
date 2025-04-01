from typing import List, Dict, Union, Set, Optional

TypeDSIDList = Union[List[int], List[str]]

DSID_MAP: Dict[str, TypeDSIDList] = {
    "data": ["periodAllYear"],
    "zjets_strong": list(range(700320, 700326)) + list(range(700335, 700338)),
    "zjets_strong_fxfx": list(range(506193, 506199)),
    "zjets_strong_fxfx_tau": list(range(512198, 512201)),
    "zjets_EW": list(range(700358, 700362)),
    "wjets_strong": list(range(700338, 700350)),
    "wjets_EW": list(range(700362, 700365)),
    "ttbar": list(range(700659, 700663)),
    "singletop_tw": [601352, 601355],
    "singletop_st": [410644, 410645, 410658, 410659, 600027, 600028],
    "diboson_strong": list(range(700600, 700606)),
    "diboson_EW": list(range(700587, 700595)),
}

MC_CAMPAIGN: Dict[str, Set[str]] = {
    "mc20a": {"r13167", "grp15", "grp16"},
    "mc20d": {"r13144", "grp17"},
    "mc20e": {"r13145", "grp18"},
}


def get_ntuple_files(
    ntuple_files: List[str],
    process_dsid_map: Optional[Dict[str, TypeDSIDList]] = None,
    mc_campaign: Optional[Dict[str, Set[str]]] = None,
) -> Dict[str, Dict[str, List[str]]]:
    """
    Maps ntuple files to processes and campaign.

    Args:
        ntuple_files (list): List of ntuple files
        process_dsid_map (dict): Maps process to DSID
        mc_campaign (dict): Maps campaign to release tags

    Returns:
        dict: Maps campaign to process to list of files
    """

    if process_dsid_map is None:
        process_dsid_map = DSID_MAP
    if mc_campaign is None:
        mc_campaign = MC_CAMPAIGN

    process_file_map: Dict[str, Dict[str, List[str]]] = {
        mc: {process: [] for process in process_dsid_map} for mc in mc_campaign
    }

    files_generator = (
        (file, mc, rtag) for mc, rtag in mc_campaign.items() for file in ntuple_files
    )

    for file, mc, rtag in files_generator:
        if not any([f"{r}" in file for r in rtag]):
            continue
        for process, dsids in process_dsid_map.items():
            if not any([f"{d}" in file for d in dsids]):
                continue
            process_file_map[mc][process].append(file)

    return process_file_map
