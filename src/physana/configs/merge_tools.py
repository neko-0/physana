import logging

from tqdm import tqdm

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


def intersect_configs(config1, config2):
    """
    Method for getting intersection of two ConfigMgr instances.
    basically moving the non-overlap processes from config2 to config1.
    """
    config1_app = config1.append_process
    config1_ls = config1.list_processes
    config1_get = config1.get_process_set
    for pset2 in config2.process_sets:
        # check by process set name first.
        # If the name does not exist, just append the entire process set object
        if pset2.name not in config1_ls():
            config1_app(pset2)
            continue
        # loop and check through each process in the process set.
        # this include checking process with systematic.
        pset1 = config1_get(pset2.name)
        diff_p = []  # buffer list to hold processes for later appending.
        diff_p_app = diff_p.append
        for p2 in pset2:
            found_diff = True
            for p1 in pset1:
                # note process set always try to yield nominal first
                # process set will skip nominal if nominal == None
                if p1.systematic is None:
                    if p2.systematic is None:
                        # if nominal is in both configs,
                        # break and move to next one in pset2
                        found_diff = False
                        break
                else:
                    if p2.systematic is None:
                        continue
                    if p1.systematic == p2.systematic:
                        found_diff = False
                        break
            if found_diff:
                diff_p_app(p2)
        for p in diff_p:
            config1_app(p, mode="merge")


def intersection(config_list, *, copy=True):
    """
    Get the intersection of lisf of configMgr objects.
    """
    first_config = None
    for config in tqdm(config_list, unit="config", leave=False):
        if first_config is None:
            first_config = config.copy() if copy else config
        else:
            intersect_configs(first_config, config)
    first_config.update_children_parent()
    return first_config


def merge(config_list, *, copy=True, update_filename=False):
    """
    Merging configMgr in the configMgrList.
    Only processes (level) will be appended.
    """
    first_config = None
    first_config_add = None
    for config in tqdm(config_list, unit="config", leave=False):
        if first_config is None:
            first_config = config.copy() if copy else config
            first_config_add = first_config.add
        else:
            first_config_add(config)
            if not update_filename:
                continue
            pending_proc = (y for x in config.process_sets for y in x)
            for p in pending_proc:
                first_config.get(p.name).get(p.systematic).update_filename(p)
    first_config.update_children_parent()
    return first_config


def intersect_histogram_systematic_band(config1, config2):
    proc_regions = ((x, y) for x in config2.processes for y in x.regions)
    # only nominal process will have the systematic band on histograms
    for process2, region2 in proc_regions:
        try:
            region1 = config1.get_process(process2.name).get(region2.name)
        except KeyError:
            logger.warning(f"Cannot find {process2.name}/{region2.name}. Skipping.")
            continue
        for histo2 in region2:
            if histo2.hist_type == "2d":
                continue
            try:
                histo1 = region1.get(histo2.name)
            except KeyError:
                logger.warning(
                    f"Cannot find {process2.name}/{region2.name}/{histo2.name} in config1"
                )
            if histo1.systematic_band is None:
                histo1._systematic_band = histo2._systematic_band
            elif histo2.systematic_band is None:
                continue
            else:
                for band2 in histo2.systematic_band.values():
                    histo1.update_systematic_band(band2)
    return config1
