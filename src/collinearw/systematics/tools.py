"""
All common logic for systematics calculation goes here
"""

# from ... import utils
from . import algorithm
import numpy as np
import time
import logging

from .core import SystematicsBase
from .syst_band import SystematicsBand
from ..histo import Histogram


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _get_alg(name):
    return getattr(algorithm, name, algorithm.passthru)


def run_auto_compute_systematics(configMgr, **kwargs):
    """
    Automatically executes the systematics processing.

    This will compute stuff.

    Args:
        configMgr (collinearw.configMgr.ConfigMgr): configuration object with all metadata and histograms needed

    Returns:
        nothing (for now)

    for process_set in configMgr.process_sets:
        print(f'ProcessSet: {process_set.name}')
        if not process_set.systematics:
            print(f'  has no systs')
            continue
        for sys_name, sys_set in configMgr.systematics.items():
            process_sys_set = process_set.get(sys_name)
            if not isinstance(process_sys_set, list):
                print(f'  skipping {sys_name}')
                continue
            sys_type = process_sys_set[0].systematic.sys_type
            algorithm = _get_alg(sys_type)
            print(f'  has {sys_name} of type {sys_type} using {algorithm}')
            computed = algorithm(process_sys_set)
            # do something with me plz
    """

    # getting systematic name and type
    sys_name_type = configMgr.list_systematic_type()

    for name, full_name, type, handle in sys_name_type:
        logger.info(f"Computing {type} {name} with {handle} method.")
        algorithm = _get_alg(handle)
        for pset in configMgr.process_sets:
            computed = algorithm(pset, name)
            if computed is not None:
                if not hasattr(pset, "computed_systematics"):
                    pset.computed_systematics = {}
                for key, data in computed.items():
                    pset.computed_systematics[(name, full_name, key)] = data


# ==============================================================================


def compute_process_set_systematics(process_set, sys_name, algorithm, *args, **kwargs):
    compute_method = _get_alg(algorithm)
    computed = compute_method(process_set, sys_name)
    if computed is not None:
        if not hasattr(process_set, "computed_systematics"):
            process_set.computed_systematics = {}
        for key, data in computed.items():
            process_set.computed_systematics[(sys_name, algorithm, key)] = data


def compute_systematics(
    configMgr,
    sys_name,
    algorithm,
    processes=[],
    *args,
    **kwargs,
):
    """
    compute sytematic with given generic/parent name and given algorithm
    """
    process_sets = kwargs.get('process_sets', configMgr.process_sets)

    # if processes are specified, only compute those.
    if processes:
        process_sets = [pset for pset in process_sets if pset.name in processes]

    for pset in process_sets:
        compute_process_set_systematics(pset, sys_name, algorithm, *args, **kwargs)


# ==============================================================================


def compute_quadrature_sum(
    config,
    process_name,
    name,
    syst_type,
    systematics,
    store_process=True,
    store_band=True,
    use_ratio=True,  # ratio is the correct method
    external_process_set=None,
    sub_band_name=None,
    regions=None,
    observables=None,
):
    """
    Compute sum in quadrature for given list of systematic within a process set object.
    This is a wrapper function that act on the `ConfigMgr` object.

    Args:
        config : ConfigMgr
        `ConfigMgr` object that contains the require processes and systematics.

        process_name: str
            name of the `core.ProcessSet` object within the provided 'config'

        name : str
            User given name to the sum.
            e.g. 'jet energy scale', 'b-tagging systematic', 'wjet-theory'

        syst_type : str
            User given type to the computed quadrature sum result.
            e.g. 'experimental', 'theory'

        use_ratio : bool, default = True
            Use ratio form in the quadrature sum, i.e. ratio_i = (var_i - norm)/norm
            then total_varation**2 = sum(ratio_i**2)

        store_band : bool, default = True
            Create SystematicsBand object and store it into the nominal process,
            with name of the band given by the 'name' argument.

        sub_band_name : str, default=None
            Name giving to the systematic sub-band. `name` will be used if
            it's not specified.

    Return:
        dictionary of SystematicsBand object.
        The key is specified by tuple of (region.name, observable.name)

    """

    # check the number of systematics, return None if empty
    n_syst = len(systematics)
    if n_syst == 0:
        return None

    regions = regions or config.region_list
    observables = observables or config.observable_list

    if external_process_set:
        # if external process set is provided, set store_process = False.
        store_process = False
        process_set = external_process_set
        process_name = process_set.name
        logger.info(f"external process set {process_name} is provided")
    else:
        process_set = config.get_process_set(process_name)

    logger.info(f"start computing quadratic sum for {process_name}")
    logger.info(
        f"number of regions and histograms ({len(regions)}, {len(observables)})"
    )
    logger.info(f"({name}, {syst_type}), number of sytematics = {n_syst}")
    logger.info(f"{store_process=}, {store_band=}, {use_ratio=}")

    if use_ratio:
        quad_sum = _get_alg("ratio_quadrature_sum")
    else:
        quad_sum = _get_alg("quadrature_sum")

    components_only = not store_process

    _perf_counter = time.perf_counter()
    quad_summed = quad_sum(
        process_set, systematics, regions, observables, components_only=components_only
    )
    logger.info(f"quadature sum -> {time.perf_counter()-_perf_counter}s W.T.")
    _perf_counter = time.perf_counter()

    if store_process:
        for side in ["up", "down"]:  # this is indeed the the uppoer and lower errors
            new_systematic = SystematicsBase(
                name, f"{name}-{syst_type}_quad_sum_{side}", "dummy", syst_type
            )
            lookup_name = new_systematic.full_name
            scaled_process = quad_summed["scaled"][side]
            scaled_process.systematic = new_systematic
            process_set.computed_systematics[lookup_name] = scaled_process

    syst_band_dict = {}
    sub_band_name = sub_band_name or name
    nominal = process_set.get()  # getting reference of the nominal process object
    components = quad_summed["components"]
    for region in nominal.regions:
        for obs in region.histograms:
            if not isinstance(obs, Histogram):
                continue
            # syst_type here is usually 'experimental' or 'theory'
            band = SystematicsBand(name, syst_type, obs.bin_content.shape)
            lookup_key = (region.name, obs.name)
            for side in ["up", "down"]:
                if components[side]:
                    sub_band = SystematicsBand(
                        sub_band_name, syst_type, obs.bin_content.shape
                    )
                    try:
                        my_components = components[side][lookup_key]
                    except KeyError:
                        continue
                    for comp_name, comp in my_components.items():
                        sub_band.add_component(side, comp_name, comp.bin_content)
                    band.update_sub_bands(sub_band)
            if store_band:
                obs.update_systematic_band(band)
            syst_band_dict[lookup_key] = band

    logger.info(f"preparing output -> {time.perf_counter()-_perf_counter}s W.T.")

    return syst_band_dict


def create_lumi_band(histogram, lumiunc=0.0, content_type="event"):
    if lumiunc == 0.0:
        return None
    syst_band = histogram.systematic_band
    if syst_band and "lumi" in histogram.systematic_band:
        lumi_band = syst_band["lumi"]
        if lumi_band.type == content_type:
            return None
        old_sigma = lumi_band.up
        if content_type == "event":  # from event -> xsec
            sigma = old_sigma / (1.0 + lumiunc)
        elif content_type == "xsec":  # from xsec -> event
            sigma = old_sigma * (1.0 + lumiunc)
        lumi_band.clear()
        lumi_band.type = content_type
    else:
        lumi_band = SystematicsBand("lumi", content_type, histogram.bin_content.shape)
        if content_type == "event":
            sigma = np.ones(histogram.bin_content.shape) * lumiunc
        elif content_type == "xsec":
            sigma = np.ones(histogram.bin_content.shape) * lumiunc / (1 + lumiunc)
        else:
            raise ValueError(f"cannot create lumi band of type {content_type}")
    lumi_band.add_component("up", "lumi_1_sigma", sigma)
    lumi_band.add_component("down", "lumi_1_sigma", sigma)
    histogram.update_systematic_band(lumi_band)
