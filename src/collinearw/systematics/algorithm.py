"""
algorithm for handling systematics
"""

import logging

import numpy as np

from .core import SystematicsBase
from ..histo.histo_alg import HistogramAlgorithm
from ..histo.cache import temporary_cache
from ..histo.histo1d import Histogram

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ==============================================================================
# ==============================================================================


def stdev(process_set, sys_name):
    """
    This method compute the standard deviation of the given systematics,
    and return mean +- 1 std.

    the return object will be instance of core.Process
    """
    plist = process_set.get(sys_name)
    if id(plist) == id(process_set.nominal):
        logger.warning(f"{process_set.name} with {sys_name} return nominal.")
        return None

    # make copy from the nominal process to store up/down std result
    std_up = process_set.nominal.copy()
    std_down = process_set.nominal.copy()
    std_up.systematic = SystematicsBase(
        sys_name, (sys_name, "stdev", "std_up"), "dummy", plist[0].systematic.sys_type
    )
    std_down.systematic = SystematicsBase(
        sys_name, (sys_name, "stdev", "std_down"), "dummy", plist[0].systematic.sys_type
    )
    for r_up, r_down in zip(std_up.regions, std_down.regions):
        up_results = []
        down_results = []
        for obs in r_up.list_observables():
            hist_buff = [p.get_region(r_up.name).get_observable(obs) for p in plist]
            std = HistogramAlgorithm.stdev(hist_buff)
            mean = HistogramAlgorithm.mean(hist_buff)
            up_results.append(mean + std)
            down_results.append(mean - std)
        r_up.clear()
        r_down.clear()
        for h_up, h_down in zip(up_results, down_results):
            r_up.add_histogram(h_up)
            r_down.add_histogram(h_down)
    return {"std_down": std_down, "std_up": std_up}


def hessian(process_set, sys_name):
    """
    Calculate Hessian PDF uncertainty.

    up = sqrt( sum( max( (+v)-n, (-v)-n, 0)^2 ) )
    down = sqrt( sum( max( n-(+v), n-(-v), 0)^2 ) )

    where (+v) and (-v) are the + and - direction of eigenvector of hessian matrix
    """

    plist = process_set.get(sys_name)  # list of processes under 'sys_name'
    if id(plist) == id(process_set.nominal):
        logger.warning(f"{process_set.name} with {sys_name} return nominal.")
        return None

    # make copy from the nominal process to store up/down std result
    sys_type = plist[0].systematic.sys_type
    up = process_set.nominal.copy()
    down = process_set.nominal.copy()
    up.systematic = SystematicsBase(
        sys_name, (sys_name, "hessian", "up"), "dummy", sys_type
    )
    down.systematic = SystematicsBase(
        sys_name, (sys_name, "hessian", "down"), "dummy", sys_type
    )
    for r_up, r_down in zip(up.regions, down.regions):
        up_results = []
        down_results = []
        nominal_region = process_set.nominal.get_region(r_up.name)
        with temporary_cache(nominal_region):
            for obs in r_up.list_observables():
                nominal_obs = nominal_region.get_observable(obs)
                content_buffer = [
                    p.get_region(r_up.name).get_observable(obs).bin_content
                    for p in plist
                ]
                pos_calc = lambda x: np.maximum(x - nominal_obs.bin_content, 0) ** 2
                neg_calc = lambda x: np.maximum(nominal_obs.bin_content - x, 0) ** 2
                up_band = np.sum(list(map(pos_calc, content_buffer)), axis=0)
                down_band = np.sum(list(map(neg_calc, content_buffer)), axis=0)
                up_histo = nominal_obs.copy(shallow=True)
                down_histo = nominal_obs.copy(shallow=True)
                up_histo.bin_content = nominal_obs.bin_content + np.sqrt(up_band)
                down_histo.bin_content = nominal_obs.bin_content - np.sqrt(down_band)
                up_results.append(up_histo)
                down_results.append(down_histo)
        r_up.clear()
        r_down.clear()
        for h_up, h_down in zip(up_results, down_results):
            r_up.add_histogram(h_up)
            r_down.add_histogram(h_down)
    return {"down": down, "up": up}


def mean(process_set, sys_name):
    plist = process_set.get(sys_name)
    if id(plist) == id(process_set.nominal):
        return None
    c_nominal = process_set.nominal.copy()
    for r in c_nominal.regions:
        results = []
        for obs in r.list_observables():
            hist_buff = [p.get_region(r.name).get_observable(obs) for p in plist]
            results.append(HistogramAlgorithm.mean(hist_buff))
        r.clear()
        for h in results:
            r.add_histogram(h)
    return {"mean": c_nominal}


def min_max(process_set, sys_name):
    plist = process_set.get(sys_name)
    if id(plist) == id(process_set.nominal):
        return None
    max_p = process_set.nominal.copy()
    min_p = process_set.nominal.copy()
    max_p.systematic = SystematicsBase(
        sys_name, (sys_name, "min_max", "max"), "dummy", plist[0].systematic.sys_type
    )
    min_p.systematic = SystematicsBase(
        sys_name, (sys_name, "min_max", "min"), "dummy", plist[0].systematic.sys_type
    )
    for r in max_p.regions:
        results = []
        rlist = [p.get_region(r.name) for p in plist]
        for obs in r.list_observables():
            hist_buff = [_r.get_observable(obs) for _r in rlist]
            results.append(HistogramAlgorithm.min_max(hist_buff))
        r.clear()
        min_p.get_region(r.name).clear()
        for max_h, min_h in results:
            r.add_histogram(max_h)
            min_p.get_region(r.name).add_histogram(min_h)
    return {"min": min_p, "max": max_p}


def symmetrize_up_down(process_set, sys_name):
    """
    symetrizing the error when the content of the up and down variation are the same
    """
    plist = process_set.get(sys_name)
    nominal = process_set.nominal
    if id(plist) == id(nominal):
        return None
    assert len(plist) == 2  # only have up and down
    up_p, down_p = plist
    # average = up_p + down_p
    # average.div(2.0)
    # diff = average - nominal
    up_diff = up_p - nominal
    dn_diff = down_p - nominal
    sym_up = nominal.copy()
    sym_dn = nominal.copy()
    # take abs values
    for nom_r in nominal.regions:
        norm_r_name = nom_r.name
        # diff_r = diff.get(norm_r_name)
        up_diff_r = up_diff.get(norm_r_name)
        dn_diff_r = dn_diff.get(norm_r_name)
        sym_up_r = sym_up.get(norm_r_name)
        sym_dn_r = sym_dn.get(norm_r_name)
        for nom_h in nom_r.histograms:
            nom_h_name = nom_h.name
            # diff_h = diff_r.get(nom_h_name)
            up_diff_h = up_diff_r.get(nom_h_name)
            dn_diff_h = dn_diff_r.get(nom_h_name)
            sym_up_h = sym_up_r.get(nom_h_name)
            sym_dn_h = sym_dn_r.get(nom_h_name)
            # np.abs(diff_h.bin_content, out=diff_h.bin_content)
            # sym_up_h.bin_content += diff_h.bin_content
            # sym_dn_h.bin_content -= diff_h.bin_content
            np.abs(up_diff_h.bin_content, out=up_diff_h.bin_content)
            np.abs(dn_diff_h.bin_content, out=dn_diff_h.bin_content)
            sym_up_h.bin_content += up_diff_h.bin_content
            sym_dn_h.bin_content -= dn_diff_h.bin_content

    sym_up.systematic = SystematicsBase(
        sys_name,
        (sys_name, "symmetrize", "up"),
        "dummy",
        handle=up_p.systematic.sys_type,
        symmetrize=False,  # don't need symmetrization here.
    )

    sym_dn.systematic = SystematicsBase(
        sys_name,
        (sys_name, "symmetrize", "down"),
        "dummy",
        handle=down_p.systematic.sys_type,
        symmetrize=False,
    )

    return {"symmetrize_up": sym_up, "symmetrize_down": sym_dn}


def passthru(process_set, sys_name):
    return (process_set.nominal,)


def quadrature_sum(
    process_set, systematics, regions=[], observables=[], *, nominal=None
):
    # check if list of regions and observables are provided.
    # try to get from process set if not provided.
    if not regions:
        regions = process_set[0].list_regions()
    if not observables:
        observables = process_set[0][0].list_observables()

    nominal = nominal or process_set.get()
    scaled_process = nominal.copy()
    band_process = nominal.copy()
    components = {}
    for region in regions:
        try:
            nominal_region = nominal.get_region(region)
        except KeyError:
            continue
        scaled_region = scaled_process.get_region(region)
        scaled_region.clear()
        band_region = band_process.get_region(region)
        band_region.clear()
        for obs in observables:
            try:
                nominal_obs = nominal_region.get_observable(obs)
            except KeyError:
                continue
            if not isinstance(nominal_obs, Histogram):
                continue
            # computing the quadrature sum of ratio difference
            scaled_observables = []
            tot_scaled = None
            syst_name = None
            nominal_syst_diff = {}
            for syst in systematics:
                syst_process = process_set.get(syst)
                if syst_process.systematic is None:
                    continue
                else:
                    syst_region = syst_process.get_region(region)
                syst_obs = syst_region.get_observable(obs)
                m_diff = abs(syst_obs - nominal_obs)
                nominal_syst_diff[syst] = m_diff
                scaled_observables.append(m_diff)
                if syst_name is None:
                    syst_name = syst_process.systematic.name
            if not scaled_observables:
                continue
            for scaled_obs in scaled_observables:
                if tot_scaled is None:
                    tot_scaled = scaled_obs * scaled_obs
                else:
                    tot_scaled += scaled_obs * scaled_obs
            tot_scaled.bin_content = np.sqrt(tot_scaled.bin_content)
            tot_band = abs(tot_scaled / nominal_obs - 1)
            scaled_region.add_histogram(tot_scaled)
            band_region.add_histogram(tot_band)
            components[(region, obs)] = nominal_syst_diff

    return {"scaled": scaled_process, "band": band_process, "components": components}


def ratio_quadrature_sum(
    process_set,
    systematics,
    regions=None,
    observables=None,
    *,
    nominal=None,
    components_only=False,
):
    """
    Compute sum in quadrature for given list of systematic within a process set object.
    Using ratio form in the sum. i.e.
        ratio_i = (var_i - norm)/norm,
        total_varation**2 = sum(ratio_i**2)

    Args:
        process_set: core.ProcessSet
            A `ProcessSet` object that contains nominal and requried systematics
            for the sum.

        regions : list(str), default = []
            List of regions to be involved for the sum. If not provided, all regions
            within the nominal process of the ProcessSet object will be used.

        observables : list(str), default = []
            Similar to regions. If not provided, all observables from first region
            of the nominal process will be used.

        nominal : core.Process, default = None
            Use external nominal `core.Process` object if provided.

        components_only: bool, default = True
            Return only the components, i.e. no scaled process and band is return
    """

    # check if list of regions and observables are provided.
    # try to get from process set if not provided.
    if regions is None:
        regions = process_set[0].list_regions()
    if observables is None:
        observables = process_set[0][0].list_observables()

    nominal = nominal or process_set.get()

    if not components_only:
        scaled_process = {"up": nominal.copy(), "down": nominal.copy()}
        band_process = {"up": nominal.copy(), "down": nominal.copy()}
    else:
        scaled_process = None
        band_process = None

    components = {"up": {}, "down": {}}

    with temporary_cache(process_set), temporary_cache(nominal):
        for region in regions:
            try:
                nominal_region = nominal.get_region(region)
            except KeyError:
                continue

            scaled_region = {}
            band_region = {}
            for side in ["up", "down"]:
                if scaled_process:
                    scaled_region[side] = scaled_process[side].get_region(region)
                    scaled_region[side].clear()
                if band_process:
                    band_region[side] = band_process[side].get_region(region)
                    band_region[side].clear()

            with temporary_cache(nominal_region):
                for obs in observables:
                    try:
                        nominal_obs = nominal_region.get_observable(obs)
                    except KeyError:
                        continue
                    # computing the quadrature sum of ratio difference
                    ratios = {"up": {}, "down": {}}
                    tot_band = {"up": None, "down": None}

                    for syst in systematics:
                        syst_process = process_set.get(syst)
                        if syst_process.systematic is None:
                            continue
                        try:
                            syst_region = syst_process.get_region(region)
                            syst_obs = syst_region.get_observable(obs)
                        except KeyError as _error:
                            logger.warning(_error)
                            logger.warning(f"skipping {syst} for {region}/{obs}")
                            break
                        # note: tot_band = sqrt( sum( (syst-nominal)/nominal)**2 ) )
                        # up_ratio = syst_obs - nominal_obs
                        # up_ratio.div(nominal_obs) # don't need error propagation
                        # up_ratio = syst_obs.copy()
                        up_ratio = syst_obs.copy(shallow=True)
                        if not np.all(up_ratio.bin_content == 0):
                            up_ratio.bin_content -= nominal_obs.bin_content
                            up_ratio.bin_content /= nominal_obs.bin_content
                        up_ratio.remove_negative_bin()
                        up_ratio.nan_to_num()
                        ratios["up"][syst] = up_ratio
                        # check if the systematic is symmetrized
                        if syst_process.systematic.symmetrize:
                            ratios["down"][syst] = up_ratio
                            continue
                        # down_ratio = nominal_obs - syst_obs
                        # down_ratio.div(nominal_obs)
                        # down_ratio = syst_obs.copy()
                        down_ratio = nominal_obs.copy(shallow=True)
                        if not np.all(syst_obs.bin_content == 0):
                            down_ratio.bin_content -= syst_obs.bin_content
                            down_ratio.bin_content /= nominal_obs.bin_content
                        else:
                            down_ratio.bin_content[:] = 0.0
                        down_ratio.remove_negative_bin()
                        down_ratio.nan_to_num()
                        ratios["down"][syst] = down_ratio

                    for side in ["up", "down"]:
                        if not ratios[side]:
                            continue
                        components[side][(region, obs)] = ratios[side]
                        for ratio in ratios[side].values():
                            if tot_band[side] is None:
                                # tot_band[side] = ratio * ratio
                                tot_band[side] = ratio.copy(shallow=True)
                                tot_band[side].bin_content **= 2
                            else:
                                # tot_band[side].add(ratio * ratio)
                                tot_band[side].bin_content += ratio.bin_content**2
                        if tot_band[side] is None:
                            continue
                        else:
                            if scaled_region or band_region:
                                tot_band[side].bin_content = np.sqrt(
                                    tot_band[side].bin_content
                                )
                                if band_region:
                                    band_region[side].add_histogram(tot_band[side])
                                if scaled_region:
                                    scaled_nominal = tot_band[side] * nominal_obs
                                    scaled_region[side].add_histogram(scaled_nominal)

    return {"scaled": scaled_process, "band": band_process, "components": components}
