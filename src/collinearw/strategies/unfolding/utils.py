import numpy as np

from ...histo import Histogram
from . import plot

def get_xsec_uncert(
    observable,
    lumi,
    lumiunc=0.0,
    integral_opts='width all',
    exclude_types=None,
    exclude_names=None,
):
    if not isinstance(observable, Histogram):
        return (None, None)

    observable = plot.scale_to_xsec(observable, lumi, lumiunc=lumiunc)

    syst_errors = None
    if observable.systematic_band is not None:
        rel_dn = observable.total_band(
            exclude_types=exclude_types,
            exclude_names=exclude_names,
            include_stats=False,
        ).down
        rel_up = observable.total_band(
            exclude_types=exclude_types,
            exclude_names=exclude_names,
            include_stats=False,
        ).up

        error_dn = rel_dn * observable.bin_content * observable.bin_width
        error_up = rel_up * observable.bin_content * observable.bin_width

        syst_errors = (np.sqrt(np.sum(error_dn**2)), np.sqrt(np.sum(error_up**2)))

        rel_tot_dn = observable.total_band(
            exclude_types=exclude_types, exclude_names=exclude_names, include_stats=True
        ).down
        rel_tot_up = observable.total_band(
            exclude_types=exclude_types, exclude_names=exclude_names, include_stats=True
        ).up

        error_tot_dn = rel_tot_dn * observable.bin_content * observable.bin_width
        error_tot_up = rel_tot_up * observable.bin_content * observable.bin_width

        tot_errors = (
            np.sqrt(np.sum(error_tot_dn**2)),
            np.sqrt(np.sum(error_tot_up**2)),
        )

    stat_errors = np.sqrt(
        np.sum(np.nan_to_num(observable.sumW2) * observable.bin_width**2)
    )
    return (observable.integral(integral_opts), stat_errors, syst_errors, tot_errors)


def merge_migration_process(norm_process, migration_process, copy=True):
    c_norm_process = norm_process.copy() if copy else norm_process
    for region in migration_process.regions:
        if region.type == "truth":
            continue
        norm_region = c_norm_process.get(region.name)
        for hist in region.histograms:
            if hist.hist_type == "1d" and hist.type == "truth":
                continue
            if hist.hist_type == "2d":
                m_hist = hist.collapse_to_underflow(axis=1)
            else:
                m_hist = hist.collapse_to_underflow()
            norm_region.get(hist.name).add(m_hist)
    return c_norm_process


def merge_migration(config, migration_IN, migration_OUT=None, systematic=False):
    """
    Merging phasespace migration histograms.
    All of the configs are assumed to have same naming for processes, regions,
    and histograms.

    migration_in_config :
        contains events that are outside of the reco phasespace selection but
        fall inside the truth phasespace selection.

    migration_out_config : default=None
        contatins events that are indside of the reco phasespace selection but
        fall outside of the truth phasespace selection.
    """
    if systematic:
        processes = (x for pset in migration_IN.process_sets for x in pset)
    else:
        processes = (x for x in migration_IN.processes)
    # collapsing to underflow bins
    for process in processes:
        if process.systematic is None:
            syst_name = None
        else:
            syst_name = process.systematic.full_name
        norm_process = config.get(process.name).get(syst_name)
        for region in process.regions:
            if region.type == "truth":  # no need to modefy any truth regions
                continue
            norm_region = norm_process.get(region.name)
            for hist in region.histograms:
                # skip 1D truth type histogram
                if hist.hist_type == "1d" and hist.type == "truth":
                    continue
                if hist.hist_type == "2d":
                    m_hist = hist.collapse_to_underflow(axis=1, inplace=True)
                else:
                    m_hist = hist.collapse_to_underflow(inplace=True)
                norm_region.get(hist.name).add(m_hist)

    if migration_OUT:
        r_type_pass_filter = set(["reco_match", "truth"])
        if systematic:
            processes = (x for pset in migration_OUT.process_sets for x in pset)
        else:
            processes = (x for x in migration_OUT.processes)
        for process in processes:
            if process.systematic is None:
                syst_name = None
            else:
                syst_name = process.systematic.full_name
            norm_process = config.get(process.name).get(syst_name)
            for region in process.regions:
                if region.type not in r_type_pass_filter:
                    continue
                norm_region = norm_process.get(region.name)
                for hist in region.histograms:
                    # check for response matrix only
                    if hist.hist_type == "2d" and region.type == "reco_match":
                        m_hist = hist.collapse_to_underflow(axis=0, inplace=True)
                    # check for 1D truth only
                    elif hist.hist_type == "1d" and hist.type == "truth":
                        m_hist = hist.collapse_to_underflow(inplace=True)
                    else:
                        # print(f"skipping migration out: {region.name}//{hist.name}")
                        continue
                    norm_region.get(hist.name).add(m_hist)

    return config
