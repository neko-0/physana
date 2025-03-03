import logging
from . import metadata

log = logging.getLogger(__name__)


def add_regions(
    configMgr,
    name,
    truth_selection,
    reco_selection,
    matching_selection,
    notmatching_selection,
    weight_truth,
    weight_reco,
    isolation_selection=None,
    filter_region_hist_types=None,
    do_notmatch_subset=False,
    skip_notmatch_regions=False,
    **kwargs,
):
    """
    Makes 4 regions with {name}
      - particle: {name}_truth: base selection + truth selection
      - reco: {name}_reco: base selection + reco selection
      - reco_match: {name}_truth_reco_matching: base selection + truth selection + reco selection + matching selection
      - reco_match_not: {name}_truth_reco_notmatching: base selection + truth selection + reco selection + not(matching selection)

    filter_region_hist_types is a dictionary that contains four keys with associated list of type values:
      - truth
      - abcd
      - reco_match
      - reco_match_not

    for the different region study types. The values are just a list of histogram types to exclude from the region under study.
    """
    log.info(
        f"Defining four regions for unfolding in {name}: \n\t- particle={name}_truth\n\t- reco={name}_reco\n\t- reco_match={name}_truth_reco_matching\n\t- reco_match_not={name}_truth_reco_notmatching"
    )
    if metadata.has(configMgr, "regions", name):
        return False

    if isolation_selection is None:
        isolation_selection = []

    filter_region_hist_types = filter_region_hist_types or {}

    regions = {
        "particle": f"{name}_truth",
        "reco": f"{name}_reco",
        "reco_match": f"{name}_truth_reco_matching",
        # "reco_match_not": f"{name}_truth_reco_notmatching",
    }

    configMgr.add_region(
        regions['particle'],
        ' && '.join([*truth_selection]),
        weight=weight_truth,
        study_type="truth",
        filter_hist_types=filter_region_hist_types.get('truth'),
        **kwargs,
    ),
    configMgr.add_region(
        regions['reco'],
        ' && '.join([*reco_selection]),
        weight=weight_reco,
        study_type="abcd",
        filter_hist_types=filter_region_hist_types.get('abcd'),
        **kwargs,
    )
    configMgr.add_region(
        regions['reco_match'],
        ' && '.join(
            [
                *truth_selection,
                *reco_selection,
                *matching_selection,
            ]
            + isolation_selection
        ),
        study_type="reco_match",
        weight=weight_reco,
        filter_hist_types=filter_region_hist_types.get('reco_match'),
        **kwargs,
    )

    if not skip_notmatch_regions:
        regions["reco_match_not"] = f"{name}_truth_reco_notmatching"
    else:
        metadata.save(configMgr, "regions", name, regions)
        return None  # early termination

    # this is the OLD not match selection.
    if do_notmatch_subset:
        configMgr.add_region(
            regions['reco_match_not'],
            ' && '.join(
                [
                    *truth_selection,
                    *reco_selection,
                    *notmatching_selection,
                ]
                + isolation_selection
            ),
            study_type="reco_match_not",
            weight=weight_reco,
            filter_hist_types=filter_region_hist_types.get('reco_match_not'),
            **kwargs,
        )
    else:
        if isolation_selection:
            isolation_selection = ['&&'.join(isolation_selection)]
        # take out the 'isTruth' flag
        truth_only = [t for t in truth_selection if t != "isTruth"]
        # comp1: events pass reco and truth. NOT matched
        comp1 = [
            '&&'.join(reco_selection),
            'isTruth',
            '&&'.join(truth_only),
            # '&&'.join(notmatching_selection),
        ] + isolation_selection
        if notmatching_selection:
            comp1.append('&&'.join(notmatching_selection))
        # comp2: events pass reco but NOT the truth. matched.
        comp2 = [
            '&&'.join(reco_selection),
            'isTruth',
            f'!({"&&".join(truth_only)})',
            # '&&'.join(matching_selection),
        ] + isolation_selection
        if matching_selection:
            comp2.append('&&'.join(matching_selection))
        # comp3: events pass reco but NOT the truth. NOT matched.
        comp3 = [
            '&&'.join(reco_selection),
            'isTruth',
            f'!({"&&".join(truth_only)})',
            # '&&'.join(notmatching_selection),
        ] + isolation_selection
        if notmatching_selection:
            comp3.append('&&'.join(notmatching_selection))
        # comp4: events pass reco, but NO truth
        comp4 = [
            '&&'.join(reco_selection),
            'isTruth==0',
        ] + isolation_selection
        # comp_1 = f"{'&&'.join(reco_selection)} && isTruth && {'&&'.join(truth_only)} && {'&&'.join(notmatching_selection)}"
        # comp_2 = f"{'&&'.join(reco_selection)} && isTruth && !({'&&'.join(truth_only)}) && {'&&'.join(matching_selection)}"
        # comp_3 = f"{'&&'.join(reco_selection)} && isTruth && !({'&&'.join(truth_only)}) && {'&&'.join(notmatching_selection)}"
        # comp_4 = f"{'&&'.join(reco_selection)} && isTruth==0"
        reco_match_no_sel = f"({'&&'.join(comp1)}) || ({'&&'.join(comp2)}) || ({'&&'.join(comp3)}) || ({'&&'.join(comp4)})"
        configMgr.add_region(
            regions['reco_match_not'],
            reco_match_no_sel,
            study_type="reco_match_not",
            weight=weight_reco,
            filter_hist_types=filter_region_hist_types.get('reco_match_not'),
            **kwargs,
        )

    metadata.save(configMgr, "regions", name, regions)


def add_observables(
    configMgr, name, unfold_to, bins, xmin, xmax, xtitle, reco=None, **kwargs
):
    """
    Creates two 1-D histograms, and 1 2-D histogram.
    The name is used as key for lookup. One can have different lookup for the same reco-truth pair, i.e.

    >>> add_observables(config, 'jet1Pt', 'jet1TruthPt', 100, 0, 1000, 'leading jet pt')
    >>> add_observables(config, 'alt_jet1Pt', 'jet1TruthPt', 100, 0, 1000, 'leading jet pt', reco='jet1Pt')

    The first one will generate:
        reco: 'jet1Pt' with lookup key 'jet1Pt'
        truth: 'jet1TruthPt' with lookup key 'jet1TruthPt'
        response: lookup key 'response_matrix_jet1Pt'

    The second one will generate:
        reco: 'jet1Pt' with lookup key 'alt_jet1Pt'
        truth: 'jet1TruthPt' with lookup key 'truth_alt_jet1Pt'
        response: lookup key 'response_matrix_alt_jet1Pt'

    """
    log.info(
        f"Defining two observables reco={reco or name}, truth={unfold_to}; and a 2D response=response_matrix_{name}"
    )

    if metadata.has(configMgr, "observables", name):
        return False

    bins_args = (bins, xmin, xmax)

    observables = {
        "reco": name,
        "truth": f"truth_{name}" if reco else unfold_to,
        "response": f"response_matrix_{name}",
    }

    configMgr.add_observable(
        observables["reco"], *bins_args, xtitle, type="reco", observable=reco, **kwargs
    )
    configMgr.add_observable(
        observables["truth"],
        *bins_args,
        f"Particle level {xtitle}",
        type="truth",
        observable=unfold_to,
        **kwargs,
    )
    configMgr.add_histogram2D(
        observables["response"],  # name
        reco or observables["reco"],  # reco
        unfold_to,  # truth
        *bins_args,
        *bins_args,
        xtitle,
        f"Particle level {xtitle}",
        type="response",
        **kwargs,
    )

    metadata.save(configMgr, "observables", name, observables)
