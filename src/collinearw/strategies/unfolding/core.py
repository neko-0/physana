"""
All common logic for unfolding goes here
"""

import warnings
import logging
import pathlib
import collections
import tqdm

# import formulate
import numpy as np
from .. import abcd
from ... import utils
from ... import histManipulate
from ...backends import RootBackend
from ...histo.histo_alg import iter_histograms
from . import metadata
from . import plot

try:
    import ROOT
except ImportError:
    warnings.warn("Cannot import ROOT!! Unfolding will NOT run properly!")

try:
    from RooUnfold import RooUnfoldResponse, RooUnfoldBayes
except ImportError:
    try:
        warnings.warn("trying to import RooUnfold via ROOT.")
        from ROOT import RooUnfoldResponse, RooUnfoldBayes
    except ImportError:
        warnings.warn("Cannot find RooUnfold.")
        RooUnfoldResponse = None
        RooUnfoldBayes = None

log = logging.getLogger(__name__)


def closure_test_with_priors(config, signal, priors, **kwargs):
    """
    JUST FOR TESTING

    Run closure test with nominal signal and different prior in the
    Baysian iterative method. Only MC signal is assumed here.

    Args:
        config (collinearw.configMgr.ConfigMgr):
            instance of ConfigMgr

        signal (str) :
            name of the nominal signal process within the config.
            Resonpse matrix of the signal process will be used throughout the test.

        priors (list(str)):
            list of process names for truth distributions. These will be
            used as different prior in the Baysian iterative method.
    """
    abcd_tags = abcd.get_abcd_tags(config)
    abcd_signal_region_name = abcd.abcd_signal_region_name

    signal_process = config.get(signal).get()  # just get nominal
    prior_processes = [config.get(prior).get() for prior in priors]

    kwargs.setdefault("subtract_notmatch", False)  # let RooUnfold to handle notmatch
    kwargs.setdefault("overflow", True)  # enable overflow bins
    kwargs.setdefault("correct_hMeas", False)  # pass in raw hMeas

    for prior in prior_processes:
        log.info(f"closure test with prior {prior.name}")
        for abcd_tag in abcd_tags:
            unfolded_process = run(
                config,
                prior,  # unfoldable
                signal_process,  # measured, topor reco distribution
                signal_process,  # truth, which use as prior
                signal_process,  # response matrix
                f"{signal}_prior_{prior.name}_{abcd_tag}",
                output_folder="closure_test_prior",
                regionRename=lambda x: abcd_signal_region_name(
                    config, x, abcd_tag
                ),  # the default map of 'reco' is not the same as ABCD, maybe we can fix it?
                **kwargs,
            )
            config.append_process(unfolded_process, mode='merge', copy=False)
    return config


def run_auto(
    configMgr,
    systematic_names=None,
    include_fakes=True,
    disable_pbar=False,
    validate_subtraction=True,
    subtract_background=True,
    old_style_closure=False,
    truth_prior=None,
    signal_meas=None,
    signal_res=None,
    smoother=None,
    smooth_signal=True,
    **kwargs,
):
    """
    Automatically executes the unfolding procedure for closure and the real thing.

    This will:

      - perform the subtraction of fakes and background from data
      - make repeated calls to :func:`run` for each signal process to do the unfolding

    Args:
        configMgr (collinearw.configMgr.ConfigMgr):
            configuration object with all metadata and histograms needed
        systematic_names (None or list of systematic names):
            systematics to run over.
            Specify [None] for nominal only. By default, process all systematics.
        validate_subtraction (bool) :
            validate the not-matched subtraction by performing same procedure with MC.
            i.e. generate unfolding with wjets(reco) - wjets(not-matched). This should be
            the same results as closure done with wjets(reco_match) if the wjets(not-matched)
            is subtracted properly.

    Returns:
        copied and modified configMgr
    """
    process_sets = configMgr.process_sets
    signals = [x for x in process_sets if x.process_type == 'signal']
    bkgs = [x for x in process_sets if x.process_type == 'bkg']
    fakes = [x for x in process_sets if x.process_type == 'fakes']
    data = [x for x in process_sets if x.process_type == 'data']
    alternative_signals = [x for x in process_sets if x.process_type == 'signal_alt']
    assert len(data) == 1
    if include_fakes:
        assert len(fakes) == 1
    data = data[0]

    # by default, run over all systematics
    if systematic_names is None:
        systematic_names = configMgr.list_all_systematics()

    # truth prior use for replacing alternative signal truth prior
    if truth_prior is not None and isinstance(truth_prior, str):
        truth_prior = configMgr.get_process(truth_prior)

    # For replacing alternative signal measure
    if signal_meas is not None and isinstance(signal_meas, str):
        signal_meas = configMgr.get_process(signal_meas)

    # response replacement for alternative
    if signal_res is not None and isinstance(signal_res, str):
        signal_res = configMgr.get_process(signal_res)

    if subtract_background:
        # Subtracted backgrounds
        # Subtract data - fakes - all mc backgrounds except mc signal
        log.info(f"Subtracting from data: {bkgs + fakes}")
        subtracted_data_name = f"{data.name}_subtracted_bkg_fakes"
        skip_processes = [p.name for p in process_sets if p not in bkgs + fakes]
        if smoother is None:
            subtracted_data_configMgr = histManipulate.Subtract_MC(
                configMgr,
                data_name=data.name,
                rename=subtracted_data_name,
                skip_processes=skip_processes,
                systematics=systematic_names,
            )
        else:
            # if smoother is provided, first sum all backgrounds, them perform smoothing
            sum_config = histManipulate.Sum_MC(
                configMgr,
                rename="sum_mc",
                skip_processes=skip_processes + [data.name],
                systematics=systematic_names,
            )
            sum_pset = sum_config.get_process_set("sum_mc")
            for h in iter_histograms(sum_pset):
                smoother(h)
            subtracted_data_configMgr = histManipulate.Subtract_MC(
                sum_config,
                data_name=data.name,
                rename=subtracted_data_name,
                skip_processes=skip_processes,
                systematics=systematic_names,
                list_processes=["sum_mc", data.name],
            )
            if smooth_signal:
                # create copy of signals, and apply same smoothing rule
                signals = [x.copy() for x in signals]
                for s in signals:
                    for h in iter_histograms(s):
                        smoother(h)
        subtracted_data = subtracted_data_configMgr.get_process_set(
            subtracted_data_name
        )
    else:
        # Unfolding can handle case with background included
        # Just get the data process
        subtracted_data = configMgr.get_process_set(data.name).copy()

    """
    for measured, truth, response, name in [
        ('wjets', 'wjets', 'wjets', 'closure'),
        ('subtracted_data', 'wjets', 'wjets', 'realthang'),
        ('subtracted_data', 'wjets', 'wjets_Sherpa', 'sanitycheck'),
    ]:
       run(configMgr, measured, truth, response, name, **kwargs)
    """
    get_abcd_tags = abcd.get_abcd_tags
    abcd_signal_region_name = abcd.abcd_signal_region_name

    abcd_tags = get_abcd_tags(subtracted_data_configMgr)
    tqdm_toplevel = tqdm.tqdm(
        ncols=100,
        total=(len(signals) + len(alternative_signals)) * (len(abcd_tags) + 1),
        position=0,
        leave=True,
        disable=disable_pbar,
    )
    for signal in signals:
        tqdm_toplevel.set_description(f"Signal = {signal.name}")
        if old_style_closure:
            # this closure is restrict to reco matched regions.
            # also it will cause problem if 1D histograms are previously
            # filtered in the reco_match regions.
            # a better closure approach is the validate_closure_ below.
            _process = run(
                subtracted_data_configMgr,
                signal.get(None),  # unfoldable
                signal.get(None),
                signal.get(None),
                signal.get(None),
                'closure',
                measured_region="reco_match",
                additionalRegions=['reco_match'],
                disable_pbar=disable_pbar,
                **kwargs,
            )
            # merge the returned unfold process into the config.
            # no need to copy since we don't use the return process anywhere.
            subtracted_data_configMgr.append_process(_process, mode="merge", copy=False)
        tqdm_toplevel.update()

        for abcd_tag in abcd_tags:
            if validate_subtraction:
                _process = run(
                    subtracted_data_configMgr,
                    signal.get(None),  # unfoldable
                    signal_meas or signal.get(None),
                    truth_prior or signal.get(None),
                    signal_res or signal.get(None),
                    f'validate_closure_{abcd_tag}',
                    regionRename=lambda x: abcd_signal_region_name(
                        subtracted_data_configMgr, x, abcd_tag
                    ),  # the default map of 'reco' is not the same as ABCD, maybe we can fix it?
                    disable_pbar=disable_pbar,
                    **kwargs,
                )
                subtracted_data_configMgr.append_process(
                    _process, mode="merge", copy=False
                )

            tqdm_toplevel.set_description(
                f"Signal = {signal.name}, abcd_tag = {abcd_tag}"
            )
            tqdm_systematic = tqdm.tqdm(
                ncols=100,
                total=len(systematic_names),
                position=1,
                leave=False,
                disable=disable_pbar,
            )
            for systematic_name in systematic_names:
                # check if systematic return nominal
                if systematic_name:
                    _syst_id = id(subtracted_data.get(systematic_name))
                    _norm_id = id(subtracted_data.get())
                    if _syst_id == _norm_id:
                        tqdm_systematic.update()
                        continue
                tqdm_systematic.set_description(f"Syst: {systematic_name or 'nominal'}")
                systematic = subtracted_data_configMgr.get_systematic(systematic_name)
                _process = run(
                    subtracted_data_configMgr,
                    subtracted_data.get(systematic_name),  # unfoldable
                    signal.get(systematic_name),
                    signal.get(systematic_name),
                    signal.get(
                        systematic_name
                        if systematic and systematic.sys_type != 'theory'
                        else None
                    ),  # unfold using nominal response matrix for theory systematics
                    f'realthang_{abcd_tag}',
                    regionRename=lambda x: abcd_signal_region_name(
                        subtracted_data_configMgr, x, abcd_tag
                    ),
                    additionalTruths=[
                        s.get(systematic_name)
                        for s in signals + alternative_signals
                        if s != signal
                    ],
                    disable_pbar=disable_pbar,
                    # saveYoda=True
                    # if 'electron' in abcd_tag and systematic_name is None
                    # else False,
                    **kwargs,
                )
                subtracted_data_configMgr.append_process(
                    _process, mode="merge", copy=False
                )
                tqdm_systematic.update()
            tqdm_toplevel.update()

    for signal in alternative_signals:
        tqdm_toplevel.set_description(f"Signal = {signal.name}")
        if old_style_closure:
            _process = run(
                subtracted_data_configMgr,
                signal.get(None),  # unfoldable
                signal_meas or signal.get(None),  # measured
                truth_prior or signal.get(None),  # truth
                signal_res or signal.get(None),  # response
                f'alt_closure_{signal.name}',
                measured_region="reco_match",
                additionalRegions=['reco_match'],
                disable_pbar=disable_pbar,
                **kwargs,
            )
            subtracted_data_configMgr.append_process(_process, mode="merge", copy=False)
        tqdm_toplevel.update()
        for abcd_tag in abcd_tags:
            if validate_subtraction:
                _process = run(
                    subtracted_data_configMgr,
                    signal.get(None),  # unfoldable
                    signal_meas or signal.get(None),  # measured
                    truth_prior or signal.get(None),  # truth
                    signal_res or signal.get(None),  # response
                    f'alt_validate_closure_{signal.name}_{abcd_tag}',
                    regionRename=lambda x: abcd_signal_region_name(
                        subtracted_data_configMgr, x, abcd_tag
                    ),  # the default map of 'reco' is not the same as ABCD, maybe we can fix it?
                    disable_pbar=disable_pbar,
                    **kwargs,
                )
                subtracted_data_configMgr.append_process(
                    _process, mode="merge", copy=False
                )
            tqdm_toplevel.set_description(
                f"Signal = {signal.name}, abcd_tag = {abcd_tag}"
            )
            _process = run(
                subtracted_data_configMgr,
                subtracted_data.get(None),  # unfoldable
                signal_meas or signal.get(None),  # measured
                truth_prior or signal.get(None),  # truth
                signal_res or signal.get(None),  # response
                f'alt_realthang_{abcd_tag}_{signal.name}',
                regionRename=lambda x: abcd_signal_region_name(
                    subtracted_data_configMgr, x, abcd_tag
                ),
                additionalTruths=[
                    s.get(None) for s in signals + alternative_signals if s != signal
                ],
                disable_pbar=disable_pbar,
                **kwargs,
            )
            subtracted_data_configMgr.append_process(_process, mode="merge", copy=False)
            tqdm_toplevel.update()
    return subtracted_data_configMgr


def run(
    configMgr,
    unfoldable,
    measured,
    truth,
    response,
    name,
    max_n_unfolds=5,
    measured_region="reco",
    regionRename=lambda x: x,
    lumi=-1.0,
    additionalTruths=[],
    additionalRegions=[],
    saveYoda=False,
    output_folder="unfolded",
    debug=False,
    disable_pbar=False,
    overflow=True,
    handleFakes=True,
    smoothit=False,
    unfold_err=2,
    correct_hMeas=False,
    subtract_notmatch=False,
    notmatch_overflow_only=False,
    store_vfakes=False,
    response_only=False,
    use_response_truth=False,
    save_all_iteration=True,
    exact_niter=False,
):
    """
    Perform the unfolding procedure using information in the configMgr.meta_data['unfold'].

      - hMeas always corresponds to the 'reco' histogram (1D)
      - hTrue always corresponds to the 'truth' histogram (1D)
      - hResponse always corresponds to the 'response' histogram (2D)

    This performs the unfolding by iterating over the number of unfolds up to
    the maximum number of unfolds and the observables to measure.

    Option for handling not-match events:
    1)  Using the RooUnfold library. The not-match components will be computed
        atomatically during the initilization and setup of `RooUnfoldResponse`,
        and the not-match components are stored in a 1D array named 'fakes'.
        If Baysian iterative method is used, the not-match, or 'fakes', will
        be assigned to an additional row of bins to the 2D N_ji mapping of
        cause to effects, which eseentially the response matrix passed via
        `RooUnfoldResponse`. Addition bin is also added to the 1D truth nCi,
        or the truth  distribution, and the sum of fakes is stored.
        To let RooUnfold handle the not-match, use the following flags:

            subtract_notmatch = False
            correct_hMeas = False
            handleFakes = True

    2)  The not-match components can be handle manually, but this requires the
        pre-fill the not-match histograms. It should work automatically if the
        config is built correctly. The manual not-match subtraction is done
        prior the `RooUnfoldResponse`, and the following flags are required:

            subtract_notmatch = True
            correct_hMeas = True
            handleFakes = False

    Args:
        configMgr (collinearw.configMgr.ConfigMgr): configuration object with all metadata and histograms needed
        measured (collinearw.core.Process): process we want to measure (used for hMeas)
        truth (collinearw.core.Process): process we want to unfold on to (used for hMeas, hTrue). This is used a prior in the Bayesian interation.
        response (collinearw.core.Process): process to use for the response matrix (used for hResponse)
        max_n_unfolds (int): maximum number of unfoldings to do
        measured_region (str): the region of the measured process to use. defaults to 'reco'.
        lumi (float): luminosity of dataset being unfolded
        additionalTruths (list of collinearw.core.Process): additional truth processes to overlay
        additionalRegions (list of str): additional regions to overlay from response process, choose from particle, reco, reco_match, reco_match_not
        saveYoda (bool): Whether to save yoda files or not
        output_folder (str): Name of the output folder to put all unfolding results
        debug (bool): Whether to make debugging plots or not

    Returns:
        nothing
    """
    log.info(
        f"Unfolding {name}: {measured.name} on to {truth.name} via response {response.name}"
    )

    syst_name = (
        "_".join(unfoldable.systematic.full_name)
        if unfoldable.systematic
        else "nominal"
    )

    # NB: return unfolded_process out
    unfolded_process = unfoldable.copy()
    unfolded_process.clear_content()
    unfolded_process.name = f'unfold_{name}'
    unfolded_process.process_type = 'unfolded'
    # do not append_process, return it
    # configMgr.append_processs(unfolded_process)
    metadata.save(configMgr, "processes", name, {"process": unfolded_process.name})

    regions = metadata.regions(configMgr)
    observables = metadata.observables(configMgr)

    # only want to keep particle-level regions in the unfolded process
    # to match with the truth MC regions we will overlay later
    # rely on ABCD tag to tell us if we keep region or not
    _truth_regions = [
        region['particle']
        for region in regions.values()
        if regionRename(region[measured_region]) is not None
    ]
    unfolded_process.regions = [
        region for region in unfolded_process.regions if region.name in _truth_regions
    ]

    _truth_histograms = [
        # formulate.from_auto(config['truth']).to_numexpr()
        config['truth']
        for _, config in observables.items()
    ]
    for region in unfolded_process.regions:
        region.histograms = [
            histogram
            for histogram in region.histograms
            if histogram.name in _truth_histograms
        ]

    tqdm_regions = tqdm.tqdm(
        total=len(regions), ncols=100, leave=False, position=2, disable=disable_pbar
    )

    yoda_data = collections.defaultdict(list)

    for unfold_name, region in regions.items():
        measured_region_name = regionRename(region[measured_region])
        # skip if there's no correspond region (e.g. closure is identity
        # function, but abcd tags will rename if it is a valid abcd region)
        if not measured_region_name:
            continue

        tqdm_regions.set_description(f"Region: {unfold_name}")
        tqdm_observables = tqdm.tqdm(
            total=len(observables),
            ncols=100,
            leave=False,
            position=3,
            disable=disable_pbar,
        )
        for observable, histogram in observables.items():
            # histogram_reco = formulate.from_auto(histogram['reco']).to_numexpr()
            histogram_reco = histogram['reco']

            tqdm_observables.set_description(f"Observable: {observable}")

            # hUnfoldable is what we unfold
            hUnfoldable = unfoldable.get(measured_region_name).get(histogram_reco)

            # if there's no pre-built not-match region, use the the response
            # matrix from match region to calculate the amount of non-match
            # by hMeasured_Reco - hResponse.Project_to_Reco
            try:
                hNotmatch = measured.get(region["reco_match_not"]).get(histogram_reco)
            except KeyError:
                log.warning("No pre-built not-match region.")
                res_ = response.get(region["reco_match"]).get(histogram['response'])
                mes_ = measured.get(measured_region_name).get(histogram_reco)
                hNotmatch = mes_ - res_.project_x()

            # if using a reco region, need to subtract the portion of the truth
            # process which does not match to reco to account for "fakes" in the
            # matching procedure. e.g. reco = reco_match + reco_match_not
            if measured_region == "reco" and subtract_notmatch:
                if notmatch_overflow_only:
                    hUnfoldable = hUnfoldable.copy()
                    hUnfoldable[0] -= hNotmatch[0]  # underflow
                    hUnfoldable[-1] -= hNotmatch[-1]  # overflow
                else:
                    hUnfoldable = hUnfoldable - hNotmatch
            hUnfoldable = hUnfoldable.root
            RootBackend.apply_process_styles(hUnfoldable, unfoldable)

            # NOTE: the measured_region_name is actually mapped by region dict
            # from above. So in the case of measured_region="reco_match", the
            # measured_region_name is pointed to "reco_match"
            # the relavent line from above is:
            # measured_region_name = regionRename(region[measured_region])
            hMeas = measured.get(measured_region_name).get(histogram_reco)
            if measured_region == "reco" and correct_hMeas:
                if notmatch_overflow_only:
                    hMeas = hMeas.copy()
                    hMeas[0] -= hNotmatch[0]  # underflow
                    hMeas[-1] -= hNotmatch[-1]  # overflow
                else:
                    hMeas = hMeas - hNotmatch
            hMeas = hMeas.root
            RootBackend.apply_process_styles(hMeas, measured)

            # hist_truth = formulate.from_auto(histogram['truth']).to_numexpr()
            hist_truth = histogram['truth']
            hTrue = truth.get(region["particle"]).get(hist_truth).root
            RootBackend.apply_process_styles(hTrue, truth)

            # histogram_response = '_'.join(
            #     map(
            #         lambda x: formulate.from_auto(x).to_numexpr(),
            #         histogram['response'].split('_'),
            #     )
            # )
            #
            # hResponse = (
            #     response.get_region(region["reco_match"])
            #     .get_histogram(histogram_response)
            #     .root
            # )
            #
            # try:
            #     hResponse = (
            #         response.get_region(region["reco_match"])
            #         .get_histogram(histogram['response'])
            #         .root
            #     )
            # except:
            #     breakpoint()
            hResponse = (
                response.get_region(region["reco_match"])
                .get_histogram(histogram['response'])
                .root
            )
            RootBackend.apply_process_styles(hResponse, response)

            # hist_truth = formulate.from_auto(histogram['truth']).to_numexpr()
            hist_truth = histogram['truth']
            hAdditionalTruths = [
                additionalTruth.get_region(region["particle"])
                .get_histogram(hist_truth)
                .root
                for additionalTruth in additionalTruths
            ]
            for hAdditionalTruth, additionalTruth in zip(
                hAdditionalTruths, additionalTruths
            ):
                RootBackend.apply_process_styles(hAdditionalTruth, additionalTruth)

            hAdditionalRegions = [
                response.get_region(region[additionalRegion])
                .get_histogram(hist_truth)
                .root
                for additionalRegion in additionalRegions
            ]
            for hAdditionalRegion in hAdditionalRegions:
                RootBackend.apply_process_styles(hAdditionalRegion, response)
                RootBackend.apply_styles(hAdditionalRegion, color=920)  # kGray

            base_output_path = (
                pathlib.Path(configMgr.out_path)
                .joinpath(
                    "plots",
                    output_folder,
                    name,
                    syst_name,
                    measured_region_name,
                )
                .resolve()
            )

            # NB: SHOULD ALL BE FROM SAME PROCESS
            if response_only:
                hMeas.Reset()
                hTrue.Reset()
            if use_response_truth:
                hTrue.Reset()
            unfold_response = RooUnfoldResponse(
                hMeas,
                hTrue,
                hResponse,
                hResponse.GetName(),
                hResponse.GetName(),
                overflow=overflow,
            )
            # This is the "fakes" caculated by the RooUnfoldResponse
            # The caculation is basically hFakes = hMeas - hResponse.ProjectMeas
            # which should be same as the sum of all not-match categories.
            if store_vfakes:
                vfakes = np.array(unfold_response.Vfakes())
                hVfakes = (
                    measured.get_region(measured_region_name)
                    .get_histogram(histogram_reco)
                    .copy()
                )
                hVfakes.name = f'vfakes_{hVfakes.name}'
                hVfakes.bin_content = vfakes
                unfolded_process.get(region["particle"]).add_histogram(
                    hVfakes, copy=False
                )

            hMeas_scaled = plot.scale_to_xsec(hMeas, lumi)
            hTrue_scaled = plot.scale_to_xsec(hTrue, lumi)
            hAdditionalTruths_scaled = [
                plot.scale_to_xsec(hist, lumi) for hist in hAdditionalTruths
            ]
            hAdditionalRegions_scaled = [
                plot.scale_to_xsec(hist, lumi) for hist in hAdditionalRegions
            ]

            # NB: chi2/plots should be enabled with debug=True and nominal-only processing
            if debug:
                plot.plot_response(
                    hResponse,
                    output_path=base_output_path.joinpath(
                        "response", f"{observable}.pdf"
                    ),
                )

            chi2 = ROOT.TH1F("chi2", "chi2", max_n_unfolds, 0.5, max_n_unfolds + 0.5)
            ndof = len(hTrue)

            tqdm_n_unfolds = tqdm.tqdm(
                total=max_n_unfolds,
                ncols=100,
                leave=False,
                position=4,
                disable=disable_pbar,
            )
            # only one iteration for systematic variations
            if exact_niter:
                iter_list = [max_n_unfolds]
            else:
                max_n_unfolds = 1 if unfoldable.systematic else max_n_unfolds
                iter_list = range(1, max_n_unfolds + 1)
            for n_unfolds in iter_list:
                tqdm_n_unfolds.set_description(f"#unfolds: {n_unfolds}")
                output_path_logs = base_output_path.joinpath(
                    "logs", f"n{n_unfolds}", observable
                )
                output_path_logs.mkdir(parents=True, exist_ok=True)

                with utils.all_redirected(
                    output_path_logs.joinpath("unfold.txt").open("w+")
                ):
                    unfold = RooUnfoldBayes(
                        unfold_response,
                        hUnfoldable,
                        n_unfolds,
                        smoothit=smoothit,
                        handleFakes=handleFakes,
                    )
                    chi2.SetBinContent(n_unfolds, (unfold.Chi2(hTrue, 2) / ndof))

                hReco = unfold.Hunfold(unfold_err)  # hReco is DEPRECATED, use Hunfold
                RootBackend.apply_process_styles(hReco, unfoldable)

                hReco_scaled = plot.scale_to_xsec(hReco, lumi)

                with utils.all_redirected(
                    output_path_logs.joinpath("table.txt").open("w+")
                ):
                    unfold.PrintTable(ROOT.cout, hTrue)

                if debug:
                    plot.plot_results(
                        hMeas_scaled,
                        [hTrue_scaled, *hAdditionalTruths_scaled],
                        hReco_scaled,
                        output_path=base_output_path.joinpath(
                            "iterations", f"n{n_unfolds}", f"{observable}.pdf"
                        ),
                    )

                # hist_truth = formulate.from_auto(histogram['truth']).to_numexpr()
                hist_truth = histogram['truth']
                if n_unfolds == 1 or exact_niter:
                    # TODO: save 1st iteration???
                    # save back in configMgr under truth histogram in particle-level region
                    unfolded_process.get(region["particle"]).get(
                        hist_truth
                    ).root = hReco

                    if debug:
                        plot.plot_results(
                            hMeas_scaled,
                            [hTrue_scaled, *hAdditionalTruths_scaled],
                            hReco_scaled,
                            output_path=base_output_path.joinpath(
                                "unfolded", f"{observable}.pdf"
                            ),
                            additionalRegions=hAdditionalRegions_scaled,
                        )

                    if saveYoda:
                        # NB: for data, need /REF prefix and 'IsRef: 1' annotation
                        yoda_data['data'].append(
                            utils.root2yoda(
                                hReco_scaled,
                                path=f"/REF/CollinearW/{unfold_name}_{observable}",
                            )
                        )
                        yoda_data['data'][-1].setAnnotation('IsRef', 1)
                        yoda_data[truth.name].append(
                            utils.root2yoda(
                                hTrue_scaled,
                                path=f"/CollinearW/{unfold_name}_{observable}",
                            )
                        )
                        for hist, process in zip(
                            hAdditionalTruths_scaled, additionalTruths
                        ):
                            yoda_data[process.name].append(
                                utils.root2yoda(
                                    hist, path=f"/CollinearW/{unfold_name}_{observable}"
                                )
                            )
                elif save_all_iteration:
                    obs_base = unfolded_process.get_region(
                        region["particle"]
                    ).get_histogram(hist_truth)
                    obs_alt = obs_base.copy()
                    obs_alt.name = f"{obs_alt.name}_n{n_unfolds}"
                    obs_alt.root = hReco
                    unfolded_process.get(region["particle"]).add_histogram(
                        obs_alt, copy=False
                    )

                tqdm_n_unfolds.update()
            tqdm_observables.update()
            if debug:
                plot.plot_chi2(
                    chi2,
                    ndof,
                    regionRename(region[measured_region]),
                    observable,
                    output_path=base_output_path.joinpath("chi2", f"{observable}.pdf"),
                )
        tqdm_regions.update()

    # need to do process.copy() / process.clear() / configMgr.append_process(process) -- to make a new process

    if yoda_data:
        yoda_dir = pathlib.Path("./yoda_file")
        yoda_dir.mkdir(parents=True, exist_ok=True)
        for process, data in yoda_data.items():
            utils.yoda_write(
                data,
                str(
                    pathlib.Path(configMgr.out_path).joinpath(
                        f"{yoda_dir}/{process}_{syst_name}_{name}.yoda"
                    )
                ),
            )

    return unfolded_process


def bayes_unfold(
    hUnfoldable,
    hMeas,
    hTruth,
    hRes,
    n_unfolds=1,
    unfold_err=2,
    smoothit=False,
    handleFakes=True,
    priors=None,
    overflow=True,
):
    name = hRes.GetName()
    response = RooUnfoldResponse(hMeas, hTruth, hRes, name, name, overflow=overflow)
    unfold = RooUnfoldBayes(
        response, hUnfoldable, n_unfolds, smoothit=smoothit, handleFakes=handleFakes
    )
    if priors:
        unfold.SetPriors(priors, overflow=overflow)
    hReco = unfold.Hunfold(unfold_err)
    return np.array(hReco), np.array(hReco.GetSumw2())
