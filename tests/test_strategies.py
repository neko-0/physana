import collinearw
import collinearw.strategies.unfolding
import collinearw.strategies.unfolding.metadata
import logging


def test_unfolding_add_regions(mock_configMgr):
    collinearw.strategies.unfolding.add_regions(
        mock_configMgr,
        "unfold_A",
        baseline_selection="baseline_selection",
        truth_selection="truth_selection",
        reco_selection="reco_selection",
        matching_selection="matching_selection",
        notmatching_selection="notmatching_selection",
        weight_truth="weight_truth",
        weight_reco="weight_reco",
    )
    assert mock_configMgr.add_region.call_count == 4
    assert 'unfold' in mock_configMgr.meta_data
    assert 'regions' in mock_configMgr.meta_data['unfold']
    assert 'unfold_A' in mock_configMgr.meta_data['unfold']['regions']
    assert mock_configMgr.meta_data['unfold']['regions']['unfold_A'] == {
        'particle': 'unfold_A_truth',
        'reco': 'unfold_A_reco',
        'reco_match': 'unfold_A_truth_reco_matching',
        'reco_match_not': 'unfold_A_truth_reco_notmatching',
    }


def test_unfolding_add_observables(mock_configMgr):
    collinearw.strategies.unfolding.add_observables(
        mock_configMgr,
        "lep1Pt",
        "lep1TruthPt",
        19,
        25,
        500,
        "Leading lepton p_{T} [GeV]",
    )

    assert mock_configMgr.add_observable.call_count == 2
    assert mock_configMgr.add_histogram2D.call_count == 1
    assert 'unfold' in mock_configMgr.meta_data
    assert 'observables' in mock_configMgr.meta_data['unfold']
    assert 'lep1Pt' in mock_configMgr.meta_data['unfold']['observables']
    assert mock_configMgr.meta_data['unfold']['observables']['lep1Pt'] == {
        'reco': 'lep1Pt',
        'response': 'response_matrix_lep1Pt',
        'truth': 'lep1TruthPt',
    }


def test_unfolding_add_regions_twice(mock_configMgr, caplog):
    collinearw.strategies.unfolding.add_regions(
        mock_configMgr,
        "unfold_A",
        baseline_selection="baseline_selection",
        truth_selection="truth_selection",
        reco_selection="reco_selection",
        matching_selection="matching_selection",
        notmatching_selection="notmatching_selection",
        weight_truth="weight_truth",
        weight_reco="weight_reco",
    )
    with caplog.at_level(logging.WARNING, 'collinearw.strategies.unfolding'):
        collinearw.strategies.unfolding.add_regions(
            mock_configMgr,
            "unfold_A",
            baseline_selection="baseline_selection2",
            truth_selection="truth_selection2",
            reco_selection="reco_selection2",
            matching_selection="matching_selection2",
            notmatching_selection="notmatching_selection2",
            weight_truth="weight_truth2",
            weight_reco="weight_reco2",
        )

    assert (
        "already registered" in caplog.text
    ), "No warning raised when adding duplicate unfolding regions"
    assert mock_configMgr.meta_data['unfold']['regions']['unfold_A'] == {
        'particle': 'unfold_A_truth',
        'reco': 'unfold_A_reco',
        'reco_match': 'unfold_A_truth_reco_matching',
        'reco_match_not': 'unfold_A_truth_reco_notmatching',
    }


def test_unfolding_add_observables_twice(mock_configMgr, caplog):
    collinearw.strategies.unfolding.add_observables(
        mock_configMgr,
        "lep1Pt",
        "lep1TruthPt",
        19,
        25,
        500,
        "Leading lepton p_{T} [GeV]",
    )
    with caplog.at_level(logging.WARNING, 'collinearw.strategies.unfolding'):
        collinearw.strategies.unfolding.add_observables(
            mock_configMgr,
            "lep1Pt",
            "lep1TruthPt2",
            19,
            25,
            500,
            "Leading lepton p_{T} [GeV]",
        )

    assert (
        "already registered" in caplog.text
    ), "No warning raised when adding duplicate unfolding observables"
    assert mock_configMgr.meta_data['unfold']['observables']['lep1Pt'] == {
        'reco': 'lep1Pt',
        'response': 'response_matrix_lep1Pt',
        'truth': 'lep1TruthPt',
    }


def test_unfolding_metatada_check(mock_configMgr, caplog):
    mock_configMgr.meta_data['unfold'] = {
        'regions': {'duplicate_region': {}},
        'observables': {'duplicate_observable': {}},
    }
    with caplog.at_level(logging.WARNING, 'collinearw.strategies.unfolding'):
        assert collinearw.strategies.unfolding.metadata.has(
            mock_configMgr, 'regions', 'duplicate_region'
        )
    assert (
        "already registered" in caplog.text
    ), "No warning raised when checking duplicate regions"
    caplog.clear()
    with caplog.at_level(logging.WARNING, 'collinearw.strategies.unfolding'):
        assert collinearw.strategies.unfolding.metadata.has(
            mock_configMgr, 'observables', 'duplicate_observable'
        )
    assert (
        "already registered" in caplog.text
    ), "No warning raised when checking duplicate observables"


def test_unfolding_metadata_save(mock_configMgr):
    assert collinearw.strategies.unfolding.metadata.save(
        mock_configMgr, 'regions', 'my_region', {'hello': 'world'}
    )
    assert mock_configMgr.meta_data['unfold']['regions']['my_region'] == {
        'hello': 'world'
    }
