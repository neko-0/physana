from collinearw import Process, Region, Histogram, Systematics


def test_histogram_full_name():
    wjets_process = Process("wjets", "wjets_NoSys")
    zjets_process = Process("zjets", "zjets_NoSys")

    sys_up = Systematics("sys", "nominal", "weight_up", "tree")
    sys_down = Systematics("sys", "nominal", "weight_down", "tree")

    lhs_region = Region("lhs", "recoWeight", "nJet25>2")
    rhs_region = Region("rhs", "recoWeight", "nJet25>2")

    histo = Histogram("jetPt", 3, 0, 100, "x title")

    lhs_region.add_histogram(histo)
    rhs_region.add_histogram(histo)

    wjets_process.add_region(lhs_region)
    zjets_process.add_region(rhs_region)

    assert wjets_process.name == 'wjets'
    assert wjets_process.full_name == '/wjets/nominal'
    assert wjets_process.get_region('lhs').name == 'lhs'
    assert wjets_process.get_region('lhs').full_name == '/wjets/nominal/lhs'
    assert wjets_process.get_region('lhs').get_histogram('jetPt').name == 'jetPt'
    assert (
        wjets_process.get_region('lhs').get_histogram('jetPt').full_name
        == '/wjets/nominal/lhs/jetPt'
    )

    # test with systematic

    zjets_process.systematic = sys_up

    assert zjets_process.name == 'zjets'
    assert zjets_process.full_name == '/zjets/sys_nominal_weight_up'
    assert zjets_process.get_region('rhs').name == 'rhs'
    assert (
        zjets_process.get_region('rhs').full_name == '/zjets/sys_nominal_weight_up/rhs'
    )
    assert zjets_process.get_region('rhs').get_histogram('jetPt').name == 'jetPt'
    assert (
        zjets_process.get_region('rhs').get_histogram('jetPt').full_name
        == '/zjets/sys_nominal_weight_up/rhs/jetPt'
    )

    zjets_process.systematic = sys_down

    assert zjets_process.name == 'zjets'
    assert zjets_process.full_name == '/zjets/sys_nominal_weight_down'
    assert zjets_process.get_region('rhs').name == 'rhs'
    assert (
        zjets_process.get_region('rhs').full_name
        == '/zjets/sys_nominal_weight_down/rhs'
    )
    assert zjets_process.get_region('rhs').get_histogram('jetPt').name == 'jetPt'
    assert (
        zjets_process.get_region('rhs').get_histogram('jetPt').full_name
        == '/zjets/sys_nominal_weight_down/rhs/jetPt'
    )
