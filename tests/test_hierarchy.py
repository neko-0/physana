from physana import Process, Region, Histogram
from physana.systematics import Systematics


def test_histogram_full_name():
    wjets_process = Process("wjets", "wjets_NoSys")
    zjets_process = Process("zjets", "zjets_NoSys")

    sys_up = Systematics("sys", "theory", "weight_up")
    sys_down = Systematics("sys", "theory", "weight_down")

    lhs_region = Region("lhs", "recoWeight", "nJet25>2")
    rhs_region = Region("rhs", "recoWeight", "nJet25>2")

    histo = Histogram("jetPt", 3, 0, 100, "x title")

    lhs_region.append(histo)
    rhs_region.append(histo)

    wjets_process.append(lhs_region)
    zjets_process.append(rhs_region)

    assert wjets_process.name == 'wjets'
    assert wjets_process.full_name == 'wjets/nominal'
    assert wjets_process.get('lhs').name == 'lhs'
    assert wjets_process.get('lhs').full_name == 'wjets/nominal/lhs'
    assert wjets_process.get('lhs').get('jetPt').name == 'jetPt'
    assert wjets_process.get('lhs').get('jetPt').full_name == 'wjets/nominal/lhs/jetPt'

    # test with systematic

    zjets_process.systematics = sys_up

    assert zjets_process.name == 'zjets'
    assert zjets_process.full_name == 'zjets/sys_theory_weight_up'
    assert zjets_process.get('rhs').name == 'rhs'
    assert zjets_process.get('rhs').full_name == 'zjets/sys_theory_weight_up/rhs'
    assert zjets_process.get('rhs').get('jetPt').name == 'jetPt'
    assert (
        zjets_process.get('rhs').get('jetPt').full_name
        == 'zjets/sys_theory_weight_up/rhs/jetPt'
    )

    zjets_process.systematics = sys_down

    assert zjets_process.name == 'zjets'
    assert zjets_process.full_name == 'zjets/sys_theory_weight_down'
    assert zjets_process.get('rhs').name == 'rhs'
    assert zjets_process.get('rhs').full_name == 'zjets/sys_theory_weight_down/rhs'
    assert zjets_process.get('rhs').get('jetPt').name == 'jetPt'
    assert (
        zjets_process.get('rhs').get('jetPt').full_name
        == 'zjets/sys_theory_weight_down/rhs/jetPt'
    )
