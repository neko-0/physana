from physana import Process, Region, Histogram
from physana.histo.tools import Filter


def test_passthru():
    filt = Filter()
    assert not filt.match('')
    assert not filt.match('process/nominal/region')


def test_filter_process():
    filt = Filter(['processExclude/nominal/*'])
    assert not filt.match('process/nominal/region')
    assert filt.match('processExclude/nominal/region')
    assert filt.match('processExclude/nominal/anotherRegion')

    p = Process('process')
    p_ex = Process('processExclude')
    r = Region('region', None, None)
    p.append(r)
    p_ex.append(r)
    assert not filt.accept(p.get('region'))
    assert filt.accept(p_ex.get('region'))
    assert filt.filter(p.get('region'))
    assert not filt.filter(p_ex.get('region'))


def test_filter_region():
    filt = Filter(['*/nominal/regionExclude'])
    assert not filt.match('process/nominal/region')
    assert filt.match('process/nominal/regionExclude')
    assert filt.match('anotherProcess/nominal/regionExclude')

    p = Process('process')
    r = Region('region', None, None)
    r_ex = Region('regionExclude', None, None)
    p.append(r)
    p.append(r_ex)
    assert not filt.accept(p.get('region'))
    assert filt.accept(p.get('regionExclude'))
    assert filt.filter(p.get('region'))
    assert not filt.filter(p.get('regionExclude'))


def test_filter_multiple():
    filt = Filter(['ttbar/*/inclusive', 'wjets/*/inclusive_2j'])
    assert not filt.match('ttbar/nominal/inclusive_2j')
    assert not filt.match('wjets/nominal/inclusive')
    assert filt.match('ttbar/nominal/inclusive')
    assert filt.match('wjets/nominal/inclusive_2j')

    ttbar = Process('ttbar')
    wjets = Process('wjets')
    inc = Region('inclusive', None, None)
    inc_2j = Region('inclusive_2j', None, None)
    ttbar.append(inc)
    ttbar.append(inc_2j)
    wjets.append(inc)
    wjets.append(inc_2j)
    assert filt.accept(ttbar.get('inclusive'))
    assert filt.accept(wjets.get('inclusive_2j'))
    assert not filt.accept(wjets.get('inclusive'))
    assert not filt.accept(ttbar.get('inclusive_2j'))
    assert not filt.filter(ttbar.get('inclusive'))
    assert not filt.filter(wjets.get('inclusive_2j'))
    assert filt.filter(wjets.get('inclusive'))
    assert filt.filter(ttbar.get('inclusive_2j'))


def test_filter_histogram_type():
    filt = Filter(["mjj", "njet"], "name")
    filt2 = Filter(key="type")
    mjj = Histogram("mjj", 10, 1, 10)
    njet = Histogram("njet", 10, 1, 10)
    wpt = Histogram("wpt", 10, 1, 10)
    assert filt.accept(mjj)
    assert filt.accept(njet)
    assert not filt.accept(wpt)
    assert filt2.accept(mjj)
    assert filt2.accept(njet)
    assert filt2.accept(wpt)
