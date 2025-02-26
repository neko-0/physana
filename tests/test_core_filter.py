from collinearw.core import Filter, Process, Region, Histogram


def test_passthru():
    filt = Filter()
    assert not filt.match('')
    assert not filt.match('/process/nominal/region')


def test_filter_process():
    filt = Filter(['/processExclude/nominal/*'])
    assert not filt.match('/process/nominal/region')
    assert filt.match('/processExclude/nominal/region')
    assert filt.match('/processExclude/nominal/anotherRegion')

    p = Process('process')
    p_ex = Process('processExclude')
    r = Region('region', None, None)
    p.add_region(r)
    p_ex.add_region(r)
    assert not filt.accept(p.get_region('region'))
    assert filt.accept(p_ex.get_region('region'))
    assert filt.filter(p.get_region('region'))
    assert not filt.filter(p_ex.get_region('region'))


def test_filter_region():
    filt = Filter(['/*/nominal/regionExclude'])
    assert not filt.match('/process/nominal/region')
    assert filt.match('/process/nominal/regionExclude')
    assert filt.match('/anotherProcess/nominal/regionExclude')

    p = Process('process')
    r = Region('region', None, None)
    r_ex = Region('regionExclude', None, None)
    p.add_region(r)
    p.add_region(r_ex)
    assert not filt.accept(p.get_region('region'))
    assert filt.accept(p.get_region('regionExclude'))
    assert filt.filter(p.get_region('region'))
    assert not filt.filter(p.get_region('regionExclude'))


def test_filter_multiple():
    filt = Filter(['/ttbar/*/inclusive', '/wjets/*/inclusive_2j'])
    assert not filt.match('/ttbar/nominal/inclusive_2j')
    assert not filt.match('/wjets/nominal/inclusive')
    assert filt.match('/ttbar/nominal/inclusive')
    assert filt.match('/wjets/nominal/inclusive_2j')

    ttbar = Process('ttbar')
    wjets = Process('wjets')
    inc = Region('inclusive', None, None)
    inc_2j = Region('inclusive_2j', None, None)
    ttbar.add_region(inc)
    ttbar.add_region(inc_2j)
    wjets.add_region(inc)
    wjets.add_region(inc_2j)
    assert filt.accept(ttbar.get_region('inclusive'))
    assert filt.accept(wjets.get_region('inclusive_2j'))
    assert not filt.accept(wjets.get_region('inclusive'))
    assert not filt.accept(ttbar.get_region('inclusive_2j'))
    assert not filt.filter(ttbar.get_region('inclusive'))
    assert not filt.filter(wjets.get_region('inclusive_2j'))
    assert filt.filter(wjets.get_region('inclusive'))
    assert filt.filter(ttbar.get_region('inclusive_2j'))


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
