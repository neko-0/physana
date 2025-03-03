from physana import XSecSumEvtW


def test_xsec_sumw_tokens():
    # this is the default setup
    xsec = XSecSumEvtW()
    xsec.match(
        "/gpfs/slac/atlas/fs1/d/mgignac/analysis/mc16a/CollinearW.SMA.v1/trees/user.mgignac.singletop.410659.e6671_s3126_r9364_p4512_CollinearW.SMA.v1_kinematic_1_t2_tree.root"
    )
    assert xsec['user'] == "mgignac"
    assert xsec['process'] == "singletop"
    assert xsec['dsid'] == "410659"
    assert xsec['syst'] == "kinematic_1"
    assert xsec['user-tag'] == "CollinearW.SMA.v1"
