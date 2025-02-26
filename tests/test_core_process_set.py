import pytest
from collinearw import ProcessSet
from collinearw import Systematics


def test_systematics_duplication():

    # create a ProcessSet with nominal
    pset = ProcessSet.create_nominal("ttbar")

    # defining systematics
    # note the name can be the same
    # the check on duplication is based on name,treename,weight
    sys1 = Systematics("sys1", "tree1", "weight1", "source")
    sys1_2 = Systematics("sys1", "tree2", "weight1", "source")
    sys1_3 = Systematics("sys1", "tree1", "weight2", "source")

    pset.add_systematic(sys1)
    pset.add_systematic(sys1_2)
    pset.add_systematic(sys1_3)

    sys2 = Systematics("sys1", "tree1", "weight1", "source")
    with pytest.raises(ValueError):
        pset.add_systematic(sys2)

    assert len(pset.systematics) == 3

    sys_plist = pset.get("sys1")
    assert isinstance(sys_plist, list)
    assert len(sys_plist) == 3

    sys_p = pset.get(("sys1", "tree1", "weight1"))
    assert sys_p.systematic == sys1
