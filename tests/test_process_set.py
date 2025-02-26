import pytest

from collinearw import ProcessSet
from collinearw.systematics import Systematics


def test_systematics_duplication():

    # create a ProcessSet with nominal
    pset = ProcessSet.create_nominal("ttbar")

    # defining systematics
    # note the name can be the same
    # the check on duplication is based on name, type, suffix
    sys1 = Systematics("sys1", "exp", "weight1")
    sys1_2 = Systematics("sys1", "theory", "weight1")
    sys1_3 = Systematics("sys1", "exp", "weight2")

    pset.add_systematics(sys1)
    pset.add_systematics(sys1_2)
    pset.add_systematics(sys1_3)

    sys2 = Systematics("sys1", "exp", "weight1")
    with pytest.raises(ValueError):
        pset.add_systematics(sys2)

    assert len(pset.systematics) == 3

    sys_plist = pset.get("sys1")
    assert isinstance(sys_plist, list)
    assert len(sys_plist) == 3

    sys_p = pset.get(("sys1", "exp", "weight1"))
    assert sys_p.systematics == sys1
