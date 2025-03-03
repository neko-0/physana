import pytest

from physana import ConfigMgr, ProcessSet, Process
from physana.systematics import Systematics


def test_configMgr_interfaces():
    config = ConfigMgr()

    # adding process
    config.add_process("ttbar")

    # the get_process should point to the nominal process (no systematics)
    # and the return object is a instance of Pocess (not ProcessSet)
    # this preserve the old interface before the systematics was added.
    process = config.get_process("ttbar")
    assert isinstance(process, Process)
    assert not isinstance(process, ProcessSet)

    # To get the ProcessSet for the same process name
    process_set = config.get_process_set("ttbar")
    assert isinstance(process_set, ProcessSet)
    assert not isinstance(process_set, Process)

    # check KeyError if process name dose not exist
    with pytest.raises(KeyError):
        _ = config.get_process("wjets")
    with pytest.raises(KeyError):
        _ = config.get_process_set("wjets")

    # the append_process should handle both ProcessSet and Process instance.
    # if the Process instance is appended, a ProcessSet will be created and the
    # Process instance will be set to the nominal. If there's systematics in the
    # Process instance, the instance will be appended into the sytematics under
    # the ProcessSet, and no nominal will be created.
    process_set = ProcessSet.create_nominal("diboson")
    process = Process("singletop")
    config.append_process(process_set)
    config.append_process(process)

    assert isinstance(config.get_process("diboson"), Process)
    assert isinstance(config.get_process("singletop"), Process)

    assert isinstance(config.get_process_set("diboson"), ProcessSet)
    assert isinstance(config.get_process_set("singletop"), ProcessSet)

    singletop_norm = config.get_process_set("singletop").nominal
    assert singletop_norm is config.get_process("singletop")

    # case where a Process instance has systematic
    sys_process = Process("zjets")
    sys_process.systematics = Systematics("dummy_sys", "dummy_type")
    # a ProcessSet will be created during append
    # but no nominal in this case
    config.append_process(sys_process)
    assert isinstance(config.get_process_set("zjets"), ProcessSet)
    with pytest.raises(KeyError):
        _ = config.get_process("zjets")
    assert config.get_process_set("zjets").nominal is None
    assert len(config.get_process_set("zjets").systematics) == 1

    assert "zjets" in config.list_processes()
    assert "zjets" not in config.list_processes(set=False)
