import pytest
import physana


@pytest.fixture(scope='function')
def mock_configMgr(mocker):
    configMgr = mocker.MagicMock(spec=physana.ConfigMgr)
    configMgr.meta_data = {}
    return configMgr
