import pytest
import collinearw


@pytest.fixture(scope='function')
def mock_configMgr(mocker):
    configMgr = mocker.MagicMock(spec=collinearw.ConfigMgr)
    configMgr.meta_data = {}
    return configMgr
