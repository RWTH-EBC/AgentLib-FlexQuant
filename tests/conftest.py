import matplotlib
import matplotlib.pyplot as plt
import pytest
import agentlib.modules
matplotlib.use('Agg')


@pytest.fixture(autouse=True)
def mock_plotting(monkeypatch):
    monkeypatch.setattr(plt, 'show', lambda *args, **kwargs: None)
    monkeypatch.setattr(plt, 'savefig', lambda *args, **kwargs: None)

@pytest.fixture(autouse=True)
def clear_agentlib_modules_registry():
    agentlib.modules._MODULE_TYPES.clear()
    agentlib.modules._LOADED_CORE_MODULES = False
