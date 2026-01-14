import matplotlib
import matplotlib.pyplot as plt
import pytest
matplotlib.use('Agg')


@pytest.fixture(autouse=True)
def mock_plotting(monkeypatch):
    monkeypatch.setattr(plt, 'show', lambda *args, **kwargs: None)
    monkeypatch.setattr(plt, 'savefig', lambda *args, **kwargs: None)
