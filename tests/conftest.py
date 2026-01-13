import matplotlib
import matplotlib.pyplot as plt
from util import convert_paths_to_absolute_in_json
matplotlib.use('Agg')


@pytest.fixture(autouse=True)
def mock_plotting(monkeypatch):
    monkeypatch.setattr(plt, 'show', lambda *args, **kwargs: None)
    monkeypatch.setattr(plt, 'savefig', lambda *args, **kwargs: None)
