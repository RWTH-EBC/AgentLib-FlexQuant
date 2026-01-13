import matplotlib
import matplotlib.pyplot as plt
import pytest
from pathlib import Path
from util import convert_paths_to_absolute_in_json
matplotlib.use('Agg')


@pytest.fixture(autouse=True)
def mock_plotting(monkeypatch):
    monkeypatch.setattr(plt, 'show', lambda *args, **kwargs: None)
    monkeypatch.setattr(plt, 'savefig', lambda *args, **kwargs: None)


@pytest.fixture(scope="session", autouse=True)
def setup_absolute_paths():
    """Convert all config file paths to absolute paths before any tests run."""
    json_files = [
        Path(__file__).parent / "sample_files" / "configs" / "flexibility_input.json",
        Path(__file__).parent / "sample_files" / "configs" / "flexibility.json",
        Path(__file__).parent / "sample_files" / "configs" / "flexibility_market_input.json",
        Path(__file__).parent / "sample_files" / "configs" / "indicator.json",
        Path(__file__).parent / "sample_files" / "configs" / "market.json",
        Path(__file__).parent / "sample_files" / "configs" / "neg_flex.json",
        Path(__file__).parent / "sample_files" / "configs" / "pos_flex.json",
        Path(__file__).parent / "sample_files" / "configs" / "simulator.json",
    ]
    convert_paths_to_absolute_in_json(json_files)
