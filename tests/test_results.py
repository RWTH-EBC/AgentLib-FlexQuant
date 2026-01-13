"""
Tests for the Results class.

This test module verifies the Results class functionality for loading and managing
flexibility analysis results.

Run with: pytest test_results.py -v
"""

import copy
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from agentlib_flexquant.data_structures.flex_results import (
    Results,
    load_indicator,
    load_market,
)


# =============================================================================
# Path configuration
# =============================================================================
# Project root (parent of tests directory)
PROJECT_ROOT = Path(__file__).parent.parent.parent

SAMPLE_FILES_DIR = Path(__file__).parent / "sample_files"
CONFIGS_DIR = SAMPLE_FILES_DIR / "configs"
RESULTS_DIR = SAMPLE_FILES_DIR / "sample_results"

# Config file paths, test for both flex configs (the one used as input for the
# FlexAgentGenerator and the one created by it)
FLEX_CONFIG_PATH = CONFIGS_DIR / "flexibility_agent_config.json"
FLEX_CONFIG_INPUT_PATH = CONFIGS_DIR / "flexibility_agent_config_input.json"
BASELINE_CONFIG_PATH = CONFIGS_DIR / "baseline.json"
POS_FLEX_CONFIG_PATH = CONFIGS_DIR / "pos_flex.json"
NEG_FLEX_CONFIG_PATH = CONFIGS_DIR / "neg_flex.json"
INDICATOR_CONFIG_PATH = CONFIGS_DIR / "indicator.json"
SIMULATOR_CONFIG_PATH = CONFIGS_DIR / "simulator.json"

# Result file paths
MPC_BASE_PATH = RESULTS_DIR / "mpc_base.csv"
MPC_POS_FLEX_PATH = RESULTS_DIR / "mpc_pos_flex.csv"
MPC_NEG_FLEX_PATH = RESULTS_DIR / "mpc_neg_flex.csv"
STATS_MPC_BASE_PATH = RESULTS_DIR / "stats_mpc_base.csv"
STATS_MPC_POS_FLEX_PATH = RESULTS_DIR / "stats_mpc_pos_flex.csv"
STATS_MPC_NEG_FLEX_PATH = RESULTS_DIR / "stats_mpc_neg_flex.csv"
INDICATOR_RESULTS_PATH = RESULTS_DIR / "flexibility_indicator.csv"
MARKET_RESULTS_PATH = RESULTS_DIR / "flexibility_market.csv"
SIMULATOR_RESULTS_PATH = RESULTS_DIR / "simulator.csv"


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(params=[FLEX_CONFIG_PATH, FLEX_CONFIG_INPUT_PATH], ids=["flex_config", "flex_config_input"])
def flex_config_path(request):
    """Return path to flex config (parameterized for both config variants)."""
    return request.param


@pytest.fixture(params=[FLEX_CONFIG_PATH, FLEX_CONFIG_INPUT_PATH], ids=["flex_config", "flex_config_input"])
def flex_config_dict(request):
    """Load and return flex config as dict (parameterized for both config variants)."""
    with open(request.param, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def simulator_config_path():
    """Return path to simulator config."""
    return SIMULATOR_CONFIG_PATH


@pytest.fixture
def simulator_config_dict():
    """Load and return simulator config as dict."""
    with open(SIMULATOR_CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def results_instance(flex_config_path, simulator_config_path):
    """Create a fully initialized Results instance."""
    return Results(
        flex_config=flex_config_path,
        simulator_agent_config=simulator_config_path,
        generated_flex_files_base_path=SAMPLE_FILES_DIR,
        results=RESULTS_DIR,
        to_timescale="seconds",
    )


@pytest.fixture
def results_instance_no_simulator(flex_config_path):
    """Create a Results instance without simulator config."""
    return Results(
        flex_config=flex_config_path,
        simulator_agent_config=None,
        generated_flex_files_base_path=SAMPLE_FILES_DIR,
        results=RESULTS_DIR,
        to_timescale="seconds",
    )


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# =============================================================================
# Tests for standalone loading functions
# =============================================================================


class TestLoadIndicator:
    """Tests for the load_indicator function."""

    def test_load_indicator_returns_dataframe(self):
        """Test that load_indicator returns a DataFrame."""
        df = load_indicator(INDICATOR_RESULTS_PATH)
        assert isinstance(df, pd.DataFrame)

    def test_load_indicator_has_multiindex(self):
        """Test that loaded DataFrame has MultiIndex."""
        df = load_indicator(INDICATOR_RESULTS_PATH)
        assert isinstance(df.index, pd.MultiIndex)
        assert df.index.nlevels == 2

    def test_load_indicator_not_empty(self):
        """Test that loaded DataFrame is not empty."""
        df = load_indicator(INDICATOR_RESULTS_PATH)
        assert len(df) > 0

    def test_load_indicator_with_string_path(self):
        """Test loading with string path instead of Path object."""
        df = load_indicator(str(INDICATOR_RESULTS_PATH))
        assert isinstance(df, pd.DataFrame)


class TestLoadMarket:
    """Tests for the load_market function."""

    def test_load_market_returns_dataframe(self):
        """Test that load_market returns a DataFrame."""
        df = load_market(MARKET_RESULTS_PATH)
        assert isinstance(df, pd.DataFrame)

    def test_load_market_has_multiindex(self):
        """Test that loaded DataFrame has MultiIndex."""
        df = load_market(MARKET_RESULTS_PATH)
        assert isinstance(df.index, pd.MultiIndex)
        assert df.index.nlevels == 2

    def test_load_market_not_empty(self):
        """Test that loaded DataFrame is not empty."""
        df = load_market(MARKET_RESULTS_PATH)
        assert len(df) > 0

    def test_load_market_with_string_path(self):
        """Test loading with string path instead of Path object."""
        df = load_market(str(MARKET_RESULTS_PATH))
        assert isinstance(df, pd.DataFrame)


# =============================================================================
# Tests for Results class initialization
# =============================================================================


class TestResultsInit:
    """Tests for Results class initialization."""

    def test_init_with_all_configs(self, flex_config_path, simulator_config_path):
        """Test initialization with all configurations provided."""
        results = Results(
            flex_config=flex_config_path,
            simulator_agent_config=simulator_config_path,
            generated_flex_files_base_path=SAMPLE_FILES_DIR,
            results=RESULTS_DIR,
            to_timescale="seconds",
        )
        assert results is not None
        assert hasattr(results, "df_baseline")
        assert hasattr(results, "df_pos_flex")
        assert hasattr(results, "df_neg_flex")
        assert hasattr(results, "df_indicator")

    def test_init_without_simulator_config(self, flex_config_path):
        """Test initialization without simulator agent config."""
        results = Results(
            flex_config=flex_config_path,
            simulator_agent_config=None,
            generated_flex_files_base_path=SAMPLE_FILES_DIR,
            results=RESULTS_DIR,
            to_timescale="seconds",
        )
        assert results is not None
        assert not hasattr(results, "df_simulation") or results.df_simulation is None

    def test_init_with_dict_flex_config(self, flex_config_dict, simulator_config_path):
        """Test initialization with flex_config as dict."""
        results = Results(
            flex_config=flex_config_dict,
            simulator_agent_config=simulator_config_path,
            generated_flex_files_base_path=SAMPLE_FILES_DIR,
            results=RESULTS_DIR,
            to_timescale="seconds",
        )
        assert results is not None

    def test_init_with_dict_simulator_config(self, flex_config_path, simulator_config_dict):
        """Test initialization with simulator_agent_config as dict."""
        results = Results(
            flex_config=flex_config_path,
            simulator_agent_config=simulator_config_dict,
            generated_flex_files_base_path=SAMPLE_FILES_DIR,
            results=RESULTS_DIR,
            to_timescale="seconds",
        )
        assert results is not None
        assert hasattr(results, "df_simulation")

    def test_init_with_string_paths(self, flex_config_path):
        """Test initialization with string paths instead of Path objects."""
        results = Results(
            flex_config=str(flex_config_path),
            simulator_agent_config=str(SIMULATOR_CONFIG_PATH),
            generated_flex_files_base_path=str(SAMPLE_FILES_DIR),
            results=str(RESULTS_DIR),
            to_timescale="seconds",
        )
        assert results is not None

    def test_init_from_existing_results_instance(self, results_instance):
        """Test that Results can be initialized from another Results instance."""
        new_results = Results(
            flex_config=None,
            simulator_agent_config=None,
            results=results_instance,
        )
        assert new_results is not None
        assert new_results.df_baseline is not None
        # Verify it's a deep copy
        assert new_results is not results_instance

    def test_init_with_different_timescales(self, flex_config_path, simulator_config_path):
        """Test initialization with different timescale options."""
        for timescale in ["seconds", "minutes", "hours"]:
            results = Results(
                flex_config=flex_config_path,
                simulator_agent_config=simulator_config_path,
                generated_flex_files_base_path=SAMPLE_FILES_DIR,
                results=RESULTS_DIR,
                to_timescale=timescale,
            )
            assert results.current_timescale_of_data == timescale


# =============================================================================
# Tests for _load_flex_config
# =============================================================================


class TestLoadFlexConfig:
    """Tests for the _load_flex_config method."""

    def test_load_flex_config_from_path(self, flex_config_path):
        """Test loading flex config from file path."""
        results = Results(
            flex_config=flex_config_path,
            simulator_agent_config=None,
            generated_flex_files_base_path=SAMPLE_FILES_DIR,
            results=RESULTS_DIR,
        )
        assert results.flex_config is not None

    def test_load_flex_config_from_dict(self, flex_config_dict):
        """Test loading flex config from dict."""
        results = Results(
            flex_config=flex_config_dict,
            simulator_agent_config=None,
            generated_flex_files_base_path=SAMPLE_FILES_DIR,
            results=RESULTS_DIR,
        )
        assert results.flex_config is not None

    def test_load_flex_config_with_custom_base_path_overrides(
        self, flex_config_path, temp_dir
    ):
        """Test that custom_base_path overrides flex_base_directory_path."""
        # Copy necessary files to temp_dir for this test
        # The custom base path should override what's in the config
        with open(flex_config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        original_base_path = config.get("flex_base_directory_path")

        results = Results(
            flex_config=flex_config_path,
            simulator_agent_config=None,
            generated_flex_files_base_path=SAMPLE_FILES_DIR,
            results=RESULTS_DIR,
        )

        # The generator config should have the overridden path
        assert str(results.flex_config.flex_base_directory_path) == str(
            SAMPLE_FILES_DIR
        )


# =============================================================================
# Tests for _load_simulator_config
# =============================================================================


class TestLoadSimulatorConfig:
    """Tests for the _load_simulator_config method."""

    def test_load_simulator_config_from_path(
        self, flex_config_path, simulator_config_path
    ):
        """Test loading simulator config from file path."""
        results = Results(
            flex_config=flex_config_path,
            simulator_agent_config=simulator_config_path,
            generated_flex_files_base_path=SAMPLE_FILES_DIR,
            results=RESULTS_DIR,
        )
        assert results.simulator_agent_config is not None
        assert results.simulator_module_config is not None

    def test_load_simulator_config_from_dict(
        self, flex_config_path, simulator_config_dict
    ):
        """Test loading simulator config from dict."""
        results = Results(
            flex_config=flex_config_path,
            simulator_agent_config=simulator_config_dict,
            generated_flex_files_base_path=SAMPLE_FILES_DIR,
            results=RESULTS_DIR,
        )
        assert results.simulator_agent_config is not None
        assert results.simulator_module_config is not None

    def test_simulator_module_has_result_filename(
        self, flex_config_path, simulator_config_path
    ):
        """Test that simulator module config has result_filename attribute."""
        results = Results(
            flex_config=flex_config_path,
            simulator_agent_config=simulator_config_path,
            generated_flex_files_base_path=SAMPLE_FILES_DIR,
            results=RESULTS_DIR,
        )
        assert hasattr(results.simulator_module_config, "result_filename")


# =============================================================================
# Tests for _resolve_sim_results_path
# =============================================================================


class TestResolveSimResultsPath:
    """Tests for the _resolve_sim_results_path method."""

    def test_resolve_absolute_path_exists(self, results_instance):
        """Test resolution when absolute path exists."""
        resolved = results_instance._resolve_sim_results_path(
            str(SIMULATOR_RESULTS_PATH.absolute()), RESULTS_DIR
        )
        assert resolved.exists()

    def test_resolve_filename_in_results_dir(self, results_instance):
        """Test resolution when only filename given and file is in results dir."""
        resolved = results_instance._resolve_sim_results_path(
            "simulator.csv", RESULTS_DIR
        )
        assert resolved.exists()
        assert resolved.name == "simulator.csv"

    def test_resolve_file_not_found_raises(self, results_instance):
        """Test that FileNotFoundError is raised when file cannot be found."""
        with pytest.raises(
            FileNotFoundError, match="Could not locate simulator results file"
        ):
            results_instance._resolve_sim_results_path(
                "nonexistent_file.csv", RESULTS_DIR
            )

    def test_resolve_with_path_object(self, results_instance):
        """Test resolution with Path object instead of string."""
        resolved = results_instance._resolve_sim_results_path(
            Path("simulator.csv"), RESULTS_DIR
        )
        assert resolved.exists()


# =============================================================================
# Tests for _load_results
# =============================================================================


class TestLoadResults:
    """Tests for the _load_results method."""

    def test_load_results_returns_tuple(self, results_instance):
        """Test that _load_results returns a tuple of (dict, path)."""
        # We can't easily call _load_results directly after init,
        # but we can verify the results were loaded correctly
        assert results_instance.df_baseline is not None
        assert results_instance.df_pos_flex is not None
        assert results_instance.df_neg_flex is not None
        assert results_instance.df_indicator is not None

    def test_load_results_with_path_string(self, flex_config_path, simulator_config_path):
        """Test loading results with string path."""
        results = Results(
            flex_config=flex_config_path,
            simulator_agent_config=simulator_config_path,
            generated_flex_files_base_path=SAMPLE_FILES_DIR,
            results=str(RESULTS_DIR),
            to_timescale="seconds",
        )
        assert results.df_baseline is not None


# =============================================================================
# Tests for _load_results_dataframes
# =============================================================================


class TestLoadResultsDataframes:
    """Tests for the _load_results_dataframes method."""

    def test_dataframes_are_pandas_dataframes(self, results_instance):
        """Test that all loaded results are pandas DataFrames."""
        assert isinstance(results_instance.df_baseline, pd.DataFrame)
        assert isinstance(results_instance.df_pos_flex, pd.DataFrame)
        assert isinstance(results_instance.df_neg_flex, pd.DataFrame)
        assert isinstance(results_instance.df_indicator, pd.DataFrame)

    def test_dataframes_not_empty(self, results_instance):
        """Test that loaded DataFrames are not empty."""
        assert len(results_instance.df_baseline) > 0
        assert len(results_instance.df_pos_flex) > 0
        assert len(results_instance.df_neg_flex) > 0
        assert len(results_instance.df_indicator) > 0

    def test_simulation_dataframe_loaded_when_simulator_present(self, results_instance):
        """Test that df_simulation is loaded when simulator config is present."""
        assert hasattr(results_instance, "df_simulation")
        assert isinstance(results_instance.df_simulation, pd.DataFrame)
        assert len(results_instance.df_simulation) > 0

    def test_simulation_dataframe_not_present_without_simulator(
        self, results_instance_no_simulator
    ):
        """Test that df_simulation is not set when simulator config is absent."""
        assert not hasattr(results_instance_no_simulator, "df_simulation")

    def test_market_dataframe_loaded_when_market_config_present(self, results_instance):
        """Test that df_market is loaded when market config is present."""
        if results_instance.flex_config.market_config:
            assert results_instance.df_market is not None
            assert isinstance(results_instance.df_market, pd.DataFrame)

    def test_market_dataframe_none_when_market_config_absent(self, flex_config_dict):
        """Test that df_market is None when market config is absent."""
        # Modify config to remove market config
        flex_config_dict["market_config"] = None
        results = Results(
            flex_config=flex_config_dict,
            simulator_agent_config=None,
            generated_flex_files_base_path=SAMPLE_FILES_DIR,
            results=RESULTS_DIR,
        )
        assert results.df_market is None


# =============================================================================
# Tests for _load_stats_dataframes
# =============================================================================


class TestLoadStatsDataframes:
    """Tests for the _load_stats_dataframes method."""

    def test_stats_dataframes_loaded(self, results_instance):
        """Test that stats DataFrames are loaded."""
        assert hasattr(results_instance, "df_baseline_stats")
        assert hasattr(results_instance, "df_pos_flex_stats")
        assert hasattr(results_instance, "df_neg_flex_stats")

    def test_stats_dataframes_are_pandas_dataframes(self, results_instance):
        """Test that stats are pandas DataFrames."""
        assert isinstance(results_instance.df_baseline_stats, pd.DataFrame)
        assert isinstance(results_instance.df_pos_flex_stats, pd.DataFrame)
        assert isinstance(results_instance.df_neg_flex_stats, pd.DataFrame)


# =============================================================================
# Tests for convert_timescale_of_dataframe_index
# =============================================================================


class TestConvertTimescale:
    """Tests for the convert_timescale_of_dataframe_index method."""

    def test_convert_to_minutes(self, flex_config_path, simulator_config_path):
        """Test converting timescale to minutes."""
        results = Results(
            flex_config=flex_config_path,
            simulator_agent_config=simulator_config_path,
            generated_flex_files_base_path=SAMPLE_FILES_DIR,
            results=RESULTS_DIR,
            to_timescale="seconds",
        )
        results.convert_timescale_of_dataframe_index(to_timescale="minutes")
        assert results.current_timescale_of_data == "minutes"

    def test_convert_to_hours(self, flex_config_path, simulator_config_path):
        """Test converting timescale to hours."""
        results = Results(
            flex_config=flex_config_path,
            simulator_agent_config=simulator_config_path,
            generated_flex_files_base_path=SAMPLE_FILES_DIR,
            results=RESULTS_DIR,
            to_timescale="seconds",
        )
        results.convert_timescale_of_dataframe_index(to_timescale="hours")
        assert results.current_timescale_of_data == "hours"

    def test_convert_updates_current_timescale(self, results_instance):
        """Test that current_timescale_of_data is updated after conversion."""
        original_timescale = results_instance.current_timescale_of_data
        new_timescale = "minutes" if original_timescale != "minutes" else "hours"
        results_instance.convert_timescale_of_dataframe_index(to_timescale=new_timescale)
        assert results_instance.current_timescale_of_data == new_timescale

    def test_convert_multiple_times(self, results_instance):
        """Test converting timescale multiple times."""
        results_instance.convert_timescale_of_dataframe_index(to_timescale="minutes")
        assert results_instance.current_timescale_of_data == "minutes"

        results_instance.convert_timescale_of_dataframe_index(to_timescale="hours")
        assert results_instance.current_timescale_of_data == "hours"

        results_instance.convert_timescale_of_dataframe_index(to_timescale="seconds")
        assert results_instance.current_timescale_of_data == "seconds"


# =============================================================================
# Tests for get_intersection_mpcs_sim
# =============================================================================


class TestGetIntersectionMpcsSim:
    """Tests for the get_intersection_mpcs_sim method."""

    def test_returns_dict(self, results_instance):
        """Test that method returns a dictionary."""
        result = results_instance.get_intersection_mpcs_sim()
        assert isinstance(result, dict)

    def test_dict_values_are_dicts(self, results_instance):
        """Test that dictionary values are also dictionaries."""
        result = results_instance.get_intersection_mpcs_sim()
        for key, value in result.items():
            assert isinstance(value, dict), f"Value for key '{key}' is not a dict"

    def test_includes_module_ids(self, results_instance):
        """Test that inner dicts contain module IDs as keys."""
        result = results_instance.get_intersection_mpcs_sim()
        if result:
            first_value = next(iter(result.values()))
            # Should contain module IDs from the configs
            assert len(first_value) > 0


# =============================================================================
# Tests for create_instance_with_skipped_validation
# =============================================================================


class TestCreateInstanceWithSkippedValidation:
    """Tests for the create_instance_with_skipped_validation method."""

    def test_bypassed_fields_metadata_stored(self, results_instance):
        """Test that _bypassed_fields metadata is stored on simulator module config."""
        if hasattr(results_instance, "simulator_module_config"):
            assert hasattr(results_instance.simulator_module_config, "_bypassed_fields")
            assert "result_filename" in results_instance.simulator_module_config._bypassed_fields

    def test_original_config_metadata_stored(self, results_instance):
        """Test that _original_config metadata is stored on instance."""
        if hasattr(results_instance, "simulator_module_config"):
            assert hasattr(results_instance.simulator_module_config, "_original_config")
            assert isinstance(results_instance.simulator_module_config._original_config, dict)

    def test_skipped_field_is_set(self, results_instance):
        """Test that skipped field (result_filename) is still set on the instance."""
        if hasattr(results_instance, "simulator_module_config"):
            assert hasattr(results_instance.simulator_module_config, "result_filename")


# =============================================================================
# Tests for __deepcopy__
# =============================================================================


class TestDeepCopy:
    """Tests for the custom __deepcopy__ implementation."""

    def test_deepcopy_creates_new_instance(self, results_instance):
        """Test that deepcopy creates a new instance."""
        copied = copy.deepcopy(results_instance)
        assert copied is not results_instance

    def test_deepcopy_preserves_dataframes(self, results_instance):
        """Test that DataFrames are properly deep copied."""
        copied = copy.deepcopy(results_instance)

        # Verify DataFrames exist in copy
        assert copied.df_baseline is not None
        assert copied.df_pos_flex is not None
        assert copied.df_neg_flex is not None

        # Verify they are different objects
        assert copied.df_baseline is not results_instance.df_baseline

    def test_deepcopy_dataframe_independence(self, results_instance):
        """Test that modifying original doesn't affect copy."""
        copied = copy.deepcopy(results_instance)

        # Store original value
        if len(results_instance.df_baseline) > 0:
            # [0, 0] is nan due to collocation
            original_value = copied.df_baseline.iloc[1, 0]

            # Modify original
            results_instance.df_baseline.iloc[1, 0] = -999999

            # Copy should be unchanged
            assert copied.df_baseline.iloc[1, 0] == original_value

    def test_deepcopy_preserves_configs(self, results_instance):
        """Test that configs are preserved in deep copy."""
        copied = copy.deepcopy(results_instance)

        assert copied.flex_config is not None
        assert copied.baseline_agent_config is not None
        assert copied.baseline_module_config is not None

    def test_deepcopy_handles_simulator_module_config(self, results_instance):
        """Test that simulator_module_config with bypassed validation is handled."""
        copied = copy.deepcopy(results_instance)

        if hasattr(results_instance, "simulator_module_config"):
            assert hasattr(copied, "simulator_module_config")
            assert copied.simulator_module_config is not results_instance.simulator_module_config
            # Check that bypassed fields metadata is preserved
            assert hasattr(copied.simulator_module_config, "_bypassed_fields")

    def test_deepcopy_preserves_timescale(self, results_instance):
        """Test that current_timescale_of_data is preserved."""
        results_instance.convert_timescale_of_dataframe_index("minutes")
        copied = copy.deepcopy(results_instance)
        assert copied.current_timescale_of_data == "minutes"


# =============================================================================
# Tests for _get_flexquant_mpc_module_type
# =============================================================================


class TestGetFlexquantMpcModuleType:
    """Tests for the _get_flexquant_mpc_module_type method."""

    def test_returns_string(self, results_instance):
        """Test that method returns a string module type."""
        module_type = results_instance._get_flexquant_mpc_module_type(
            results_instance.baseline_agent_config
        )
        assert isinstance(module_type, str)

    def test_raises_when_no_matching_type(self, results_instance):
        """Test that ModuleNotFoundError is raised when no matching type found."""
        mock_agent_config = MagicMock()
        mock_agent_config.id = "test_agent"
        mock_agent_config.modules = [{"type": "unknown_type"}]

        with pytest.raises(ModuleNotFoundError, match="no matching mpc module type"):
            results_instance._get_flexquant_mpc_module_type(mock_agent_config)


# =============================================================================
# Tests for agent and module configs loading
# =============================================================================


class TestAgentModuleConfigs:
    """Tests for agent and module configuration loading."""

    def test_baseline_configs_loaded(self, results_instance):
        """Test that baseline agent and module configs are loaded."""
        assert results_instance.baseline_agent_config is not None
        assert results_instance.baseline_module_config is not None

    def test_pos_flex_configs_loaded(self, results_instance):
        """Test that positive flexibility agent and module configs are loaded."""
        assert results_instance.pos_flex_agent_config is not None
        assert results_instance.pos_flex_module_config is not None

    def test_neg_flex_configs_loaded(self, results_instance):
        """Test that negative flexibility agent and module configs are loaded."""
        assert results_instance.neg_flex_agent_config is not None
        assert results_instance.neg_flex_module_config is not None

    def test_indicator_configs_loaded(self, results_instance):
        """Test that indicator agent and module configs are loaded."""
        assert results_instance.indicator_agent_config is not None
        assert results_instance.indicator_module_config is not None

    def test_market_configs_loaded_when_present(self, results_instance):
        """Test that market configs are loaded when market_config is present."""
        if results_instance.flex_config.market_config:
            assert results_instance.market_agent_config is not None
            assert results_instance.market_module_config is not None


# =============================================================================
# Integration tests
# =============================================================================


class TestResultsIntegration:
    """Integration tests for the Results class."""

    def test_full_initialization_workflow(
        self, flex_config_path, simulator_config_path
    ):
        """Test complete initialization with all components."""
        results = Results(
            flex_config=flex_config_path,
            simulator_agent_config=simulator_config_path,
            generated_flex_files_base_path=SAMPLE_FILES_DIR,
            results=RESULTS_DIR,
            to_timescale="seconds",
        )

        # Verify all major components are initialized
        assert results.flex_config is not None
        assert results.df_baseline is not None
        assert results.df_pos_flex is not None
        assert results.df_neg_flex is not None
        assert results.df_indicator is not None
        assert results.df_baseline_stats is not None
        assert results.current_timescale_of_data == "seconds"

    def test_initialization_without_optional_components(self, flex_config_dict):
        """Test initialization without market and simulator."""
        flex_config_dict["market_config"] = None

        results = Results(
            flex_config=flex_config_dict,
            simulator_agent_config=None,
            generated_flex_files_base_path=SAMPLE_FILES_DIR,
            results=RESULTS_DIR,
            to_timescale="seconds",
        )

        assert results.df_baseline is not None
        assert results.df_market is None

    def test_copy_and_modify_workflow(self, results_instance):
        """Test copying results and modifying independently."""
        # Create a copy
        copied = Results(
            flex_config=None,
            simulator_agent_config=None,
            results=results_instance,
        )

        # Modify copy's timescale
        copied.convert_timescale_of_dataframe_index("hours")

        # Original should be unchanged
        assert results_instance.current_timescale_of_data == "seconds"
        assert copied.current_timescale_of_data == "hours"

    def test_timescale_conversion_chain(self, results_instance):
        """Test converting timescale through all options."""
        # Start with seconds
        assert results_instance.current_timescale_of_data == "seconds"

        # Convert to minutes
        results_instance.convert_timescale_of_dataframe_index("minutes")
        assert results_instance.current_timescale_of_data == "minutes"

        # Convert to hours
        results_instance.convert_timescale_of_dataframe_index("hours")
        assert results_instance.current_timescale_of_data == "hours"

        # Convert back to seconds
        results_instance.convert_timescale_of_dataframe_index("seconds")
        assert results_instance.current_timescale_of_data == "seconds"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
