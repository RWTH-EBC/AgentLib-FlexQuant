"""
Tests for the Dashboard class.

Run with: pytest test_dashboard.py -v
"""
import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from dash import Dash
from agentlib_flexquant.utils.interactive import Dashboard, CustomBound


class TestCustomBound:
    """Tests for the CustomBound class."""

    def test_init_with_all_params(self):
        """Test CustomBound initialization with all parameters."""
        bound = CustomBound("T", "T_lower", "T_upper")
        assert bound.for_variable == "T"
        assert bound.lower_bound == "T_lower"
        assert bound.upper_bound == "T_upper"

    def test_init_with_defaults(self):
        """Test CustomBound initialization with default values."""
        bound = CustomBound("T")
        assert bound.for_variable == "T"
        assert bound.lower_bound is None
        assert bound.upper_bound is None

    def test_init_with_only_lower_bound(self):
        """Test CustomBound initialization with only lower bound."""
        bound = CustomBound("T", lb_name="T_min")
        assert bound.for_variable == "T"
        assert bound.lower_bound == "T_min"
        assert bound.upper_bound is None

    def test_init_with_only_upper_bound(self):
        """Test CustomBound initialization with only upper bound."""
        bound = CustomBound("T", ub_name="T_max")
        assert bound.for_variable == "T"
        assert bound.lower_bound is None
        assert bound.upper_bound == "T_max"


@pytest.fixture
def mock_dashboard():
    """Create a mock Dashboard instance for testing."""
    with patch.object(Dashboard, '__init__', lambda self, **kwargs: None):
        dashboard = Dashboard.__new__(Dashboard)

        # Setup minimal required state
        dashboard.current_timescale_of_data = "hours"
        dashboard.current_timescale_input = "hours"
        dashboard.port = None
        dashboard.custom_bounds = []

        # Mock agent configs
        dashboard.baseline_agent_config = MagicMock(id="baseline")
        dashboard.pos_flex_agent_config = MagicMock(id="pos_flex")
        dashboard.neg_flex_agent_config = MagicMock(id="neg_flex")
        dashboard.simulator_agent_config = MagicMock(id="simulator")

        # Mock module configs
        dashboard.baseline_module_config = MagicMock(
            module_id="baseline_mpc",
            controls=[],
            time_step=3600,
            prediction_horizon=24,
            optimization_backend={"discretization_options": {"method": "collocation"}},
        )
        dashboard.pos_flex_module_config = MagicMock(module_id="pos_mpc")
        dashboard.neg_flex_module_config = MagicMock(module_id="neg_mpc")
        dashboard.simulator_module_config = MagicMock(module_id="sim")

        # Create mock dataframes with proper structure
        time_index = pd.MultiIndex.from_product(
            [[0, 1, 2], [0, 1, 2, 3]], names=["time", "step"]
        )

        dashboard.df_baseline = pd.DataFrame(
            np.random.rand(12, 2),
            index=time_index,
            columns=pd.MultiIndex.from_tuples(
                [("variable", "T"), ("variable", "power")]
            ),
        )
        dashboard.df_pos_flex = dashboard.df_baseline.copy()
        dashboard.df_neg_flex = dashboard.df_baseline.copy()

        dashboard.df_baseline_stats = pd.DataFrame(
            {"iter_count": [5, 6, 7]}, index=[0, 1, 2]
        )
        dashboard.df_pos_flex_stats = dashboard.df_baseline_stats.copy()
        dashboard.df_neg_flex_stats = dashboard.df_baseline_stats.copy()

        dashboard.df_simulation = pd.DataFrame(
            {"T": [20, 21, 22, 23], "power": [100, 110, 105, 115]}, index=[0, 1, 2, 3]
        )

        dashboard.df_indicator = pd.DataFrame(
            np.random.rand(12, 2),
            index=time_index,
            columns=["energy_flex_positive", "energy_flex_negative"],
        )

        market_index = pd.MultiIndex.from_product(
            [[0, 1], [0]], names=["time", "step"]
        )
        dashboard.df_market = pd.DataFrame(
            {"status": ["pending", "pending"], "pos_price": [10, 12], "neg_price": [8, 9]},
            index=market_index,
        )

        dashboard.intersection_mpcs_sim = {
            "T": {
                "baseline_mpc": "T",
                "pos_mpc": "T",
                "neg_mpc": "T",
                "sim": "T",
            }
        }

        dashboard.plotting_variables = ["iter_count", "T"]
        dashboard.kpi_names_pos = {"energy_flex": "energy_flex_positive"}
        dashboard.kpi_names_neg = {"energy_flex": "energy_flex_negative"}

        dashboard.LINE_PROPERTIES = {
            "baseline": {"color": "black"},
            "pos_flex": {"color": "blue"},
            "neg_flex": {"color": "red"},
            "simulator": {"color": "black"},
            "bounds": {"color": "grey"},
            "characteristic_times_current": {"color": "grey", "dash": "dash"},
            "characteristic_times_accepted": {"color": "yellow"},
        }
        dashboard.bounds_key = "bounds"
        dashboard.characteristic_times_current_key = "characteristic_times_current"
        dashboard.characteristic_times_accepted_key = "characteristic_times_accepted"
        dashboard.label_positive = "positive"
        dashboard.label_negative = "negative"
        dashboard.MPC_ITERATIONS = "iter_count"

        # Mock methods that may be called
        dashboard.convert_timescale_of_dataframe_index = MagicMock()
        dashboard.get_intersection_mpcs_sim = MagicMock(
            return_value=dashboard.intersection_mpcs_sim
        )

        return dashboard


class TestDashboardCreateApp:
    """This test module verifies the Dashboard functionality without running the actual
        Dash server. It uses pytest fixtures and mocking to:

        1. Create a mock Dashboard instance with simulated data (dataframes, configs, etc.)
           that bypasses the actual initialization which requires real data files.

        2. Test the separation of app creation from app running:
           - `create_app()` returns a configured Dash app without blocking
           - `show()` calls `create_app()` internally and then runs the server

        3. Verify component behavior:
           - CustomBound initialization with various parameter combinations
           - Layout creation with all required UI components (sliders, checkboxes, dropdowns)
           - Callback registration for interactivity
           - Plotting helper methods add correct traces/shapes to figures

        4. Test edge cases:
           - Single vs. list of CustomBounds
           - None vs. provided port numbers
           - Multiple app creation calls

        The mocking approach allows testing the Dashboard logic without dependencies on
        external data files, FMU models, or a running web server. Tests validate that
        the refactored structure (separating `create_app` from `show`) works correctly
        and enables testability.
    """

    def test_create_app_returns_dash_instance(self, mock_dashboard):
        """Test that create_app returns a Dash app instance."""
        app = mock_dashboard.create_app()
        assert isinstance(app, Dash)

    def test_create_app_sets_empty_custom_bounds_when_none(self, mock_dashboard):
        """Test that custom_bounds is set to empty list when None is passed."""
        mock_dashboard.create_app(custom_bounds=None)
        assert mock_dashboard.custom_bounds == []

    def test_create_app_wraps_single_custom_bound_in_list(self, mock_dashboard):
        """Test that a single CustomBound is wrapped in a list."""
        bound = CustomBound("T", "T_lower", "T_upper")
        mock_dashboard.create_app(custom_bounds=bound)
        assert len(mock_dashboard.custom_bounds) == 1
        assert mock_dashboard.custom_bounds[0] is bound

    def test_create_app_accepts_list_of_custom_bounds(self, mock_dashboard):
        """Test that a list of CustomBounds is accepted."""
        bounds = [
            CustomBound("T", "T_lower", "T_upper"),
            CustomBound("P", "P_lower", "P_upper"),
        ]
        mock_dashboard.create_app(custom_bounds=bounds)
        assert mock_dashboard.custom_bounds == bounds
        assert len(mock_dashboard.custom_bounds) == 2

    def test_create_app_has_layout(self, mock_dashboard):
        """Test that the created app has a layout."""
        app = mock_dashboard.create_app()
        assert app.layout is not None

    def test_create_app_layout_contains_required_components(self, mock_dashboard):
        """Test that the layout contains required component IDs."""
        app = mock_dashboard.create_app()

        # Convert layout to string to check for component IDs
        layout_str = str(app.layout)

        assert "time_slider" in layout_str
        assert "time_typing" in layout_str
        assert "time_unit" in layout_str
        assert "graphs_container_variables" in layout_str
        assert "accepted_characteristic_times" in layout_str
        assert "current_characteristic_times" in layout_str
        assert "zoom_to_offer_window" in layout_str
        assert "zoom_to_prediction_interval" in layout_str


class TestDashboardShow:
    """Tests for the Dashboard.show() method."""

    def test_show_does_not_block_when_mocked(self, mock_dashboard):
        """Test that show() can complete when app.run() is mocked."""
        with patch("webbrowser.open_new_tab") as mock_browser, patch.object(
            Dash, "run"
        ) as mock_run:
            mock_dashboard.port = 8050
            mock_dashboard.show()

            mock_browser.assert_called_once_with("http://localhost:8050")
            mock_run.assert_called_once_with(debug=False, port=8050)

    def test_show_uses_provided_port(self, mock_dashboard):
        """Test that show() uses the provided port."""
        mock_dashboard.port = 9999

        with patch("webbrowser.open_new_tab") as mock_browser, patch.object(
            Dash, "run"
        ) as mock_run:
            mock_dashboard.show()

            mock_browser.assert_called_once_with("http://localhost:9999")
            mock_run.assert_called_once_with(debug=False, port=9999)

    def test_show_gets_port_when_not_provided(self, mock_dashboard):
        """Test that show() gets a port when none is provided."""
        mock_dashboard.port = None

        with patch("webbrowser.open_new_tab"), patch.object(
            Dash, "run"
        ) as mock_run, patch(
            "agentlib_flexquant.utils.interactive.get_port", return_value=8888
        ):
            mock_dashboard.show()

            mock_run.assert_called_once_with(debug=False, port=8888)

    def test_show_with_custom_bounds(self, mock_dashboard):
        """Test show() with custom bounds."""
        with patch("webbrowser.open_new_tab"), patch.object(Dash, "run"):
            mock_dashboard.port = 8050
            bound = CustomBound("T", "T_lower", "T_upper")
            mock_dashboard.show(custom_bounds=bound)

            assert len(mock_dashboard.custom_bounds) == 1
            assert mock_dashboard.custom_bounds[0].for_variable == "T"

    def test_show_calls_create_app(self, mock_dashboard):
        """Test that show() calls create_app() internally."""
        with patch("webbrowser.open_new_tab"), patch.object(Dash, "run"), patch.object(
            mock_dashboard, "create_app", wraps=mock_dashboard.create_app
        ) as mock_create:
            mock_dashboard.port = 8050
            mock_dashboard.show()

            mock_create.assert_called_once()


class TestDashboardLayout:
    """Tests for the Dashboard layout creation."""

    def test_create_layout_returns_list(self, mock_dashboard):
        """Test that _create_layout returns a list."""
        layout = mock_dashboard._create_layout()
        assert isinstance(layout, list)

    def test_create_layout_has_results_header(self, mock_dashboard):
        """Test that the layout has a Results header."""
        layout = mock_dashboard._create_layout()

        # First element should be H1 with "Results"
        from dash import html

        assert isinstance(layout[0], html.H1)
        assert layout[0].children == "Results"


class TestDashboardCallbacks:
    """Tests for Dashboard callbacks registration."""

    def test_callbacks_are_registered(self, mock_dashboard):
        """Test that callbacks are registered on the app."""
        app = mock_dashboard.create_app()

        # Check that callbacks were registered by checking the callback_map
        assert len(app.callback_map) > 0

    def test_time_slider_callback_registered(self, mock_dashboard):
        """Test that the time slider callback is registered."""
        app = mock_dashboard.create_app()

        # Look for the time_slider output in callback map
        callback_outputs = [str(key) for key in app.callback_map.keys()]
        time_slider_registered = any("time_slider" in output for output in callback_outputs)
        assert time_slider_registered

    def test_graphs_container_callback_registered(self, mock_dashboard):
        """Test that the graphs container callback is registered."""
        app = mock_dashboard.create_app()

        callback_outputs = [str(key) for key in app.callback_map.keys()]
        graphs_registered = any(
            "graphs_container_variables" in output for output in callback_outputs
        )
        assert graphs_registered


class TestDashboardPlottingMethods:
    """Tests for Dashboard plotting helper methods."""

    def test_mark_time_adds_vline(self, mock_dashboard):
        """Test that _mark_time adds a vertical line to the figure."""
        from plotly import graph_objects as go

        fig = go.Figure()
        mock_dashboard._mark_time(fig, at_time_step=1.0, line_prop={"color": "green"})

        # Check that a shape (vline) was added
        assert len(fig.layout.shapes) == 1

    def test_plot_mpc_stats_adds_traces(self, mock_dashboard):
        """Test that _plot_mpc_stats adds traces to the figure."""
        from plotly import graph_objects as go

        fig = go.Figure()
        mock_dashboard._plot_mpc_stats(fig, variable="iter_count")

        # Should add 3 traces (baseline, pos_flex, neg_flex)
        assert len(fig.data) == 3


class TestDashboardIntegration:
    """Integration tests for the Dashboard."""

    def test_full_app_creation_workflow(self, mock_dashboard):
        """Test the complete app creation workflow."""
        # Create app with custom bounds
        bounds = [
            CustomBound("T", "T_lower", "T_upper"),
        ]
        app = mock_dashboard.create_app(custom_bounds=bounds)

        # Verify app is properly configured
        assert isinstance(app, Dash)
        assert app.layout is not None
        assert len(app.callback_map) > 0
        assert mock_dashboard.custom_bounds == bounds

    def test_app_can_be_created_multiple_times(self, mock_dashboard):
        """Test that create_app can be called multiple times."""
        app1 = mock_dashboard.create_app()
        app2 = mock_dashboard.create_app(custom_bounds=CustomBound("T"))

        assert isinstance(app1, Dash)
        assert isinstance(app2, Dash)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
