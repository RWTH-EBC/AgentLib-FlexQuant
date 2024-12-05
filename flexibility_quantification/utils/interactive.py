from typing import get_args, Union
from pydantic import FilePath
import pandas as pd

import webbrowser
from dash import Dash, html, dcc, callback, Output, Input, ctx
from plotly import graph_objects as go

from agentlib.core.agent import AgentConfig

from agentlib_mpc.utils import TimeConversionTypes, TIME_CONVERSION
from agentlib_mpc.utils.analysis import mpc_at_time_step
from agentlib_mpc.utils.plotting.interactive import get_port    # solver_return, obj_plot  -> didn't work out for stats

from flexibility_quantification.data_structures.flexquant import FlexQuantConfig
import flexibility_quantification.data_structures.globals as glbs
import flexibility_quantification.data_structures.results as flex_results


class Dashboard(flex_results.Results):
    """
    Class for the dashboard of flexquant
    """

    # Constants for plotting variables
    MPC_ITERATIONS: str = "iter_count"
    # todo: get names from the kpis (after merge)
    energyflex: str = "energy_flexibility"
    flex_price: str = "costs"
    market_price: str = "price"

    # Label for the positive and negative flexibilities
    # todo: get names from the directions (after merge)
    label_positive: str = "positive"
    label_negative: str = "negative"

    # Keys for line properties
    bounds_key: str = "bounds"
    characteristic_times_current_key: str = "characteristic_times_current"
    characteristic_times_accepted_key: str = "characteristic_times_accepted"

    def __init__(
            self,
            flex_config: Union[str, FilePath, FlexQuantConfig],
            simulator_agent_config: Union[str, FilePath, AgentConfig],
            results: Union[str, FilePath, dict[str, dict[str, pd.DataFrame]]] = None,
            to_timescale: TimeConversionTypes = "hours"
    ):
        super().__init__(flex_config=flex_config, simulator_agent_config=simulator_agent_config, results=results, to_timescale=to_timescale)
        self.current_timescale_input = self.current_timescale_of_data
        self.intersection_mpcs_sim = self.get_intersection_mpcs_sim()

        # Define line properties
        self.LINE_PROPERTIES: dict = {
            self.simulator_agent_config.id: {
                "color": "black",
            },
            self.baseline_agent_config.id: {
                "color": "black",
            },
            self.neg_flex_agent_config.id: {
                "color": "red",
            },
            self.pos_flex_agent_config.id: {
                "color": "blue",
            },
            self.bounds_key: {
                "color": "grey",
            },
            self.characteristic_times_current_key: {
                "color": "grey",
                "dash": "dash",
            },
            self.characteristic_times_accepted_key: {
                "color": "yellow",
            },
        }

    def show(self, temperature_var_name: str = None, ub_comfort_var_name: str = None, lb_comfort_var_name: str = None):
        """
        Shows the dashboard in a web browser containing:
        -- Statistics of the MPCs solver
        -- The states, controls, and the power variable of the MPCs and the simulator
        -- KPIs of the flexibility quantification
        -- Markings of the characteristic flexibility times

        Optional arguments to show the comfort bounds:
        -- temperature_var_name: The name of the temperature variable in the MPC to plot the comfort bounds into
        -- ub_comfort_var_name: The name of the upper comfort bound variable in the MPC
        -- lb_comfort_var_name: The name of the lower comfort bound variable in the MPC
        """

        # Plotting functions
        def mark_characteristic_times(fig: go.Figure, at_time_step: float, line_prop: dict = None) -> go.Figure:
            """
            Add markers of the characteristic times to the plot for a time step

            Keyword arguments:
            fig -- The figure to plot the results into
            time_step -- When to show the markers
            line_prop -- The graphic properties of the lines as in plotly
            """
            if line_prop is None:
                line_prop = self.LINE_PROPERTIES[self.characteristic_times_current_key]
            try:
                df_characteristic_times = self.df_indicator.xs(0, level="time")

                offer_time = at_time_step
                rel_market_time = df_characteristic_times.loc[at_time_step, glbs.MARKET_TIME] / TIME_CONVERSION[self.current_timescale_of_data]
                rel_prep_time = df_characteristic_times.loc[at_time_step, glbs.PREP_TIME] / TIME_CONVERSION[self.current_timescale_of_data]
                flex_event_duration = df_characteristic_times.loc[at_time_step, glbs.FLEX_EVENT_DURATION] / TIME_CONVERSION[self.current_timescale_of_data]

                fig.add_vline(x=offer_time, line=line_prop, layer="below")
                fig.add_vline(x=offer_time + rel_prep_time, line=line_prop, layer="below")
                fig.add_vline(x=offer_time + rel_prep_time + rel_market_time, line=line_prop, layer="below")
                fig.add_vline(x=offer_time + rel_prep_time + rel_market_time + flex_event_duration, line=line_prop, layer="below")
            except KeyError:
                pass  # No data of characteristic times available, e.g. if offer accepted
            return fig

        def mark_characteristic_times_of_accepted_offers(fig: go.Figure) -> go.Figure:
            """ Add markers of the characteristic times for accepted offers to the plot
            """
            df_accepted_offers = self.df_market["status"].str.contains(pat="OfferStatus.accepted")
            for i in df_accepted_offers.index.to_list():
                if df_accepted_offers[i]:
                    fig = mark_characteristic_times(fig=fig, at_time_step=i[0], line_prop=self.LINE_PROPERTIES[self.characteristic_times_accepted_key])
            return fig

        def plot_one_mpc_variable(fig: go.Figure, variable: str, time_step: float) -> go.Figure:
            # Get bounds
            def _get_bound(var_type: str, var_name: str):
                return self.df_baseline[(var_type, var_name)].xs(0, level=1)

            if variable == temperature_var_name:
                if lb_comfort_var_name is None:
                    df_lb = None
                else:
                    try:
                        df_lb = _get_bound(var_type="variable", var_name=lb_comfort_var_name)
                    except KeyError:
                        df_lb = _get_bound(var_type="parameter", var_name=lb_comfort_var_name)
                if ub_comfort_var_name is None:
                    df_ub = None
                else:
                    try:
                        df_ub = _get_bound(var_type="variable", var_name=ub_comfort_var_name)
                    except KeyError:
                        df_ub = _get_bound(var_type="parameter", var_name=ub_comfort_var_name)
            elif variable in [control.name for control in self.baseline_module_config.controls]:
                df_lb = _get_bound(var_type="lower", var_name=variable)
                df_ub = _get_bound(var_type="upper", var_name=variable)
            else:
                df_lb = None
                df_ub = None

            # Plot bounds
            if df_lb is not None:
                fig.add_trace(go.Scatter(name="Lower bound", x=df_lb.index, y=df_lb, mode="lines", line=self.LINE_PROPERTIES[self.bounds_key], zorder=1))
            if df_ub is not None:
                fig.add_trace(go.Scatter(name="Upper bound", x=df_ub.index, y=df_ub, mode="lines", line=self.LINE_PROPERTIES[self.bounds_key], zorder=1))

            # Get the data for the plot
            df_sim = self.df_simulation[self.intersection_mpcs_sim[variable][self.simulator_agent_config.id]]
            df_neg = mpc_at_time_step(data=self.df_neg_flex, time_step=time_step, variable=self.intersection_mpcs_sim[variable][self.neg_flex_agent_config.id], index_offset=False).dropna()
            df_pos = mpc_at_time_step(data=self.df_pos_flex, time_step=time_step, variable=self.intersection_mpcs_sim[variable][self.pos_flex_agent_config.id], index_offset=False).dropna()
            df_bas = mpc_at_time_step(data=self.df_baseline, time_step=time_step, variable=self.intersection_mpcs_sim[variable][self.baseline_agent_config.id], index_offset=False).dropna()

            # Plot the data
            try:
                fig.add_trace(go.Scatter(name=self.simulator_agent_config.id, x=df_sim.index, y=df_sim, mode="lines", line=self.LINE_PROPERTIES[self.simulator_agent_config.id], zorder=2))
            except KeyError:
                pass    # When the simulator variable name was not found from the intersection
            fig.add_trace(go.Scatter(name=self.baseline_agent_config.id, x=df_bas.index, y=df_bas, mode="lines", line=self.LINE_PROPERTIES[self.baseline_agent_config.id] | {"dash": "dash"}, zorder=3))
            fig.add_trace(go.Scatter(name=self.neg_flex_agent_config.id, x=df_neg.index, y=df_neg, mode="lines", line=self.LINE_PROPERTIES[self.neg_flex_agent_config.id] | {"dash": "dash"}, zorder=4))
            fig.add_trace(go.Scatter(name=self.pos_flex_agent_config.id, x=df_pos.index, y=df_pos, mode="lines", line=self.LINE_PROPERTIES[self.pos_flex_agent_config.id] | {"dash": "dash"}, zorder=4))

            return fig

        def plot_mpc_stats(fig: go.Figure, variable: str) -> go.Figure:
            fig.add_trace(go.Scatter(name=self.baseline_agent_config.id, x=self.df_baseline_stats.index, y=self.df_baseline_stats[variable], mode="markers", line=self.LINE_PROPERTIES[self.baseline_agent_config.id]))
            fig.add_trace(go.Scatter(name=self.pos_flex_agent_config.id, x=self.df_pos_flex_stats.index, y=self.df_pos_flex_stats[variable], mode="markers", line=self.LINE_PROPERTIES[self.pos_flex_agent_config.id]))
            fig.add_trace(go.Scatter(name=self.neg_flex_agent_config.id, x=self.df_neg_flex_stats.index, y=self.df_neg_flex_stats[variable], mode="markers", line=self.LINE_PROPERTIES[self.neg_flex_agent_config.id]))
            return fig

        # todo: generic kpi plot function
        def plot_flexibility_kpi(fig: go.Figure, variable=None) -> go.Figure:
            df_ind = self.df_indicator.xs(0, level=1)
            fig.add_trace(go.Scatter(name=self.label_positive, x=df_ind.index, y=df_ind["energyflex_pos"], mode="lines+markers", line=self.LINE_PROPERTIES[self.pos_flex_agent_config.id]))
            fig.add_trace(go.Scatter(name=self.label_negative, x=df_ind.index, y=df_ind["energyflex_neg"], mode="lines+markers", line=self.LINE_PROPERTIES[self.neg_flex_agent_config.id]))
            return fig

        def plot_flexibility_price(fig: go.Figure, variable=None) -> go.Figure:
            df_ind = self.df_indicator.xs(0, level=1)
            fig.add_trace(go.Scatter(name=self.label_positive, x=df_ind.index, y=df_ind["costs_pos"], mode="lines+markers", line=self.LINE_PROPERTIES[self.pos_flex_agent_config.id]))
            fig.add_trace(go.Scatter(name=self.label_negative, x=df_ind.index, y=df_ind["costs_neg"], mode="lines+markers", line=self.LINE_PROPERTIES[self.neg_flex_agent_config.id]))
            return fig

        def plot_market_results(fig: go.Figure, variable: str) -> go.Figure:
            df_flex_market_index = self.df_market.index.droplevel("time")
            if variable in self.df_market.columns:
                fig.add_trace(go.Scatter(name=self.label_positive, x=df_flex_market_index, y=self.df_market[variable], mode="lines+markers", line=self.LINE_PROPERTIES[self.pos_flex_agent_config.id]))
            else:
                # todo: solve generic
                pos_var = f"pos_{variable}"
                neg_var = f"neg_{variable}"
                fig.add_trace(go.Scatter(name=self.label_positive, x=df_flex_market_index, y=self.df_market[pos_var], mode="lines+markers", line=self.LINE_PROPERTIES[self.pos_flex_agent_config.id]))
                fig.add_trace(go.Scatter(name=self.label_negative, x=df_flex_market_index, y=self.df_market[neg_var], mode="lines+markers", line=self.LINE_PROPERTIES[self.neg_flex_agent_config.id]))
            return fig

        def create_plot_for_one_variable(variable: str, at_time_step: float, show_current_characteristic_times: bool, zoom_to_prediction_interval: bool = False) -> go.Figure:
            """
            Create a plot for one variable

            Keyword arguments:
            variable -- The variable to plot
            time_step -- The time_step to show the mpc predictions and the characteristic times
            show_current_characteristic_times -- Whether to show the characteristic times
            """
            # Create the figure
            fig = go.Figure()

            # Plot variable
            if variable in self.df_baseline_stats.columns:
                plot_mpc_stats(fig=fig, variable=variable)
            elif variable in self.intersection_mpcs_sim.keys():
                plot_one_mpc_variable(fig=fig, variable=variable, time_step=at_time_step)
            elif variable == self.energyflex:   # todo: generic kpi function
                plot_flexibility_kpi(fig=fig)
            elif any(variable in label for label in self.df_indicator.columns):
                plot_flexibility_price(fig=fig)
            elif any(variable in label for label in self.df_market.columns):
                plot_market_results(fig=fig, variable=variable)
            else:
                raise ValueError(f"No plotting function found for variable {variable}")

            # Plot characteristic times
            mark_characteristic_times_of_accepted_offers(fig=fig)
            if show_current_characteristic_times and variable in self.intersection_mpcs_sim.keys():
                mark_characteristic_times(fig=fig, at_time_step=at_time_step)

            # Set layout
            if zoom_to_prediction_interval:
                xlim_left = at_time_step
                xlim_right = at_time_step + self.df_baseline.index[-1][-1]
            else:
                xlim_left = self.df_simulation.index[0]
                xlim_right = self.df_simulation.index[-1] + self.df_baseline.index[-1][-1]

            fig.update_layout(yaxis_title=variable,
                              xaxis_title=f"Time in {self.current_timescale_of_data}",
                              xaxis_range=[xlim_left, xlim_right],
                              height=350, margin=dict(t=20, b=20))
            fig.update_xaxes(dtick=round(self.baseline_module_config.prediction_horizon / 6) * self.baseline_module_config.time_step / TIME_CONVERSION[self.current_timescale_of_data])
            return fig

        def get_variables_for_plotting() -> list[str]:
            # MPC stats
            variables = [self.MPC_ITERATIONS]

            # MPC and sim variables
            variables.extend([key for key in self.intersection_mpcs_sim.keys()])

            # Flexibility kpis
            # todo: get names from the kpis (after merge)
            variables.append(self.energyflex)
            variables.append(self.flex_price)
            variables.append(self.market_price)

            return variables

        # Create the app
        app = Dash(__name__, title="Results")
        app.layout = [
            html.H1("Results"),
            html.Div(
                children=[
                    # Options
                    html.Div(
                        children=[
                            html.H3("Options"),
                            dcc.Checklist(id="current_characteristic_times",
                                          options=[{"label": "Show characteristic times (current)", "value": True}],
                                          value=[True],
                                          style={"display": "inline-block", "padding-right": "10px"}),
                            dcc.Checklist(id="zoom_to_prediction_interval",
                                          options=[{"label": "Zoom to mpc prediction interval", "value": False}],
                                          style={"display": "inline-block"}),
                            # dcc.Dropdown(id="additional_variables",
                            #              options=[], multi=True,
                            #              placeholder="Additional variables"),
                        ],
                    ),
                    # Time input
                    html.Div(
                        children=[
                            html.H3(
                                children=f"Time:",
                                style={"display": "inline-block", "padding-right": "10px"}),
                            dcc.Input(
                                id="time_typing", type="number",
                                min=0, max=1, value=0,  # will be updated in the callback
                                style={"display": "inline-block"}),
                            dcc.Dropdown(
                                id="time_unit",
                                options=get_args(TimeConversionTypes),
                                value=self.current_timescale_input,
                                style={"display": "inline-block", "verticalAlign": "middle", "padding-left": "10px", "width": "100px"}),
                        ],
                    ),
                    dcc.Slider(id="time_slider",
                               min=0, max=1, value=0,   # will be updated in the callback
                               tooltip={"placement": "bottom", "always_visible": True},
                               marks=None,
                               updatemode="drag")
                ], style={
                    "width": "88%", "padding-left": "0%", "padding-right": "12%",
                    # Make the options sticky to the top of the page
                    "position": "sticky", "top": "0", "overflow-y": "visible", "z-index": "100", "background-color": "white"
                }
            ),
            # Container for the graphs, will be updated in the callback
            html.Div(id="graphs_container_variables", children=[]),
        ]

        # Callbacks
        # Update the time value or the time unit
        @callback(
            Output(component_id="time_slider", component_property="value"),
            Output(component_id="time_slider", component_property="min"),
            Output(component_id="time_slider", component_property="max"),
            Output(component_id="time_slider", component_property="step"),

            Output(component_id="time_typing", component_property="value"),
            Output(component_id="time_typing", component_property="min"),
            Output(component_id="time_typing", component_property="max"),
            Output(component_id="time_typing", component_property="step"),

            Input(component_id="time_typing", component_property="value"),
            Input(component_id="time_slider", component_property="value"),
            Input(component_id="time_unit", component_property="value")
        )
        def update_time_index_of_input(time_typing: float, time_slider: float, time_unit: TimeConversionTypes) -> [float]:
            # get trigger id
            trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

            # Get the value for the sliders
            if trigger_id == "time_slider":
                value = time_slider
            elif trigger_id == "time_unit":
                value = time_typing * TIME_CONVERSION[self.current_timescale_input] / TIME_CONVERSION[time_unit]
            else:
                value = time_typing

            # Convert the index to the given time unit if necessary
            if trigger_id == "time_unit":
                self.convert_timescale_of_dataframe_index(to_timescale=time_unit)

            # Get the index for the slider types
            times = self.df_baseline.index.get_level_values(0).unique()
            minimum = times[0]
            maximum = times[-1]
            step = times[1] - times[0]

            self.current_timescale_input = time_unit

            return (value, minimum, maximum, step,
                    value, minimum, maximum, step)

        # Update the graphs
        @callback(
            Output(component_id="graphs_container_variables", component_property="children"),
            Input(component_id="current_characteristic_times", component_property="value"),
            Input(component_id="zoom_to_prediction_interval", component_property="value"),
            Input(component_id="time_typing", component_property="value")
        )
        def update_graph(current_characteristic_times: bool, zoom_to_prediction_interval: bool, time_step: float):
            """ Update all graphs based on the options and slider values
            """
            figs = []
            for variable in get_variables_for_plotting():
                fig = create_plot_for_one_variable(
                    variable=variable,
                    at_time_step=time_step,
                    show_current_characteristic_times=current_characteristic_times,
                    zoom_to_prediction_interval=zoom_to_prediction_interval
                )
                figs.append(dcc.Graph(id=f"graph_{variable}", figure=fig))
            return figs

        # Run the app
        port = get_port()
        webbrowser.open_new_tab(f"http://localhost:{port}")
        app.run(debug=False, port=port)
