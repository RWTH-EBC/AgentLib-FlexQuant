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
    energyflex: str = "energy_flexibility"
    price: str = "flexibility_price"
    mpc_iterations: str = "mpc_iterations"

    # Label for the positive and negative flexibilities
    label_positive: str = "positive"
    label_negative: str = "negative"

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
            "bounds": {
                "color": "grey",
            },
            "characteristic_times_current": {
                "color": "grey",
                "dash": "dash",
            },
            "characteristic_times_accepted": {
                "color": "yellow",
            },
        }

    def show(self):

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
                line_prop = self.LINE_PROPERTIES["characteristic_times_current"]
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
                    fig = mark_characteristic_times(fig=fig, at_time_step=i[0], line_prop=self.LINE_PROPERTIES["characteristic_times_accepted"])
            return fig

        def plot_one_mpc_variable(fig: go.Figure, variable: str, time_step: float) -> go.Figure:
            # Plot bounds # todo
            # if variable in ["T", "T_out"]:
            #     # In the simulation the bounds set in the constraints doesn't affect the bounds of the state, so they need to be plotted manually
            #     df_lb = self.df_baseline[("parameter", "T_lower")].xs(0, level=1)
            #     fig.add_trace(go.Scatter(name="T_lower", x=df_lb.index, y=df_lb, mode="lines", line=self.LINE_PROPERTIES["bounds"]))
            #     df_ub = self.df_baseline[("parameter", "T_upper")].xs(0, level=1)
            #     fig.add_trace(go.Scatter(name="T_upper", x=df_ub.index, y=df_ub, mode="lines", line=self.LINE_PROPERTIES["bounds"]))
            # else:
            #     # Default case
            #     df_lb = self.df_baseline[("lower", variable)].xs(0, level=1)
            #     df_ub = self.df_baseline[("upper", variable)].xs(0, level=1)
            #     if df_lb.notna().all():
            #         fig.add_trace(go.Scatter(name="T_lower", x=df_lb.index, y=df_lb, mode="lines", line=self.LINE_PROPERTIES["bounds"]))
            #     if df_ub.notna().all():
            #         fig.add_trace(go.Scatter(name="T_upper", x=df_ub.index, y=df_ub, mode="lines", line=self.LINE_PROPERTIES["bounds"]))

            # Get the data for the plot
            df_sim = self.df_simulation[self.intersection_mpcs_sim[variable][self.simulator_agent_config.id]]
            df_neg = mpc_at_time_step(data=self.df_neg_flex, time_step=time_step, variable=self.intersection_mpcs_sim[variable][self.neg_flex_agent_config.id], index_offset=False).dropna()
            df_pos = mpc_at_time_step(data=self.df_pos_flex, time_step=time_step, variable=self.intersection_mpcs_sim[variable][self.pos_flex_agent_config.id], index_offset=False).dropna()
            df_bas = mpc_at_time_step(data=self.df_baseline, time_step=time_step, variable=self.intersection_mpcs_sim[variable][self.baseline_agent_config.id], index_offset=False).dropna()

            # Plot the data
            try:
                fig.add_trace(go.Scatter(name=self.simulator_agent_config.id, x=df_sim.index, y=df_sim, mode="lines", line=self.LINE_PROPERTIES[self.simulator_agent_config.id], zorder=0))
            except KeyError:
                pass    # When the simulator variable name was not found from the intersection
            fig.add_trace(go.Scatter(name=self.neg_flex_agent_config.id, x=df_neg.index, y=df_neg, mode="lines", line=self.LINE_PROPERTIES[self.neg_flex_agent_config.id] | {"dash": "dash"}, zorder=2))
            fig.add_trace(go.Scatter(name=self.pos_flex_agent_config.id, x=df_pos.index, y=df_pos, mode="lines", line=self.LINE_PROPERTIES[self.pos_flex_agent_config.id] | {"dash": "dash"}, zorder=2))
            fig.add_trace(go.Scatter(name=self.baseline_agent_config.id, x=df_bas.index, y=df_bas, mode="lines", line=self.LINE_PROPERTIES[self.baseline_agent_config.id] | {"dash": "dash"}, zorder=1))

            return fig

        def plot_mpc_iterations(fig: go.Figure) -> go.Figure:
            fig.add_trace(go.Scatter(name=self.baseline_agent_config.id, x=self.df_baseline_stats.index, y=self.df_baseline_stats["iter_count"], mode="markers", line=self.LINE_PROPERTIES[self.baseline_agent_config.id]))
            fig.add_trace(go.Scatter(name=self.pos_flex_agent_config.id, x=self.df_pos_flex_stats.index, y=self.df_pos_flex_stats["iter_count"], mode="markers", line=self.LINE_PROPERTIES[self.pos_flex_agent_config.id]))
            fig.add_trace(go.Scatter(name=self.neg_flex_agent_config.id, x=self.df_neg_flex_stats.index, y=self.df_neg_flex_stats["iter_count"], mode="markers", line=self.LINE_PROPERTIES[self.neg_flex_agent_config.id]))
            return fig

        # todo: generic kpi plot function
        def plot_energy_flexibility(fig: go.Figure) -> go.Figure:
            df_ind = self.df_indicator.xs(0, level=1)
            fig.add_trace(go.Scatter(name=self.label_positive, x=df_ind.index, y=df_ind[glbs.ENERGYFLEX_POS], mode="lines+markers", line=self.LINE_PROPERTIES[self.pos_flex_agent_config.id]))
            fig.add_trace(go.Scatter(name=self.label_negative, x=df_ind.index, y=df_ind[glbs.ENERGYFLEX_NEG], mode="lines+markers", line=self.LINE_PROPERTIES[self.neg_flex_agent_config.id]))
            return fig

        def plot_flexibility_prices(fig: go.Figure) -> go.Figure:
            df_flex_market_index = self.df_market.index.droplevel("time")
            fig.add_trace(go.Scatter(name=self.label_positive, x=df_flex_market_index, y=self.df_market["pos_price"], mode="lines+markers", line=self.LINE_PROPERTIES[self.pos_flex_agent_config.id]))
            fig.add_trace(go.Scatter(name=self.label_negative, x=df_flex_market_index, y=self.df_market["neg_price"], mode="lines+markers", line=self.LINE_PROPERTIES[self.neg_flex_agent_config.id]))
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
            if variable == self.mpc_iterations:
                plot_mpc_iterations(fig=fig)
            elif variable == self.energyflex:
                plot_energy_flexibility(fig=fig)
            elif variable == self.price:
                plot_flexibility_prices(fig=fig)
            elif variable in self.intersection_mpcs_sim.keys():
                plot_one_mpc_variable(fig=fig, variable=variable, time_step=at_time_step)
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
            return fig

        def get_variables_for_plotting() -> list[str]:
            variables = [key for key in self.intersection_mpcs_sim.keys()]

            # Add custom variables
            variables.append(self.energyflex)
            variables.append(self.price)
            variables.append(self.mpc_iterations)

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
