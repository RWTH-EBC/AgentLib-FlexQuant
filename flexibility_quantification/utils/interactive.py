from typing import Union
from pydantic import FilePath
import pandas as pd

import webbrowser
from dash import Dash, html, dcc, callback, Output, Input
from plotly import graph_objects as go

from agentlib.core.agent import AgentConfig

from agentlib_mpc.utils import TimeConversionTypes, TIME_CONVERSION
from agentlib_mpc.utils.analysis import mpc_at_time_step
from agentlib_mpc.utils.plotting.interactive import get_port    # solver_return, obj_plot  -> didn't work out for stats

from flexibility_quantification.data_structures.flexquant import FlexQuantConfig
import flexibility_quantification.data_structures.globals as glbs
import flexibility_quantification.data_structures.flexresults as flexresults


class FlexDashboard(flexresults.FlexResults):
    """
    Class for the dashboard of flexquant
    """

    # Constants for plotting variables
    energyflex: str = "energyflex"
    price: str = "price"
    mpc_iterations: str = "iterations"

    # Label for the positive and negative flexibilities
    label_positive: str = "positive"
    label_negative: str = "negative"

    timescale: TimeConversionTypes = "hours"  # todo: time convertion

    def __init__(
            self,
            flex_config: Union[str, FilePath, FlexQuantConfig],
            simulator_agent_config: Union[str, FilePath, AgentConfig],
            results: Union[str, FilePath, dict[str, dict[str, pd.DataFrame]]] = None
    ):
        super().__init__(flex_config=flex_config, simulator_agent_config=simulator_agent_config, results=results)
        # Define line properties
        self.LINE_PROPERTIES: dict = {
            self.simulator_agent_id: {
                "color": "black",
            },
            self.baseline_agent_id: {
                "color": "black",
            },
            self.neg_flex_agent_id: {
                "color": "red",
            },
            self.pos_flex_agent_id: {
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

        def mark_characteristic_times(fig: go.Figure, at_time_step: int, line_prop: dict = None) -> go.Figure:
            """Add markers of the characteristic times to the plot for a time step

            Keyword arguments:
            fig -- The figure to plot the results into
            time_step -- When to show the markers
            line_prop -- The graphic properties of the lines as in plotly
            """
            if line_prop is None:
                line_prop = self.LINE_PROPERTIES["characteristic_times_current"]
            try:
                df_characteristic_times = self.df_indicator.xs(0, level="time")
                df_characteristic_times.index = df_characteristic_times.index / TIME_CONVERSION[self.timescale]

                offer_time = at_time_step
                rel_market_time = df_characteristic_times.loc[at_time_step, glbs.MARKET_TIME] / TIME_CONVERSION[self.timescale]
                rel_prep_time = df_characteristic_times.loc[at_time_step, glbs.PREP_TIME] / TIME_CONVERSION[self.timescale]
                flex_event_duration = df_characteristic_times.loc[at_time_step, glbs.FLEX_EVENT_DURATION] / TIME_CONVERSION[self.timescale]

                fig.add_vline(x=offer_time, line=line_prop)
                fig.add_vline(x=offer_time + rel_prep_time, line=line_prop)
                fig.add_vline(x=offer_time + rel_prep_time + rel_market_time, line=line_prop)
                fig.add_vline(x=offer_time + rel_prep_time + rel_market_time + flex_event_duration, line=line_prop)
            except KeyError:
                pass  # No data of characteristic times available, e.g. if offer accepted
            return fig

        def mark_characteristic_times_of_accepted_offers(fig: go.Figure) -> go.Figure:
            """ Add markers of the characteristic times for accepted offers to the plot
            """
            df_accepted_offers = self.df_flex_market["status"].str.contains(pat="OfferStatus.accepted")
            for i in df_accepted_offers.index.to_list():
                if df_accepted_offers[i]:
                    fig = mark_characteristic_times(fig=fig, at_time_step=i[0] / TIME_CONVERSION[self.timescale], line_prop=self.LINE_PROPERTIES["characteristic_times_accepted"])
            return fig

        def plot_one_mpc_variable(fig: go.Figure, variable: str, time_step: int) -> go.Figure:
            """ Create a plot for the mpc variable

            Keyword arguments:
            fig -- The figure to plot the results into
            variable -- The variable to plot
            time_step -- The timestep from when the mpc predictions should be shown
            """
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
            df_sim = self.df_simulation[variable]
            df_neg = mpc_at_time_step(data=self.df_neg_flex, time_step=time_step * TIME_CONVERSION[self.timescale], variable=variable, index_offset=False).dropna()
            df_pos = mpc_at_time_step(data=self.df_pos_flex, time_step=time_step * TIME_CONVERSION[self.timescale], variable=variable, index_offset=False).dropna()
            df_bas = mpc_at_time_step(data=self.df_baseline, time_step=time_step * TIME_CONVERSION[self.timescale], variable=variable, index_offset=False).dropna()

            # Plot the data
            fig.add_trace(go.Scatter(name=self.simulator_agent_id, x=df_sim.index / TIME_CONVERSION[self.timescale], y=df_sim, mode="lines", line=self.LINE_PROPERTIES[self.simulator_agent_id]))
            fig.add_trace(go.Scatter(name=self.neg_flex_agent_id, x=df_neg.index / TIME_CONVERSION[self.timescale], y=df_neg, mode="lines", line=self.LINE_PROPERTIES[self.neg_flex_agent_id] | {"dash": "dash"}))
            fig.add_trace(go.Scatter(name=self.pos_flex_agent_id, x=df_pos.index / TIME_CONVERSION[self.timescale], y=df_pos, mode="lines", line=self.LINE_PROPERTIES[self.pos_flex_agent_id] | {"dash": "dash"}))
            fig.add_trace(go.Scatter(name=self.baseline_agent_id, x=df_bas.index / TIME_CONVERSION[self.timescale], y=df_bas, mode="lines", line=self.LINE_PROPERTIES[self.baseline_agent_id] | {"dash": "dash"}))

            return fig

        def plot_flexprices(fig: go.Figure) -> go.Figure:
            df_flex_market_index = self.df_flex_market.index.droplevel("time") / TIME_CONVERSION[self.timescale]
            fig.add_trace(go.Scatter(name=self.label_positive, x=df_flex_market_index, y=self.df_flex_market["pos_price"], mode="lines+markers", line=self.LINE_PROPERTIES[self.pos_flex_agent_id]))
            fig.add_trace(go.Scatter(name=self.label_negative, x=df_flex_market_index, y=self.df_flex_market["neg_price"], mode="lines+markers", line=self.LINE_PROPERTIES[self.neg_flex_agent_id]))
            return fig

        def plot_energyflex(fig: go.Figure) -> go.Figure:
            df_ind = self.df_indicator.xs(0, level=1)
            fig.add_trace(go.Scatter(name=self.label_positive, x=df_ind.index / TIME_CONVERSION[self.timescale], y=df_ind[glbs.ENERGYFLEX_POS], mode="lines+markers", line=self.LINE_PROPERTIES[self.pos_flex_agent_id]))
            fig.add_trace(go.Scatter(name=self.label_negative, x=df_ind.index / TIME_CONVERSION[self.timescale], y=df_ind[glbs.ENERGYFLEX_NEG], mode="lines+markers", line=self.LINE_PROPERTIES[self.neg_flex_agent_id]))
            return fig

        def plot_mpc_iterations(fig: go.Figure) -> go.Figure:
            fig.add_trace(go.Scatter(name=self.baseline_agent_id, x=self.df_baseline_stats.index / TIME_CONVERSION[self.timescale], y=self.df_baseline_stats["iter_count"], mode="markers", line=self.LINE_PROPERTIES[self.baseline_agent_id]))
            fig.add_trace(go.Scatter(name=self.pos_flex_agent_id, x=self.df_pos_flex_stats.index / TIME_CONVERSION[self.timescale], y=self.df_pos_flex_stats["iter_count"], mode="markers", line=self.LINE_PROPERTIES[self.pos_flex_agent_id]))
            fig.add_trace(go.Scatter(name=self.neg_flex_agent_id, x=self.df_neg_flex_stats.index / TIME_CONVERSION[self.timescale], y=self.df_neg_flex_stats["iter_count"], mode="markers", line=self.LINE_PROPERTIES[self.neg_flex_agent_id]))
            return fig

        def create_plot_for_one_variable(variable: str, at_time_step: int, show_current_characteristic_times: bool, zoom_to_prediction_interval: bool = False) -> go.Figure:
            """ Create a plot for one variable

            Keyword arguments:
            variable -- The variable to plot
            time_step -- The time_step to show the mpc predictions and the characteristic times
            show_current_characteristic_times -- Whether to show the characteristic times
            """
            fig = go.Figure()

            mark_characteristic_times_of_accepted_offers(fig=fig)

            # Plot variable
            if variable == self.mpc_iterations:
                plot_mpc_iterations(fig=fig)
            elif variable == self.energyflex:
                plot_energyflex(fig=fig)
            elif variable == self.price:
                plot_flexprices(fig=fig)
            else:
                if show_current_characteristic_times:
                    mark_characteristic_times(fig=fig, at_time_step=at_time_step)
                plot_one_mpc_variable(fig=fig, variable=variable, time_step=at_time_step)

            # set layout
            if zoom_to_prediction_interval:
                xlim_left = at_time_step
                xlim_right = at_time_step + self.df_baseline.index[-1][-1]
            else:
                xlim_left = self.df_simulation.index[0]
                xlim_right = self.df_simulation.index[-1] + self.df_baseline.index[-1][-1]
            xlim_left /= TIME_CONVERSION[self.timescale]
            xlim_right /= TIME_CONVERSION[self.timescale]
            fig.update_layout(yaxis_title=variable, xaxis_title=f"Time in {self.timescale}", xaxis_range=[xlim_left, xlim_right])

            return fig

        def get_variables_for_plotting() -> list[str]:
            # Get the intersection of quantities from sim and mpcs
            # if using a fmu for the simulation, make sure to change the column names to the ones used in the mpc, e.g. with a mapping
            variables_sim = self.df_simulation.columns.to_list()
            variables_mpc = self.df_baseline["variable"].columns.to_list()
            variables_posflex = self.df_pos_flex["variable"].columns.to_list()
            variables_negflex = self.df_neg_flex["variable"].columns.to_list()
            variables = list(set(variables_sim) & set(variables_mpc) & set(variables_posflex) & set(variables_negflex))

            # Add custom variables
            variables.append(self.energyflex)
            variables.append(self.price)
            variables.append(self.mpc_iterations)

            return variables

        # Get variables for plotting
        variables_for_plotting = get_variables_for_plotting()

        # Create the app
        # Get the mpc index for slider
        mpc_index = self.df_baseline.index.get_level_values(0).unique() / TIME_CONVERSION[self.timescale]

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
                            html.H3(f"Time in {self.timescale}:", style={"display": "inline-block", "padding-right": "10px"}),
                            dcc.Input(id="input_time", type="number",
                                      min=mpc_index[0], max=mpc_index[-1], step=mpc_index[1] - mpc_index[0],
                                      value=mpc_index[0],
                                      style={"display": "inline-block"}),
                        ],
                    ),
                    dcc.Slider(id="slider_time",
                               min=mpc_index[0], max=mpc_index[-1], step=mpc_index[1] - mpc_index[0],
                               tooltip={"placement": "bottom", "always_visible": True},
                               value=mpc_index[0], updatemode="drag")
                ], style={
                    "width": "88%", "padding-left": "0%", "padding-right": "12%",
                    # make the options sticky to the top of the page
                    "position": "sticky", "top": "0", "overflow-y": "visible", "z-index": "100", "background-color": "white"
                }
            ),
            # Container for the graphs
            html.Div(id="graphs_container_variables", children=[]),
        ]

        # Callbacks for changing inputs
        @callback(
            Output(component_id="slider_time", component_property="value"),
            Input(component_id="input_time", component_property="value")
        )
        @callback(
            Output(component_id="input_time", component_property="value"),
            Input(component_id="slider_time", component_property="value")
        )
        def update_time(time):
            return time

        @callback(
            Output(component_id="graphs_container_variables", component_property="children"),
            Input(component_id="current_characteristic_times", component_property="value"),
            Input(component_id="zoom_to_prediction_interval", component_property="value"),
            Input(component_id="slider_time", component_property="value"),
        )
        def update_graph(current_characteristic_times: bool, zoom_to_prediction_interval: bool, time_step: int):
            """ Update all graphs based on the options and slider values
            """
            figs = []
            for variable in variables_for_plotting:
                fig = create_plot_for_one_variable(variable=variable, at_time_step=time_step, show_current_characteristic_times=current_characteristic_times, zoom_to_prediction_interval=zoom_to_prediction_interval)
                figs.append(dcc.Graph(id=f"graph_{variable}", figure=fig))
            return figs

        # Run the app
        port = get_port()
        webbrowser.open_new_tab(f"http://localhost:{port}")
        app.run(debug=False, port=port)
