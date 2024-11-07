from pathlib import Path
from typing import Union
import webbrowser

import pandas as pd

from plotly import graph_objects as go
from dash import Dash, html, dcc, callback, Output, Input

from agentlib.core.agent import AgentConfig

from flexibility_quantification.utils.data_loading import load_agent_configs_and_results, convert_timescale_index, STATS_KEY
from agentlib_mpc.utils import TimeConversionTypes, TIME_CONVERSION
from agentlib_mpc.utils.plotting.interactive import solver_return, obj_plot
from flexibility_quantification.utils.config_management import (
    SIMULATOR_AGENT_KEY,
    BASELINE_AGENT_KEY,
    POS_FLEX_AGENT_KEY,
    NEG_FLEX_AGENT_KEY,
    INDICATOR_AGENT_KEY,
    FLEX_MARKET_AGENT_KEY,
)
import flexibility_quantification.data_structures.globals as glbs

from agentlib_mpc.utils.analysis import mpc_at_time_step
from agentlib_mpc.utils.plotting.interactive import get_port


def show_flex_dashboard(results: dict[str, dict[str, pd.DataFrame]], agent_configs: dict[str, AgentConfig], time_unit: TimeConversionTypes = "seconds"):
    """
    Interactive dashboard to plot the flexibility results
    Primarily for debugging

    Keyword arguments:
    results -- The results as a dictionary of the dataframes
    agent_configs -- The agent configurations as a dictionary with the agent key and the agent config as value
    time_unit -- The unit to scale the time to
    """
    # convert results to the desired time unit
    results = convert_timescale_index(results=results, time_unit=time_unit)

    # get agent and module ids
    simulator_agent_id = agent_configs[SIMULATOR_AGENT_KEY].id
    simulator_module_id = agent_configs[SIMULATOR_AGENT_KEY].modules[1]["module_id"]
    baseline_agent_id = agent_configs[BASELINE_AGENT_KEY].id
    baseline_module_id = agent_configs[BASELINE_AGENT_KEY].modules[1]["module_id"]
    pos_flex_agent_id = agent_configs[POS_FLEX_AGENT_KEY].id
    pos_flex_module_id = agent_configs[POS_FLEX_AGENT_KEY].modules[1]["module_id"]
    neg_flex_agent_id = agent_configs[NEG_FLEX_AGENT_KEY].id
    neg_flex_module_id = agent_configs[NEG_FLEX_AGENT_KEY].modules[1]["module_id"]
    indicator_agent_id = agent_configs[INDICATOR_AGENT_KEY].id
    indicator_module_id = agent_configs[INDICATOR_AGENT_KEY].modules[1]["module_id"]
    flex_market_agent_id = agent_configs[FLEX_MARKET_AGENT_KEY].id
    flex_market_module_id = agent_configs[FLEX_MARKET_AGENT_KEY].modules[1]["module_id"]

    # get dataframes
    df_simulation = results[simulator_agent_id][simulator_module_id]
    df_baseline = results[baseline_agent_id][baseline_module_id]
    df_baseline_stats = results[baseline_agent_id][STATS_KEY]
    df_pos_flex = results[pos_flex_agent_id][pos_flex_module_id]
    df_pos_flex_stats = results[pos_flex_agent_id][STATS_KEY]
    df_neg_flex = results[neg_flex_agent_id][neg_flex_module_id]
    df_neg_flex_stats = results[neg_flex_agent_id][STATS_KEY]
    df_indicator = results[indicator_agent_id][indicator_module_id]
    df_flex_market = results[flex_market_agent_id][flex_market_module_id]
    
    # define constants
    ENERGYFLEX = "energyflex"
    PRICE = "price"
    MPC_ITERATIONS = "iterations"
    
    # define line properties
    LINE_PROPERTIES = {
        simulator_agent_id: {
            "color": "black",
        },
        baseline_agent_id: {
            "color": "black",
        },
        neg_flex_agent_id: {
            "color": "red",
        },
        pos_flex_agent_id: {
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

    # define functions for plotting, they access the variables from the outer scope (show_flex_dashboard)
    def mark_characteristic_times(fig: go.Figure, timestamp: int, line_prop: dict = None) -> go.Figure:
        """
        Add markers of the characteristic times to the plot for a timestamp

        Keyword arguments:
        fig -- The figure to plot the results into
        timestamp -- When to show the markers
        line_prop -- The graphic properties of the lines as in plotly
        """
        if line_prop is None:
            line_prop = LINE_PROPERTIES["characteristic_times_current"]
        try:
            offer_time = timestamp
            rel_market_time = df_indicator.loc[(timestamp, 0), glbs.MARKET_TIME] / TIME_CONVERSION[time_unit]
            rel_prep_time = df_indicator.loc[(timestamp, 0), glbs.PREP_TIME] / TIME_CONVERSION[time_unit]
            flex_event_duration = df_indicator.loc[(timestamp, 0), glbs.FLEX_EVENT_DURATION] / TIME_CONVERSION[time_unit]

            fig.add_vline(x=offer_time, line=line_prop)
            fig.add_vline(x=offer_time + rel_prep_time, line=line_prop)
            fig.add_vline(x=offer_time + rel_prep_time + rel_market_time, line=line_prop)
            fig.add_vline(x=offer_time + rel_prep_time + rel_market_time + flex_event_duration, line=line_prop)
        except KeyError:
            pass  # no data of characteristic times available, e.g. if offer accepted
        return fig

    def mark_characteristic_times_of_accepted_offers(fig: go.Figure) -> go.Figure:
        """ Add markers of the characteristic times for accepted offers to the plot
        """
        df_accepted_offers = df_flex_market["status"].str.contains(pat="OfferStatus.accepted")
        for i in df_accepted_offers.index.to_list():
            if df_accepted_offers[i]:
                fig = mark_characteristic_times(fig=fig, timestamp=i[0], line_prop=LINE_PROPERTIES["characteristic_times_accepted"])
        return fig

    def plot_one_mpc_variable(fig: go.Figure, variable: str, timestamp: int) -> go.Figure:
        """ Create a plot for the mpc variable

        Keyword arguments:
        fig -- The figure to plot the results into
        variable -- The variable to plot
        timestamp -- The timestamp from when the mpc predictions should be shown
        """
        if variable in ["T", "T_out"]:
            df_lb = df_baseline[("parameter", "T_lower")].xs(0, level=1)
            fig.add_trace(go.Scatter(name="T_lower", x=df_lb.index, y=df_lb, mode="lines", line=LINE_PROPERTIES["bounds"]))
            df_ub = df_baseline[("parameter", "T_upper")].xs(0, level=1)
            fig.add_trace(go.Scatter(name="T_upper", x=df_ub.index, y=df_ub, mode="lines", line=LINE_PROPERTIES["bounds"]))
        else:
            df_lb = df_baseline[("lower", variable)].xs(0, level=1)
            df_ub = df_baseline[("upper", variable)].xs(0, level=1)
            if df_lb.notna().all() and df_ub.notna().all():
                fig.add_trace(go.Scatter(name="T_lower", x=df_lb.index, y=df_lb, mode="lines", line=LINE_PROPERTIES["bounds"]))
                fig.add_trace(go.Scatter(name="T_upper", x=df_ub.index, y=df_ub, mode="lines", line=LINE_PROPERTIES["bounds"]))

        df_sim = df_simulation[variable]
        df_neg = mpc_at_time_step(data=df_neg_flex, time_step=timestamp, variable=variable).dropna()
        df_pos = mpc_at_time_step(data=df_pos_flex, time_step=timestamp, variable=variable).dropna()
        df_bas = mpc_at_time_step(data=df_baseline, time_step=timestamp, variable=variable).dropna()

        fig.add_trace(go.Scatter(name=simulator_agent_id, x=df_sim.index, y=df_sim, mode="lines", line=LINE_PROPERTIES[simulator_agent_id]))
        fig.add_trace(go.Scatter(name=neg_flex_agent_id, x=df_neg.index, y=df_neg, mode="lines", line=LINE_PROPERTIES[neg_flex_agent_id] | {"dash": "dash"}))
        fig.add_trace(go.Scatter(name=pos_flex_agent_id, x=df_pos.index, y=df_pos, mode="lines", line=LINE_PROPERTIES[pos_flex_agent_id] | {"dash": "dash"}))
        fig.add_trace(go.Scatter(name=baseline_agent_id, x=df_bas.index, y=df_bas, mode="lines", line=LINE_PROPERTIES[baseline_agent_id] | {"dash": "dash"}))

        return fig

    def plot_flexprices(fig: go.Figure) -> go.Figure:
        df_flex_market_index = df_flex_market.index.droplevel("time")
        fig.add_trace(go.Scatter(name="positive", x=df_flex_market_index, y=df_flex_market["pos_price"], mode="lines+markers", line=LINE_PROPERTIES[pos_flex_agent_id]))
        fig.add_trace(go.Scatter(name="negative", x=df_flex_market_index, y=df_flex_market["neg_price"], mode="lines+markers", line=LINE_PROPERTIES[neg_flex_agent_id]))
        return fig

    def plot_energyflex(fig: go.Figure) -> go.Figure:
        df_ind = df_indicator.xs(0, level=1)
        fig.add_trace(go.Scatter(name="Energyflex: positive", x=df_ind.index, y=df_ind["energyflex_pos"], mode="lines+markers", line=LINE_PROPERTIES[pos_flex_agent_id]))
        fig.add_trace(go.Scatter(name="Energyflex: negative", x=df_ind.index, y=df_ind["energyflex_neg"], mode="lines+markers", line=LINE_PROPERTIES[neg_flex_agent_id]))
        return fig

    def plot_mpc_iterations(fig: go.Figure) -> go.Figure:
        fig.add_trace(go.Scatter(name=baseline_agent_id, x=df_baseline_stats.index, y=df_baseline_stats["iter_count"], mode="markers", line=LINE_PROPERTIES[baseline_agent_id]))
        fig.add_trace(go.Scatter(name=pos_flex_agent_id, x=df_pos_flex_stats.index, y=df_pos_flex_stats["iter_count"], mode="markers", line=LINE_PROPERTIES[pos_flex_agent_id]))
        fig.add_trace(go.Scatter(name=neg_flex_agent_id, x=df_neg_flex_stats.index, y=df_neg_flex_stats["iter_count"], mode="markers", line=LINE_PROPERTIES[neg_flex_agent_id]))
        return fig

    def create_plot_for_one_variable(variable: str, timestamp: int, show_characteristic_times: bool, complete_predction_horizon: bool = False) -> go.Figure:
        """ Create a plot for one variable

        Keyword arguments:
        variable -- The variable to plot
        timestamp -- The timestamp to show the mpc predictions and the characteristic times
        show_characteristic_times -- Whether to show the characteristic times
        """
        fig = go.Figure()

        mark_characteristic_times_of_accepted_offers(fig=fig)

        # plot variable
        if variable == MPC_ITERATIONS:
            plot_mpc_iterations(fig=fig)
        elif variable == ENERGYFLEX:
            plot_energyflex(fig=fig)
        elif variable == PRICE:
            plot_flexprices(fig=fig)
        else:
            if show_characteristic_times:
                mark_characteristic_times(fig=fig, timestamp=timestamp)
            plot_one_mpc_variable(fig=fig, variable=variable, timestamp=timestamp)

        # set layout
        x_left = df_simulation.index[0]
        x_right = df_simulation.index[-1]
        if complete_predction_horizon:
            x_right += df_baseline.index[-1][-1]
        fig.update_layout(yaxis_title=variable, xaxis_title=f"Time in {time_unit}", xaxis_range=[x_left, x_right])

        return fig

    def get_variables_for_plotting() -> list[str]:
        """ Get the possible variables to plot
        """
        # get intersection of quantities from sim and mpcs
        variables_sim = df_simulation.columns.to_list()
        variables_mpc = df_baseline["variable"].columns.to_list()
        variables_posflex = df_pos_flex["variable"].columns.to_list()
        variables_negflex = df_neg_flex["variable"].columns.to_list()
        variables = list(set(variables_sim) & set(variables_mpc) & set(variables_posflex) & set(variables_negflex))

        # add custom variables
        variables.append(ENERGYFLEX)
        variables.append(PRICE)
        variables.append(MPC_ITERATIONS)

        return variables

    # Create the app
    # Get mpc index for slider
    mpc_index = df_baseline.index.get_level_values(0).unique()

    app = Dash(__name__, title="Results")
    app.layout = [
        html.H1("Results"),
        html.Div(
            children=[
                html.H3("Options"),
                dcc.Checklist(id="characteristic_times_current",
                              options=[{"label": "Show characteristic times (current)", "value": True}],
                              value=[True]),
                dcc.Checklist(id="complete_predction_horizon",
                              options=[{"label": "Show complete prediction horizon", "value": True}],
                              value=[True]),
                html.H3("Time"),
                dcc.Slider(id="slider_time",
                           min=mpc_index[0], max=mpc_index[-1], step=mpc_index[1] - mpc_index[0],
                           value=mpc_index[0], updatemode="drag")
            ],
            style={
                "width": "88%", "padding-left": "0%", "padding-right": "12%",
                "position": "sticky", "top": "0", "overflow-y": "visible", "z-index": "100", "background-color": "white"
            }
        ),
        html.Div(id="graphs_container_variables", children=[]),
    ]

    # Callbacks
    @callback(
        Output(component_id="graphs_container_variables", component_property="children"),
        Input(component_id="characteristic_times_current", component_property="value"),
        Input(component_id="complete_predction_horizon", component_property="value"),
        Input(component_id="slider_time", component_property="value"),
    )
    def update_graph(characteristic_times: bool, complete_predction_horizon: bool, timestamp: int):
        figs = []
        for variable in get_variables_for_plotting():
            fig = create_plot_for_one_variable(variable=variable, timestamp=timestamp, show_characteristic_times=characteristic_times, complete_predction_horizon=complete_predction_horizon)
            figs.append(dcc.Graph(id=f"graph_{variable}", figure=fig))
        return figs

    # Run the app
    port = get_port()
    webbrowser.open_new_tab(f"http://localhost:{port}")
    app.run(debug=False, port=port)


def plot_results(agent_configs: Union[list[str], list[Path]], results: Union[str, Path, dict[str, pd.DataFrame]]):
    """Plot the results of the simulation, after loading necessary data

    Keyword arguments:
    agent_configs -- The paths to the agent configs used for the simulation
    results -- The results as a path to the results folder or the dictionary of the dataframes
    """
    agent_configs, results = load_agent_configs_and_results(agent_configs_paths=agent_configs, results=results)

    # space for further plotting scripts

    show_flex_dashboard(results=results, agent_configs=agent_configs, time_unit="hours")
