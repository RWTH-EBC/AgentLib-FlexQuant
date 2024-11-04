from pathlib import Path
from typing import Union
import socket
import webbrowser

import pandas as pd

from plotly import graph_objects as go
from dash import Dash, html, dcc, callback, Output, Input

from flexibility_quantification.utils.data_handling import load_results, RES_TYPE, convert_timescale_index, TIME_CONV_FACTOR, baselineID, posFlexID, negFlexID
import flexibility_quantification.data_structures.globals as glbs

from agentlib_mpc.utils.analysis import mpc_at_time_step
from agentlib_mpc.utils.plotting.interactive import get_port


ENERGYFLEX = "energyflex"
PRICE = "price"


def mark_characteristic_times(fig: go.Figure, results: RES_TYPE, timestamp: int, time_unit: str = "h", line_prop: dict = None) -> go.Figure:
    """
    Add markers of the characteristic times to the plot for a timestamp

    Keyword arguments:
    fig -- The figure to plot the results into
    results -- The results as a dictionary of the dataframes
    timestamp -- When to show the markers
    time_unit -- The time unit the index of the dataframes is in (options: "s", "min", "h", "d")
    line_prop -- The graphic properties of the lines as in plotly

    Returns:
    fig -- The figure with the added markers
    """
    if line_prop is None:
        line_prop = dict(color="grey", dash="dash")
    df_indicator = results["FlexibilityIndicator"]["FlexibilityIndicator"].copy()
    try:
        offer_time = timestamp
        rel_market_time = df_indicator.loc[(timestamp, 0), glbs.MARKET_TIME] / TIME_CONV_FACTOR[time_unit]
        rel_prep_time = df_indicator.loc[(timestamp, 0), glbs.PREP_TIME] / TIME_CONV_FACTOR[time_unit]
        flex_event_duration = df_indicator.loc[(timestamp, 0), glbs.FLEX_EVENT_DURATION] / TIME_CONV_FACTOR[time_unit]

        fig.add_vline(x=offer_time, line=line_prop)
        fig.add_vline(x=offer_time + rel_prep_time, line=line_prop)
        fig.add_vline(x=offer_time + rel_prep_time + rel_market_time, line=line_prop)
        fig.add_vline(x=offer_time + rel_prep_time + rel_market_time + flex_event_duration, line=line_prop)
    except KeyError:
        pass    # no data of characteristic times available, e.g. if offer accepted
    return fig


def mark_characteristic_times_of_accepted_offers(fig: go.Figure, results: RES_TYPE, time_unit: str = "h") -> go.Figure:
    """
    Add markers of the characteristic times for accepted offers to the plot

    Keyword arguments:
    fig -- The figure to plot the results into
    results -- The results as a dictionary of the dataframes
    time_unit -- The time unit the index of the dataframes is in (options: "s", "min", "h", "d")

    Returns:
    fig -- The figure with the added characteristic times
    """
    df_accepted = results["FlexibilityMarket"]["FlexibilityMarket"]["status"].copy().str.contains(pat="OfferStatus.accepted")
    for i in df_accepted.index.to_list():
        if df_accepted[i]:
            fig = mark_characteristic_times(fig=fig, results=results, timestamp=i[0], time_unit=time_unit, line_prop=dict(color="yellow"))
    return fig


def create_plot_for_mpc_variable(results: RES_TYPE, fig: go.Figure, variable: str, timestamp: int) -> go.Figure:
    """
    Create a plot for the mpc variable

    Keyword arguments:
    results -- The results as a dictionary of the dataframes
    fig -- The figure to plot the results into
    variable -- The variable to plot
    timestamp -- The timestamp to show the mpc predictions

    Returns:
    fig -- The figure with the added plot
    """
    if variable in ["T", "T_out"]:
        df_lb = results["FlexModel"][baselineID][("parameter", "T_lower")].copy().xs(0, level=1)
        fig.add_trace(go.Scatter(name="T_lower", x=df_lb.index, y=df_lb, mode="lines", line=dict(color="grey")))
        df_ub = results["FlexModel"][baselineID][("parameter", "T_upper")].copy().xs(0, level=1)
        fig.add_trace(go.Scatter(name="T_upper", x=df_ub.index, y=df_ub, mode="lines", line=dict(color="grey")))
    else:
        df_lb = results["FlexModel"][baselineID][("lower", variable)].copy().xs(0, level=1)
        df_ub = results["FlexModel"][baselineID][("upper", variable)].copy().xs(0, level=1)
        if df_lb.notna().all() and df_ub.notna().all():
            fig.add_trace(go.Scatter(name="T_lower", x=df_lb.index, y=df_lb, mode="lines", line=dict(color="grey")))
            fig.add_trace(go.Scatter(name="T_upper", x=df_ub.index, y=df_ub, mode="lines", line=dict(color="grey")))

    df_sim = results["SimAgent"]["room"][variable].copy()
    df_neg = mpc_at_time_step(data=results[negFlexID][negFlexID], time_step=timestamp, variable=variable).dropna()
    df_pos = mpc_at_time_step(data=results[posFlexID][posFlexID], time_step=timestamp, variable=variable).dropna()
    df_bas = mpc_at_time_step(data=results["FlexModel"][baselineID], time_step=timestamp, variable=variable).dropna()

    fig.add_trace(go.Scatter(name="Simulation", x=df_sim.index, y=df_sim, mode="lines", line=dict(color="black")))
    fig.add_trace(go.Scatter(name=negFlexID, x=df_neg.index, y=df_neg, mode="lines", line=dict(color="red", dash="dash")))
    fig.add_trace(go.Scatter(name=posFlexID, x=df_pos.index, y=df_pos, mode="lines", line=dict(color="blue", dash="dash")))
    fig.add_trace(go.Scatter(name=baselineID, x=df_bas.index, y=df_bas, mode="lines", line=dict(color="black", dash="dash")))

    return fig


def create_plot_for_price(results: RES_TYPE, fig: go.Figure) -> go.Figure:
    """
    Create a plot for the price results

    Keyword arguments:
    results -- The results as a dictionary of the dataframes
    fig -- The figure to plot the results into

    Returns:
    fig -- The figure with the added plot
    """

    df_market = results["FlexibilityMarket"]["FlexibilityMarket"].copy()
    df_market.index = df_market.index.droplevel("time")
    fig.add_trace(go.Scatter(name="positive", x=df_market.index, y=df_market["pos_price"], mode="lines+markers", line=dict(color="blue")))
    fig.add_trace(go.Scatter(name="negative", x=df_market.index, y=df_market["neg_price"], mode="lines+markers", line=dict(color="red")))
    return fig


def create_plot_for_energyflex(results: RES_TYPE, fig: go.Figure) -> go.Figure:
    """
    Create a plot for the energyflex results

    Keyword arguments:
    results -- The results as a dictionary of the dataframes
    fig -- The figure to plot the results into

    Returns:
    fig -- The figure with the added plot
    """
    df_ind = results["FlexibilityIndicator"]["FlexibilityIndicator"].copy().xs(0, level=1)
    fig.add_trace(go.Scatter(name="Energyflex: positive", x=df_ind.index, y=df_ind["energyflex_pos"], mode="lines+markers", line=dict(color="blue")))
    fig.add_trace(go.Scatter(name="Energyflex: negative", x=df_ind.index, y=df_ind["energyflex_neg"], mode="lines+markers", line=dict(color="red")))
    return fig


def create_plot_for_one_variable(results: RES_TYPE, variable: str, time_unit: str, timestamp: int, show_characteristic_times: bool) -> go.Figure:
    """
    Create a plot for one variable

    Keyword arguments:
    results -- The results as a dictionary of the dataframes
    variable -- The variable to plot
    time_unit -- The time unit to convert the index to (options: "s", "min", "h", "d"; assumption: index is in seconds)
    timestamp -- The timestamp to show the mpc predictions and the characteristic times
    show_characteristic_times -- Whether to show the characteristic times

    Returns:
    fig -- The figure with the added plot
    """
    fig = go.Figure()

    mark_characteristic_times_of_accepted_offers(fig=fig, results=results, time_unit=time_unit)

    if variable == ENERGYFLEX:
        fig = create_plot_for_energyflex(results=results, fig=fig)
    elif variable == PRICE:
        fig = create_plot_for_price(results=results, fig=fig)
    else:
        if show_characteristic_times:
            mark_characteristic_times(fig=fig, results=results, timestamp=timestamp, time_unit=time_unit)
        fig = create_plot_for_mpc_variable(results=results, fig=fig, variable=variable, timestamp=timestamp)

    x_left = results["SimAgent"]["room"].index[0]
    x_right = results["SimAgent"]["room"].index[-1] + results["FlexModel"][baselineID].index[-1][-1]
    fig.update_layout(yaxis_title=variable, xaxis_title=f"Time in {time_unit}", xaxis_range=[x_left, x_right])

    return fig


def get_variable_options(results: RES_TYPE) -> list[str]:
    """
    Get the possible variables to plot

    Keyword arguments:
    results -- The results as a dictionary of the dataframes

    Returns:
    options -- The possible variables to plot
    """
    # get intersection of quantities from sim and mpcs
    options_sim = results["SimAgent"]["room"].columns.to_list()
    options_mpc = results["FlexModel"][baselineID]["variable"].columns.to_list()
    options_posflex = results[posFlexID][posFlexID]["variable"].columns.to_list()
    options_negflex = results[negFlexID][negFlexID]["variable"].columns.to_list()
    options = list(set(options_sim) & set(options_mpc) & set(options_posflex) & set(options_negflex))

    # add custom options
    options.append(ENERGYFLEX)
    options.append(PRICE)

    return options


def show_flex_dashboard(results: RES_TYPE, time_unit: str = "h"):
    """
    Interactive dashboard to plot the flexibility results
    Primarily for debugging

    Keyword arguments:
    results -- The results as a dictionary of the dataframes
    time_unit -- The time unit to convert the index to (default "h"; options: "s", "min", "h", "d"; assumption: index is in seconds)
    """    
    results = convert_timescale_index(results=results, time_unit=time_unit)
    mpc_index = results["FlexModel"][baselineID].index.get_level_values(0).unique()
    
    # Create the app
    app = Dash(__name__, title="Results")
    app.layout = [
        html.H1("Results"),
        html.Div(
            children=[
                html.H3("Options"),
                dcc.Checklist(id="characteristic_times",
                              options=[{"label": "Show characteristic times (current)", "value": False}],
                              value=[False]),
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
        html.Div(id="graphs_container_div", children=[]),
    ]

    # Callbacks
    @callback(
        Output(component_id="graphs_container_div", component_property="children"),
        Input(component_id="characteristic_times", component_property="value"),
        Input(component_id="slider_time", component_property="value"),
    )
    def update_graph(characteristic_times: bool, timestamp: int):
        variable_options = get_variable_options(results=results)
        figs = []
        for variable in variable_options:
            fig = create_plot_for_one_variable(
                results=results, variable=variable,
                timestamp=timestamp, show_characteristic_times=characteristic_times, time_unit=time_unit
            )
            figs.append(dcc.Graph(id=f"graph_{variable}", figure=fig))
        return figs

    # Run the app
    port = get_port()
    webbrowser.open_new_tab(f"http://localhost:{port}")
    app.run(debug=False, port=port)


def plot_results(results: Union[str, Path, RES_TYPE]):
    """
    Plot the results of the simulation

    Keyword arguments:
    results -- The results as a path to the results folder or a dictionary of the dataframes
    """
    if isinstance(results, (str, Path)):
        results = load_results(res_path=results)
    elif not isinstance(results, dict):
        raise ValueError("Results must be a path or a dictionary")

    show_flex_dashboard(results=results, time_unit="h")
