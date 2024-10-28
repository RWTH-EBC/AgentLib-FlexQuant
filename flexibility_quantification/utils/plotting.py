from pathlib import Path
from typing import Union
import socket
import webbrowser

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from plotly import graph_objects as go
from plotly import express as px
from dash import Dash, html, dash_table, dcc, callback, Output, Input, Patch

from flexibility_quantification.utils.data_handling import load_results, res_type, convert_timescale, conv_factor
import agentlib_mpc.utils.plotting.basic as mpcplot
from agentlib_mpc.utils.analysis import mpc_at_time_step


def add_phase_marker(fig: go.Figure, results: res_type, unit: str = "h") -> go.Figure:
    df_indicator = results["FlexibilityIndicator"]["FlexibilityIndicator"]
    df_accepted = results["FlexibilityMarket"]["FlexibilityMarket"]["status"].str.contains(pat="OfferStatus.accepted")
    for i in df_accepted.index.to_list():
        if df_accepted[i]:
            offer_time = i[0]
            rel_market_time = df_indicator.loc[(i[0], 0), "market_time"] / conv_factor[unit]
            rel_prep_time = df_indicator.loc[(i[0], 0), "prep_time"] / conv_factor[unit]
            flex_event_duration = df_indicator.loc[(i[0], 0), "flex_event_duration"] / conv_factor[unit]

            fig.add_vline(x=offer_time, line=dict(color="grey"))
            fig.add_vline(x=offer_time + rel_market_time, line=dict(color="grey"))
            fig.add_vline(x=offer_time + rel_market_time + rel_prep_time, line=dict(color="grey"))
            fig.add_vline(x=offer_time + rel_market_time + rel_prep_time + flex_event_duration, line=dict(color="grey"))

    return fig


def plot_option(results: res_type, option: str, unit: str, timestamp: int = None, show_bounds: bool = True, show_phases: bool = True) -> list[dcc.Graph]:
    fig = go.Figure()
    if show_phases:
        add_phase_marker(fig=fig, results=results)
    if show_bounds and option in ["T", "T_out"]:
        df_lb = results["SimAgent"]["room"]["T_lower"]
        fig.add_trace(go.Scatter(x=df_lb.index, y=df_lb, mode="lines", name="T_lower", line=dict(color="grey")))
        df_ub = results["SimAgent"]["room"]["T_upper"]
        fig.add_trace(go.Scatter(x=df_ub.index, y=df_ub, mode="lines", name="T_upper", line=dict(color="grey")))

    try:
        df_sim = results["SimAgent"]["room"][option]
        fig.add_trace(go.Scatter(x=df_sim.index, y=df_sim, mode="lines", name="Sim", line=dict(color="black")))
    except KeyError:
        pass

    try:
        df_neg = mpc_at_time_step(data=results["NegFlexMPC"]["NegFlexMPC"], time_step=timestamp, variable=option).ffill()
        fig.add_trace(go.Scatter(x=df_neg.index, y=df_neg, mode="lines", name="NegFlexMPC", line=dict(color="red", dash="dash")))
    except KeyError:
        pass

    try:
        df_pos = mpc_at_time_step(data=results["PosFlexMPC"]["PosFlexMPC"], time_step=timestamp, variable=option).ffill()
        fig.add_trace(go.Scatter(x=df_pos.index, y=df_pos, mode="lines", name="PosFlexMPC", line=dict(color="green", dash="dash")))
    except KeyError:
        pass

    try:
        df_mpc = mpc_at_time_step(data=results["FlexModel"]["Baseline"], time_step=timestamp, variable=option).ffill()
        fig.add_trace(go.Scatter(x=df_mpc.index, y=df_mpc, mode="lines", name="Baseline", line=dict(color="blue", dash="dash")))
    except KeyError:
        pass

    try:
        df_ind = results["FlexibilityIndicator"]["FlexibilityIndicator"].xs(0, level=1)
        if option == "energyflex":
            fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind["energyflex_pos"], mode="lines", name="positive"))
            fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind["energyflex_neg"], mode="lines", name="negative"))
        else:
            fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind[option], mode="lines", name="FlexibilityIndicator", line=dict(color="black")))
    except KeyError:
        pass

    if option == "price":
        df_market = results["FlexibilityMarket"]["FlexibilityMarket"].xs(0.5, level=1)
        fig.add_trace(go.Scatter(x=df_market.index, y=-df_market["pos_price"], mode="lines+markers", name="positive"))
        fig.add_trace(go.Scatter(x=df_market.index, y=df_market["neg_price"], mode="lines+markers", name="negative"))

    fig.update_layout(title=option, xaxis_title=f"Time in {unit}", yaxis_title=f"{option} in Unit", xaxis_range=[results["SimAgent"]["room"].index[0], results["SimAgent"]["room"].index[-1]])     # title="Electric power"

    return fig


def get_options(results: res_type):
    options_sim = results["SimAgent"]["room"].columns.to_list()
    options_mpc = results["FlexModel"]["Baseline"]["variable"].columns.to_list()
    options_posflex = results["PosFlexMPC"]["PosFlexMPC"]["variable"].columns.to_list()
    options_negflex = results["NegFlexMPC"]["NegFlexMPC"]["variable"].columns.to_list()
    options = list(set(options_sim) & set(options_mpc) & set(options_posflex) & set(options_negflex))

    options.append("energyflex")
    options.append("price")

    return options


def get_port():
    port = 8050
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            is_free = s.connect_ex(("localhost", port)) != 0
        if is_free:
            return port
        else:
            port += 1


def interactive(results: res_type, unit: str = "h"):
    results = convert_timescale(results=results, unit=unit)

    app = Dash(__name__, title="Results")

    mpc_index = results["FlexModel"]["Baseline"].index.get_level_values(0).unique()

    app.layout = [
        dcc.Slider(id="slider_time", min=mpc_index[0], max=mpc_index[-1], step=mpc_index[1]-mpc_index[0], value=mpc_index[0], updatemode="drag"),
        dcc.Checklist(id="options", options=get_options(results=results), value=["T_out", "P_el"], inline=True),
        html.Div(id="graphs_container_div", children=[]),
    ]

    @callback(
        Output(component_id="graphs_container_div", component_property="children"),
        Input(component_id="slider_time", component_property="value"),
        Input(component_id="options", component_property="value")
    )
    def update_graph(timestamp: int, options: list[str]):
        figs = []
        for option in options:
            fig = plot_option(results=results, option=option, timestamp=timestamp, show_bounds=True, show_phases=True, unit=unit)
            figs.append(dcc.Graph(id=f"graph_{option}", figure=fig))
        return figs

    port = get_port()
    webbrowser.open_new_tab(f"http://localhost:{port}")
    app.run(debug=False, port=port)


def plot_results(results: Union[str, Path, res_type]):
    if isinstance(results, (str, Path)):
        results = load_results(res_path=results)
    elif not isinstance(results, dict):
        raise ValueError("Results must be a path or a dictionary")

    interactive(results=results, unit="h")
