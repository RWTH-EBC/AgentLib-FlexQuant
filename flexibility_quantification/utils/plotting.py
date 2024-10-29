from pathlib import Path
from typing import Union
import socket
import webbrowser

import pandas as pd

from plotly import graph_objects as go
from dash import Dash, html, dcc, callback, Output, Input

from flexibility_quantification.utils.data_handling import load_results, res_type, convert_timescale_index, conv_factor
from agentlib_mpc.utils.analysis import mpc_at_time_step


def add_phase_marker(fig: go.Figure, results: res_type, timestamp: int, time_unit: str = "h") -> go.Figure:
    df_indicator = results["FlexibilityIndicator"]["FlexibilityIndicator"].copy()
    try:
        offer_time = timestamp
        rel_market_time = df_indicator.loc[(timestamp, 0), "market_time"] / conv_factor[time_unit]
        rel_prep_time = df_indicator.loc[(timestamp, 0), "prep_time"] / conv_factor[time_unit]
        flex_event_duration = df_indicator.loc[(timestamp, 0), "flex_event_duration"] / conv_factor[time_unit]

        fig.add_vline(x=offer_time, line=dict(color="grey"))
        fig.add_vline(x=offer_time + rel_prep_time, line=dict(color="grey"))
        fig.add_vline(x=offer_time + rel_prep_time + rel_market_time, line=dict(color="grey"))
        fig.add_vline(x=offer_time + rel_prep_time + rel_market_time + flex_event_duration, line=dict(color="grey"))
    except KeyError:
        pass    # no phase data available, e.g. if offer accepted
    return fig


def add_phase_marker_accepted(fig: go.Figure, results: res_type, time_unit: str = "h") -> go.Figure:
    df_accepted = results["FlexibilityMarket"]["FlexibilityMarket"]["status"].copy().str.contains(pat="OfferStatus.accepted")
    for i in df_accepted.index.to_list():
        if df_accepted[i]:
            fig = add_phase_marker(fig=fig, results=results, timestamp=i[0], time_unit=time_unit)
    return fig


def plot_mpcs(results: res_type, fig: go.Figure, variable: str, timestamp: int, show_bounds: bool) -> go.Figure:
    if show_bounds:
        if variable in ["T", "T_out"]:
            df_lb = results["FlexModel"]["Baseline"][("parameter", "T_lower")].copy().xs(0, level=1)
            fig.add_trace(go.Scatter(name="T_lower", x=df_lb.index, y=df_lb, mode="lines", line=dict(color="grey")))
            df_ub = results["FlexModel"]["Baseline"][("parameter", "T_upper")].copy().xs(0, level=1)
            fig.add_trace(go.Scatter(name="T_upper", x=df_ub.index, y=df_ub, mode="lines", line=dict(color="grey")))
        else:
            df_lb = results["FlexModel"]["Baseline"][("lower", variable)].copy().xs(0, level=1)
            df_ub = results["FlexModel"]["Baseline"][("upper", variable)].copy().xs(0, level=1)
            if df_lb.notna().all() and df_ub.notna().all():
                fig.add_trace(go.Scatter(name="T_lower", x=df_lb.index, y=df_lb, mode="lines", line=dict(color="grey")))
                fig.add_trace(go.Scatter(name="T_upper", x=df_ub.index, y=df_ub, mode="lines", line=dict(color="grey")))

    df_sim = results["SimAgent"]["room"][variable].copy()
    df_neg = mpc_at_time_step(data=results["NegFlexMPC"]["NegFlexMPC"], time_step=timestamp, variable=variable).dropna()
    df_pos = mpc_at_time_step(data=results["PosFlexMPC"]["PosFlexMPC"], time_step=timestamp, variable=variable).dropna()
    df_bas = mpc_at_time_step(data=results["FlexModel"]["Baseline"], time_step=timestamp, variable=variable).dropna()

    fig.add_trace(go.Scatter(name="Simulation", x=df_sim.index, y=df_sim, mode="lines", line=dict(color="black")))
    fig.add_trace(go.Scatter(name="NegFlexMPC", x=df_neg.index, y=df_neg, mode="lines", line=dict(color="red", dash="dash")))
    fig.add_trace(go.Scatter(name="PosFlexMPC", x=df_pos.index, y=df_pos, mode="lines", line=dict(color="green", dash="dash")))
    fig.add_trace(go.Scatter(name="Baseline", x=df_bas.index, y=df_bas, mode="lines", line=dict(color="blue", dash="dash")))

    return fig


def plot_price(results: res_type, fig: go.Figure) -> go.Figure:
    df_market = results["FlexibilityMarket"]["FlexibilityMarket"].copy()
    df_market.index = df_market.index.droplevel("time")
    fig.add_trace(go.Scatter(name="positive", x=df_market.index, y=df_market["pos_price"], mode="lines+markers", line=dict(color="green")))
    fig.add_trace(go.Scatter(name="negative", x=df_market.index, y=df_market["neg_price"], mode="lines+markers", line=dict(color="red")))
    return fig


def plot_energyflex(results: res_type, fig: go.Figure) -> go.Figure:
    df_ind = results["FlexibilityIndicator"]["FlexibilityIndicator"].copy().xs(0, level=1)
    fig.add_trace(go.Scatter(name="Energyflex: positive", x=df_ind.index, y=df_ind["energyflex_pos"], mode="lines+markers", line=dict(color="green")))
    fig.add_trace(go.Scatter(name="Energyflex: negative", x=df_ind.index, y=df_ind["energyflex_neg"], mode="lines+markers", line=dict(color="red")))
    return fig


def plot_one(results: res_type, variable: str, time_unit: str, timestamp: int = None, show_bounds: bool = True, show_phases: bool = False, show_phases_accepted: bool = True) -> go.Figure:
    fig = go.Figure()
    if variable == "energyflex":
        fig = plot_energyflex(results=results, fig=fig)
    elif variable == "price":
        fig = plot_price(results=results, fig=fig)
    else:
        if show_phases:
            add_phase_marker(fig=fig, results=results, timestamp=timestamp, time_unit=time_unit)
        if show_phases_accepted:
            add_phase_marker_accepted(fig=fig, results=results, time_unit=time_unit)
        fig = plot_mpcs(results=results, fig=fig, variable=variable, timestamp=timestamp, show_bounds=show_bounds)

    fig.update_layout(title=variable, yaxis_title=variable, xaxis_title=f"Time in {time_unit}", xaxis_range=[results["SimAgent"]["room"].index[0], results["SimAgent"]["room"].index[-1]])

    return fig


def get_variable_options(results: res_type):
    # get intersection of quantities from sim and mpcs
    options_sim = results["SimAgent"]["room"].columns.to_list()
    options_mpc = results["FlexModel"]["Baseline"]["variable"].columns.to_list()
    options_posflex = results["PosFlexMPC"]["PosFlexMPC"]["variable"].columns.to_list()
    options_negflex = results["NegFlexMPC"]["NegFlexMPC"]["variable"].columns.to_list()
    options = list(set(options_sim) & set(options_mpc) & set(options_posflex) & set(options_negflex))

    # add custom options
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


def interactive(results: res_type, time_unit: str = "h"):
    """
    Interactive dashboard to plot the results
    """    
    results = convert_timescale_index(results=results, time_unit=time_unit)
    mpc_index = results["FlexModel"]["Baseline"].index.get_level_values(0).unique()
    
    # Create the app
    app = Dash(__name__, title="Results")
    app.layout = [
        html.H1("Results"),
        html.H2("Options"),
        dcc.Checklist(id="bounds", options=[{"label": "Show bounds", "value": True}], value=[True]),
        dcc.Checklist(id="phases_accepted", options=[{"label": "Show phases (accepted)", "value": True}], value=[True]),
        dcc.Checklist(id="phases_current", options=[{"label": "Show phases (current)", "value": False}], value=[False]),
        dcc.Checklist(id="variable_options", options=get_variable_options(results=results), value=[], inline=True),
        html.H2("Time"),
        html.Div(
            dcc.Slider(id="slider_time", min=mpc_index[0], max=mpc_index[-1], step=mpc_index[1]-mpc_index[0], value=mpc_index[0], updatemode="drag"),
            style={"width": "90%"}
        ),
        html.H2("Graphs"),
        html.Div(id="graphs_container_div", children=[]),
    ]

    # Callbacks
    @callback(
        Output(component_id="graphs_container_div", component_property="children"),
        Input(component_id="bounds", component_property="value"),
        Input(component_id="phases_accepted", component_property="value"),
        Input(component_id="phases_current", component_property="value"),
        Input(component_id="slider_time", component_property="value"),
        Input(component_id="variable_options", component_property="value")
    )
    def update_graph(show_bounds: bool, phases_accepted: bool, phases_current: bool, timestamp: int, variable_options: list[str]):
        figs = []
        for variable in variable_options:
            fig = plot_one(results=results, variable=variable, timestamp=timestamp, show_bounds=show_bounds, show_phases=phases_current, show_phases_accepted=phases_accepted, time_unit=time_unit)
            figs.append(dcc.Graph(id=f"graph_{variable}", figure=fig))
        return figs

    # Run the app
    port = get_port()
    webbrowser.open_new_tab(f"http://localhost:{port}")
    app.run(debug=False, port=port)


def plot_results(results: Union[str, Path, res_type]):
    if isinstance(results, (str, Path)):
        results = load_results(res_path=results)
    elif not isinstance(results, dict):
        raise ValueError("Results must be a path or a dictionary")

    interactive(results=results, time_unit="h")
