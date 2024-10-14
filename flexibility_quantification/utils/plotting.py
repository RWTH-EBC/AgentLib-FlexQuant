from pathlib import Path
from typing import Union
import socket
import webbrowser

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from plotly import graph_objects as go
from plotly import express as px
from dash import Dash, html, dash_table, dcc, callback, Output, Input

from flexibility_quantification.utils.data_handling import load_results, res_type
import agentlib_mpc.utils.plotting.basic as mpcplot
from agentlib_mpc.utils.analysis import mpc_at_time_step


def format_axes(axs: tuple[plt.Axes, ...]):
    x_ticks = np.arange(0, 3600 * 6 + 1, 3600)
    x_tick_labels = [int(tick / 3600) for tick in x_ticks]
    axs[-1].set_xticks(x_ticks)
    axs[-1].set_xticklabels(x_tick_labels)
    axs[-1].set_xlabel("Time in hours")
    for ax in axs:
        mpcplot.make_grid(ax)
        ax.set_xlim(0, 3600 * 6)


def set_phase_marker(ax: plt.Axes, position: list[int] = None):   # TODO: position
    if position is None:
        position = [9000, 9900, 10800, 18000]
    elif len(position) != 4:
        raise ValueError("Position must have 4 values (begin offer, acceptance offer, start, end)")
    for pos in position:
        ax.vlines(pos, ymin=-1000, ymax=5000, colors="black")


def plot_disturbances(results: res_type):
    fig, axs = mpcplot.make_fig(style=mpcplot.Style(use_tex=False), rows=2)
    (ax1, ax2) = axs
    # load
    ax1.set_ylabel("$\dot{Q}_{Room}$ in W")
    results["SimAgent"]["room"]["load"].plot(ax=ax1)
    # T_in
    ax2.set_ylabel("$T_{in}$ in K")
    results["SimAgent"]["room"]["T_in"].plot(ax=ax2)

    format_axes(axs=axs)


def plot_room_temp(results: res_type):
    fig, axs = mpcplot.make_fig(style=mpcplot.Style(use_tex=False), rows=1)
    ax1 = axs[0]
    # T out
    ax1.set_ylabel("$T_{room}$ in K")
    results["SimAgent"]["room"]["T_upper"].plot(ax=ax1, color="0.5")
    results["SimAgent"]["room"]["T_lower"].plot(ax=ax1, color="0.5")
    results["SimAgent"]["room"]["T_out"].plot(ax=ax1, color=mpcplot.EBCColors.dark_grey)
    mpc_at_time_step(data=results["NegFlexMPC"]["NegFlexMPC"],
                     time_step=9000, variable="T").plot(ax=ax1, label="neg", linestyle="--",
                                                        color=mpcplot.EBCColors.red)
    mpc_at_time_step(data=results["PosFlexMPC"]["PosFlexMPC"],
                     time_step=9000, variable="T").plot(ax=ax1, label="pos", linestyle="--",
                                                        color=mpcplot.EBCColors.blue)
    mpc_at_time_step(data=results["FlexModel"]["Baseline"],
                     time_step=9900, variable="T").plot(ax=ax1, label="base", linestyle="--",
                                                        color=mpcplot.EBCColors.dark_grey)
    ax1.legend()
    set_phase_marker(ax=ax1)
    ax1.set_ylim(289, 299)
    format_axes(axs=axs)


def plot_predictions(results: res_type):
    fig, axs = mpcplot.make_fig(style=mpcplot.Style(use_tex=False), rows=2)
    (ax1, ax2) = axs
    # P_el
    ax1.set_ylabel("$P_{el}$ in kW")
    results["SimAgent"]["room"]["P_el"].plot(ax=ax1, color=mpcplot.EBCColors.dark_grey)
    mpc_at_time_step(data=results["NegFlexMPC"]["NegFlexMPC"],
                     time_step=9000, variable="P_el").ffill().plot(ax=ax1, drawstyle="steps-post",
                                                                   label="neg", linestyle="--",
                                                                   color=mpcplot.EBCColors.red)
    mpc_at_time_step(data=results["PosFlexMPC"]["PosFlexMPC"],
                     time_step=9000, variable="P_el").ffill().plot(ax=ax1, drawstyle="steps-post",
                                                                   label="pos", linestyle="--",
                                                                   color=mpcplot.EBCColors.blue)
    mpc_at_time_step(data=results["FlexModel"]["Baseline"],
                     time_step=9900, variable="P_el").ffill().plot(ax=ax1, drawstyle="steps-post",
                                                                   label="base", linestyle="--",
                                                                   color=mpcplot.EBCColors.dark_grey)
    ax1.legend()
    set_phase_marker(ax=ax1)
    ax1.set_ylim(-0.1, 1)
    # mdot
    ax2.set_ylabel("$\dot{m}$ in kg/s")
    results["SimAgent"]["room"]["mDot"].plot(ax=ax2, color=mpcplot.EBCColors.dark_grey)
    mpc_at_time_step(data=results["NegFlexMPC"]["NegFlexMPC"],
                     time_step=9000, variable="mDot").ffill().plot(ax=ax2, drawstyle="steps-post",
                                                                   label="neg", linestyle="--",
                                                                   color=mpcplot.EBCColors.red)
    mpc_at_time_step(data=results["PosFlexMPC"]["PosFlexMPC"],
                     time_step=9000, variable="mDot").ffill().plot(ax=ax2, drawstyle="steps-post",
                                                                   label="pos", linestyle="--",
                                                                   color=mpcplot.EBCColors.blue)
    mpc_at_time_step(data=results["FlexModel"]["Baseline"],
                     time_step=9900, variable="mDot").ffill().plot(ax=ax2, drawstyle="steps-post",
                                                                   label="base", linestyle="--",
                                                                   color=mpcplot.EBCColors.dark_grey)
    ax2.legend()
    set_phase_marker(ax=ax2)
    ax2.set_ylim(0, 0.06)

    format_axes(axs=axs)


def plot_flexibility(results: res_type):
    # get only the first prediction time of each time step
    ind_res = results["FlexibilityIndicator"]["FlexibilityIndicator"]
    energy_flex_neg = ind_res.xs("energyflex_neg", axis=1).droplevel(1).dropna()
    energy_flex_pos = ind_res.xs("energyflex_pos", axis=1).droplevel(1).dropna()
    fig, axs = mpcplot.make_fig(style=mpcplot.Style(use_tex=False), rows=1)
    ax1 = axs[0]
    ax1.set_ylabel("$\epsilon$ in kWh")
    energy_flex_neg.plot(ax=ax1, label="neg", color=mpcplot.EBCColors.red)
    energy_flex_pos.plot(ax=ax1, label="pos", color=mpcplot.EBCColors.blue)
    ax1.legend()
    format_axes(axs=axs)


def get_port():
    port = 8050
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            is_free = s.connect_ex(("localhost", port)) != 0
        if is_free:
            return port
        else:
            port += 1


def plot_temperature(results: res_type, timestamp: int = None) -> go.Figure:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=results["SimAgent"]["room"].index,
                                 y=results["SimAgent"]["room"]["T_out"],
                                 mode="lines",
                                 name="Sim"))
        fig.add_trace(go.Scatter(x=results["SimAgent"]["room"].index,
                                 y=results["SimAgent"]["room"]["T_lower"],
                                 mode="lines",
                                 name="T_lower"))
        fig.add_trace(go.Scatter(x=results["SimAgent"]["room"].index,
                                 y=results["SimAgent"]["room"]["T_upper"],
                                 mode="lines",
                                 name="T_upper"))

        if timestamp is not None:
            df_neg = mpc_at_time_step(data=results["NegFlexMPC"]["NegFlexMPC"], time_step=timestamp, variable="T")
            fig.add_trace(go.Scatter(x=df_neg.index,
                                     y=df_neg,
                                     mode="lines",
                                     name="NegFlexMPC"))

            df_pos = mpc_at_time_step(data=results["PosFlexMPC"]["PosFlexMPC"], time_step=timestamp, variable="T")
            fig.add_trace(go.Scatter(x=df_pos.index,
                                     y=df_pos,
                                     mode="lines",
                                     name="PosFlexMPC"))

            df_mpc = mpc_at_time_step(data=results["FlexModel"]["Baseline"], time_step=timestamp, variable="T")
            fig.add_trace(go.Scatter(x=df_mpc.index,
                                     y=df_mpc,
                                     mode="lines",
                                     name="Baseline"))

        return fig


def interactive(results: res_type):
    app = Dash(__name__, title="Results")

    mpc_index = results["NegFlexMPC"]["NegFlexMPC"].index.get_level_values(0).unique()

    # Define the layout of the webpage
    app.layout = [
            dcc.Slider(id="slider_time", min=mpc_index[0], max=mpc_index[-1], step=mpc_index[1]-mpc_index[0], value=mpc_index[0]),
            dcc.Graph(id="graph_temperature", figure=plot_temperature(results=results, timestamp=mpc_index[0])),
    ]

    @callback(
        Output(component_id="graph_temperature", component_property="figure"),
        Input(component_id="slider_time", component_property="value")
    )
    def update_graph_temperature(timestamp: int):
        return plot_temperature(results=results, timestamp=timestamp)

    port = get_port()
    webbrowser.open_new_tab(f"http://localhost:{port}")
    app.run(debug=False, port=port)


def plot_results(results: Union[str, Path, res_type]):
    if isinstance(results, (str, Path)):
        results = load_results(res_path=results)
    elif not isinstance(results, dict):
        raise ValueError("Results must be a path or a dictionary")

    #plot_disturbances(results=results)
    #plot_room_temp(results=results)
    #plot_predictions(results=results)
    #plot_flexibility(results=results)

    #plt.show()

    interactive(results=results)
