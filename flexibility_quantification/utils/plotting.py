from pathlib import Path
from typing import Union
from flexibility_quantification.utils.data_handling import load_results, res_type
import numpy as np
import matplotlib.pyplot as plt
import agentlib_mpc.utils.plotting.basic as mpcplot
from agentlib_mpc.utils.analysis import mpc_at_time_step


def plot_disturbances(results: res_type):
    fig, axs = mpcplot.make_fig(style=mpcplot.Style(use_tex=False), rows=2)
    (ax1, ax2) = axs
    # load
    ax1.set_ylabel("$\dot{Q}_{Room}$ in W")
    results["SimAgent"]["room"]["load"].plot(ax=ax1)
    # T_in
    ax2.set_ylabel("$T_{in}$ in K")
    results["SimAgent"]["room"]["T_in"].plot(ax=ax2)
    x_ticks = np.arange(0, 3600 * 6 + 1, 3600)
    x_tick_labels = [int(tick / 3600) for tick in x_ticks]
    ax2.set_xticks(x_ticks)
    ax2.set_xticklabels(x_tick_labels)
    ax2.set_xlabel("Time in hours")
    for ax in axs:
        mpcplot.make_grid(ax)
        ax.set_xlim(0, 3600 * 6)


def plot_room_temp(results: res_type):
    # room temp
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
    ax1.vlines(9000, ymin=0, ymax=500, colors="black")
    ax1.vlines(9900, ymin=0, ymax=500, colors="black")
    ax1.vlines(10800, ymin=0, ymax=500, colors="black")
    ax1.vlines(18000, ymin=0, ymax=500, colors="black")
    ax1.set_ylim(289, 299)
    x_ticks = np.arange(0, 3600 * 6 + 1, 3600)
    x_tick_labels = [int(tick / 3600) for tick in x_ticks]
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(x_tick_labels)
    ax1.set_xlabel("Time in hours")
    for ax in axs:
        mpcplot.make_grid(ax)
        ax.set_xlim(0, 3600 * 6)


def plot_predictions(results: res_type):
    # predictions
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
    ax1.vlines(9000, ymin=-1000, ymax=5000, colors="black")
    ax1.vlines(9900, ymin=-1000, ymax=5000, colors="black")
    ax1.vlines(10800, ymin=-1000, ymax=5000, colors="black")
    ax1.vlines(18000, ymin=-1000, ymax=5000, colors="black")
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
    ax2.vlines(9000, ymin=0, ymax=500, colors="black")
    ax2.vlines(9900, ymin=0, ymax=500, colors="black")
    ax2.vlines(10800, ymin=0, ymax=500, colors="black")
    ax2.vlines(18000, ymin=0, ymax=500, colors="black")
    ax2.set_ylim(0, 0.06)

    x_ticks = np.arange(0, 3600 * 6 + 1, 3600)
    x_tick_labels = [int(tick / 3600) for tick in x_ticks]
    ax2.set_xticks(x_ticks)
    ax2.set_xticklabels(x_tick_labels)
    ax2.set_xlabel("Time in hours")
    for ax in axs:
        mpcplot.make_grid(ax)
        ax.set_xlim(0, 3600 * 6)


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

    x_ticks = np.arange(0, 3600 * 6 + 1, 3600)
    x_tick_labels = [int(tick / 3600) for tick in x_ticks]
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(x_tick_labels)
    ax1.set_xlabel("Time in hours")
    for ax in axs:
        mpcplot.make_grid(ax)
        ax.set_xlim(0, 3600 * 6)


def plot_results(results: Union[str, Path, res_type]):
    if isinstance(results, (str, Path)):
        results = load_results(res_path=results)
    elif isinstance(results, dict):
        pass
    else:
        raise ValueError("Results must be a path or a dictionary")

    plot_disturbances(results=results)

    plot_room_temp(results=results)

    plot_predictions(results=results)

    plot_flexibility(results=results)

    plt.show()
