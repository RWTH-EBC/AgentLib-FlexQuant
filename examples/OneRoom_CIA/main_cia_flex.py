import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from pathlib import Path
from agentlib.utils.multi_agent_system import LocalMASAgency
from agentlib_mpc.utils.plotting.mpc import plot_mpc
import numpy as np
import agentlib_mpc.utils.plotting.basic as mpcplot
from agentlib_mpc.utils.analysis import load_sim, load_mpc
from agentlib_mpc.utils.analysis import mpc_at_time_step
from agentlib_flexquant.generate_flex_agents import FlexAgentGenerator
import logging
import pandas as pd
from agentlib_flexquant.utils.interactive import Dashboard

# Set the log-level
logging.basicConfig(level=logging.WARN)
until = 12000

time_of_activation = 1500

ENV_CONFIG = {"rt": False, "factor": 0.01, "t_sample": 10}


def run_example(until=until):
    results = []
    mpc_config = "mpc_and_sim/simple_cia_mpc.json"
    sim_config = "mpc_and_sim/simple_cia_sim.json"
    predictor_config = "predictor/predictor_config.json"
    flex_config = "flex_configs/flexibility_agent_config.json"
    agent_configs = [sim_config, predictor_config]

    config_list = FlexAgentGenerator(
        flex_config=flex_config, mpc_agent_config=mpc_config
    ).generate_flex_agents()
    agent_configs.extend(config_list)

    mas = LocalMASAgency(
        agent_configs=agent_configs, env=ENV_CONFIG, variable_logging=False
    )

    mas.run(until=until)
    results = mas.get_results(cleanup=False)

    # disturbances
    fig, axs = mpcplot.make_fig(style=mpcplot.Style(use_tex=False), rows=1)
    ax1 = axs[0]
    # load
    ax1.set_ylabel("$dot{Q}_{Room}$ in W")
    results["SimAgent"]["room"]["load"].plot(ax=ax1)
    x_ticks = np.arange(0, 3600 * 6 + 1, 3600)
    x_tick_labels = [int(tick / 3600) for tick in x_ticks]
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(x_tick_labels)
    ax1.set_xlabel("Time in hours")
    for ax in axs:
        mpcplot.make_grid(ax)
        ax.set_xlim(0, 3600 * 6)

    # room temp
    fig, axs = mpcplot.make_fig(style=mpcplot.Style(use_tex=False), rows=1)
    ax1 = axs[0]
    # T out
    ax1.set_ylabel("$T_{room}$ in K")
    results["SimAgent"]["room"]["T_upper"].plot(ax=ax1, color="0.5")
    results["SimAgent"]["room"]["T_out"].plot(ax=ax1, color=mpcplot.EBCColors.dark_grey)
    mpc_at_time_step(
        data=results["myMPCAgent"]["Baseline"], time_step=time_of_activation, variable="T"
    ).plot(ax=ax1, label="base", linestyle="--", color=mpcplot.EBCColors.dark_grey)
    mpc_at_time_step(
        data=results["NegFlexMPC"]["NegFlexMPC"], time_step=time_of_activation, variable="T"
    ).plot(ax=ax1, label="neg", linestyle="--", color=mpcplot.EBCColors.red)
    mpc_at_time_step(
        data=results["PosFlexMPC"]["PosFlexMPC"], time_step=time_of_activation, variable="T"
    ).plot(ax=ax1, label="pos", linestyle="--", color=mpcplot.EBCColors.blue)

    ax1.legend()
    ax1.vlines(time_of_activation, ymin=0, ymax=500, colors="black")
    ax1.vlines(time_of_activation + 300, ymin=0, ymax=500, colors="black")
    ax1.vlines(time_of_activation + 600, ymin=0, ymax=500, colors="black")
    ax1.vlines(time_of_activation + 3000, ymin=0, ymax=500, colors="black")

    ax1.set_ylim(289, 299)
    x_ticks = np.arange(0, 3600 * 6 + 1, 3600)
    x_tick_labels = [int(tick / 3600) for tick in x_ticks]
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(x_tick_labels)
    ax1.set_xlabel("Time in hours")
    for ax in axs:
        mpcplot.make_grid(ax)
        ax.set_xlim(0, 3600 * 6)

    # predictions
    fig, axs = mpcplot.make_fig(style=mpcplot.Style(use_tex=False), rows=1)
    ax1 = axs[0]
    # P_el
    ax1.set_ylabel("$P_{el}$ in W")
    results["SimAgent"]["room"]["P_el"].plot(ax=ax1, color=mpcplot.EBCColors.dark_grey)
    mpc_at_time_step(
        data=results["NegFlexMPC"]["NegFlexMPC"], time_step=time_of_activation, variable="P_el"
    ).ffill().plot(
        ax=ax1,
        drawstyle="steps-post",
        label="neg",
        linestyle="--",
        color=mpcplot.EBCColors.red,
    )
    mpc_at_time_step(
        data=results["PosFlexMPC"]["PosFlexMPC"], time_step=time_of_activation, variable="P_el"
    ).ffill().plot(
        ax=ax1,
        drawstyle="steps-post",
        label="pos",
        linestyle="--",
        color=mpcplot.EBCColors.blue,
    )
    mpc_at_time_step(
        data=results["myMPCAgent"]["Baseline"], time_step=time_of_activation, variable="P_el"
    ).ffill().plot(
        ax=ax1,
        drawstyle="steps-post",
        label="base",
        linestyle="--",
        color=mpcplot.EBCColors.dark_grey,
    )
    ax1.legend()
    ax1.vlines(time_of_activation, ymin=0, ymax=500, colors="black")
    ax1.vlines(time_of_activation + 300, ymin=0, ymax=500, colors="black")
    ax1.vlines(time_of_activation + 600, ymin=0, ymax=500, colors="black")
    ax1.vlines(time_of_activation + 3000, ymin=0, ymax=500, colors="black")

    # flexibility
    # get only the first prediction time of each time step
    ind_res = results["FlexibilityIndicator"]["FlexibilityIndicator"]
    energy_flex_neg = ind_res.xs("negative_energy_flex", axis=1).droplevel(1).dropna()
    energy_flex_pos = ind_res.xs("positive_energy_flex", axis=1).droplevel(1).dropna()
    fig, axs = mpcplot.make_fig(style=mpcplot.Style(use_tex=False), rows=1)
    ax1 = axs[0]
    ax1.set_ylabel("$epsilon$ in kWh")
    energy_flex_neg.plot(ax=ax1, label="neg", color=mpcplot.EBCColors.red)
    energy_flex_pos.plot(ax=ax1, label="pos", color=mpcplot.EBCColors.blue)
    ax1.yaxis.set_major_formatter(FormatStrFormatter("%.4f"))

    ax1.legend()

    x_ticks = np.arange(0, 3600 * 6 + 1, 3600)
    x_tick_labels = [int(tick / 3600) for tick in x_ticks]
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(x_tick_labels)
    ax1.set_xlabel("Time in hours")
    for ax in axs:
        mpcplot.make_grid(ax)
        ax.set_xlim(0, 3600 * 6)

    # plt.show()

    Dashboard(
        flex_config="flex_configs/flexibility_agent_config.json",
        simulator_agent_config="mpc_and_sim/simple_cia_sim.json",
        results=results
    ).show()

if __name__ == "__main__":
    run_example(until)
