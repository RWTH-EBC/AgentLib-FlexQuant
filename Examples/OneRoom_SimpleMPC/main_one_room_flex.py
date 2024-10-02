

import matplotlib.pyplot as plt
from pathlib import Path
from agentlib.utils.multi_agent_system import LocalMASAgency
import numpy as np
import agentlib_mpc.utils.plotting.basic as mpcplot
from agentlib_mpc.utils.analysis import load_sim, load_mpc
from agentlib_mpc.utils.analysis import mpc_at_time_step
from flexibility_quantification.generate_flex_agents import FlexAgentGenerator
import logging
import pandas as pd

# Set the log-level
logging.basicConfig(level=logging.WARN)
until = 14400

ENV_CONFIG = {"rt": False, "factor": 0.01, "t_sample": 60}


def run_example(until=until):
    results = []
    mpc_config = "mpc_and_sim/simple_model.json"
    sim_config = "mpc_and_sim/simple_sim.json"
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

    # TODO: outsource plotting in different script

    # for external plot script
    ResultsT = dict[str, dict[str, pd.DataFrame]]
    res_path = "results"

    def load_results() -> ResultsT:
        results = {
            "Simulation": {"room": load_sim(Path(res_path, "sim_room.csv"))},
            "mpc": {
                "baseline": load_mpc(Path(res_path, "mpc_base.csv")),
                "pos": load_mpc(Path(res_path, "mpc_neg_flex.csv")),
                "neg": load_mpc(Path(res_path, "mpc_pos_flex.csv")),
            },
            # TODO: implement load functions
            # "indicator": {"admm_module": load_indicator(Path(res_path, "flexibility_indicator.csv"))},
            # "market": {"admm_module": load_market(Path(res_path, "flexibility_market.csv"))},
        }
        return results

    if results is None:
        results = load_results()

    # disturbances
    fig, axs = mpcplot.make_fig(style=mpcplot.Style(use_tex=False), rows=2)
    (ax1, ax2) = axs
    # load
    ax1.set_ylabel("$\dot{Q}_{Room}$ in W")
    results["SimAgent"]["room"]["load"].plot(ax=ax1)
    # T_in
    ax2.set_ylabel("$T_{in}$ in K")
    results["SimAgent"]["room"]["T_in"].plot(ax=ax2)
    x_ticks = np.arange(0, 3600 * 4 + 1, 3600)
    x_tick_labels = [int(tick / 3600) for tick in x_ticks]
    ax2.set_xticks(x_ticks)
    ax2.set_xticklabels(x_tick_labels)
    ax2.set_xlabel("Time in hours")
    for ax in axs:
        mpcplot.make_grid(ax)
        ax.set_xlim(0, 3600 * 4)

    # room temp
    fig, axs = mpcplot.make_fig(style=mpcplot.Style(use_tex=False), rows=1)
    ax1 = axs[0]
    # T out
    ax1.set_ylabel("$T_{room}$ in K")
    results["SimAgent"]["room"]["T_upper"].plot(ax=ax1, color="0.5")
    results["SimAgent"]["room"]["T_lower"].plot(ax=ax1, color="0.5")
    results["SimAgent"]["room"]["T_out"].plot(ax=ax1, color=mpcplot.EBCColors.dark_grey)
    mpc_at_time_step(
        data=results["NegFlexMPC"]["NegFlexMPC"], time_step=9000, variable="T"
    ).plot(ax=ax1, label="neg", linestyle="--", color=mpcplot.EBCColors.red)
    mpc_at_time_step(
        data=results["PosFlexMPC"]["PosFlexMPC"], time_step=9000, variable="T"
    ).plot(ax=ax1, label="pos", linestyle="--", color=mpcplot.EBCColors.blue)
    mpc_at_time_step(
        data=results["FlexModel"]["Baseline"], time_step=9900, variable="T"
    ).plot(ax=ax1, label="base", linestyle="--", color=mpcplot.EBCColors.dark_grey)

    ax1.legend()
    ax1.vlines(9000, ymin=0, ymax=500, colors="black")
    ax1.vlines(9900, ymin=0, ymax=500, colors="black")
    ax1.vlines(10800, ymin=0, ymax=500, colors="black")
    ax1.vlines(18000, ymin=0, ymax=500, colors="black")

    ax1.set_ylim(289, 299)
    x_ticks = np.arange(0, 3600 * 4 + 1, 3600)
    x_tick_labels = [int(tick / 3600) for tick in x_ticks]
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(x_tick_labels)
    ax1.set_xlabel("Time in hours")
    for ax in axs:
        mpcplot.make_grid(ax)
        ax.set_xlim(0, 3600 * 4)

    # predictions
    fig, axs = mpcplot.make_fig(style=mpcplot.Style(use_tex=False), rows=2)
    (ax1, ax2) = axs
    # P_el
    ax1.set_ylabel("$P_{el}$ in kW")
    results["SimAgent"]["room"]["P_el"].plot(ax=ax1, color=mpcplot.EBCColors.dark_grey)
    mpc_at_time_step(
        data=results["NegFlexMPC"]["NegFlexMPC"], time_step=9000, variable="P_el"
    ).ffill().plot(
        ax=ax1,
        drawstyle="steps-post",
        label="neg",
        linestyle="--",
        color=mpcplot.EBCColors.red,
    )
    mpc_at_time_step(
        data=results["PosFlexMPC"]["PosFlexMPC"], time_step=9000, variable="P_el"
    ).ffill().plot(
        ax=ax1,
        drawstyle="steps-post",
        label="pos",
        linestyle="--",
        color=mpcplot.EBCColors.blue,
    )
    mpc_at_time_step(
        data=results["FlexModel"]["Baseline"], time_step=9900, variable="P_el"
    ).ffill().plot(
        ax=ax1,
        drawstyle="steps-post",
        label="base",
        linestyle="--",
        color=mpcplot.EBCColors.dark_grey,
    )
    ax1.legend()
    ax1.vlines(9000, ymin=-1000, ymax=5000, colors="black")
    ax1.vlines(9900, ymin=-1000, ymax=5000, colors="black")
    ax1.vlines(10800, ymin=-1000, ymax=5000, colors="black")
    ax1.vlines(18000, ymin=-1000, ymax=5000, colors="black")
    ax1.set_ylim(-0.1, 1)

    # mdot
    ax2.set_ylabel("$\dot{m}$ in kg/s")
    results["SimAgent"]["room"]["mDot"].plot(ax=ax2, color=mpcplot.EBCColors.dark_grey)
    mpc_at_time_step(
        data=results["NegFlexMPC"]["NegFlexMPC"], time_step=9000, variable="mDot"
    ).ffill().plot(
        ax=ax2,
        drawstyle="steps-post",
        label="neg",
        linestyle="--",
        color=mpcplot.EBCColors.red,
    )
    mpc_at_time_step(
        data=results["PosFlexMPC"]["PosFlexMPC"], time_step=9000, variable="mDot"
    ).ffill().plot(
        ax=ax2,
        drawstyle="steps-post",
        label="pos",
        linestyle="--",
        color=mpcplot.EBCColors.blue,
    )
    mpc_at_time_step(
        data=results["FlexModel"]["Baseline"], time_step=9900, variable="mDot"
    ).ffill().plot(
        ax=ax2,
        drawstyle="steps-post",
        label="base",
        linestyle="--",
        color=mpcplot.EBCColors.dark_grey,
    )

    ax2.legend()
    ax2.vlines(9000, ymin=0, ymax=500, colors="black")
    ax2.vlines(9900, ymin=0, ymax=500, colors="black")
    ax2.vlines(10800, ymin=0, ymax=500, colors="black")
    ax2.vlines(18000, ymin=0, ymax=500, colors="black")

    ax2.set_ylim(0, 0.06)

    x_ticks = np.arange(0, 3600 * 4 + 1, 3600)
    x_tick_labels = [int(tick / 3600) for tick in x_ticks]
    ax2.set_xticks(x_ticks)
    ax2.set_xticklabels(x_tick_labels)
    ax2.set_xlabel("Time in hours")
    for ax in axs:
        mpcplot.make_grid(ax)
        ax.set_xlim(0, 3600 * 4)

    # flexibility
    # get only the first prediction time of each time step
    ind_res = results["FlexibilityIndicator"]["FlexibilityIndicator"]
    energy_flex_neg = ind_res.xs("energyflex_neg", axis=1).droplevel(1).dropna()
    energy_flex_pos = ind_res.xs("energyflex_pos", axis=1).droplevel(1).dropna()
    fig, axs = mpcplot.make_fig(style=mpcplot.Style(use_tex=False), rows=1)
    ax1 = axs[0]
    ax1.set_ylabel("$\epsilon$ in kWh")
    energy_flex_neg.plot(ax=ax1, label="neg")
    energy_flex_pos.plot(ax=ax1, label="pos")
    energy_flex_neg.plot(ax=ax1, label="neg", color=mpcplot.EBCColors.red)
    energy_flex_pos.plot(ax=ax1, label="pos", color=mpcplot.EBCColors.blue)
    ax1.yaxis.set_major_formatter(FormatStrFormatter("%.4f"))

    ax1.legend()

    x_ticks = np.arange(0, 3600 * 4 + 1, 3600)
    x_tick_labels = [int(tick / 3600) for tick in x_ticks]
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(x_tick_labels)
    ax1.set_xlabel("Time in hours")
    for ax in axs:
        mpcplot.make_grid(ax)
        ax.set_xlim(0, 3600 * 4)

    plt.show()


if __name__ == "__main__":
    run_example(until)
