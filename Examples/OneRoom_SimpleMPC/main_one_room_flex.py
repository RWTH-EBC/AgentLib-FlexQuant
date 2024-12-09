import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from pathlib import Path
from agentlib.utils.multi_agent_system import LocalMASAgency
import numpy as np
import agentlib_mpc.utils.plotting.basic as mpcplot
from agentlib_mpc.utils.analysis import load_sim, load_mpc
from agentlib_mpc.utils.analysis import mpc_at_time_step
from flexibility_quantification.generate_flex_agents import FlexAgentGenerator
import logging
import pandas as pd
import json
import os
import shutil


# Set the log-level
logging.basicConfig(level=logging.WARN)
until = 21600

ENV_CONFIG = {"rt": False, "factor": 0.01, "t_sample": 60}


def run_example(until=until, offer_type=None):
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
            # "indicator": {"admm_module": load_indicator(Path(res_path,
            # "flexibility_indicator.csv"))},
            # "market": {"admm_module": load_market(Path(res_path,
            # "flexibility_market.csv"))},
        }
        return results

    if results is None:
        results = load_results()

    # create the folder to store the figure
    Path("plots").mkdir(parents=True, exist_ok=True)
    Path(f"plots/plots_{offer_type}").mkdir(parents=True, exist_ok=True)

    # disturbances
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

    # save the figure
    plt.savefig(f"plots/plots_{offer_type}/disturbances.svg", format='svg')
    plt.close()

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

    ax1.set_ylim(284, 301)
    x_ticks = np.arange(0, 3600 * 6 + 1, 3600)
    x_tick_labels = [int(tick / 3600) for tick in x_ticks]
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(x_tick_labels)
    ax1.set_xlabel("Time in hours")
    for ax in axs:
        mpcplot.make_grid(ax)
        ax.set_xlim(0, 3600 * 6)

    # save the figure
    plt.savefig(f"plots/plots_{offer_type}/room_temp.svg", format='svg')
    plt.close()

    # predictions
    fig, axs = mpcplot.make_fig(style=mpcplot.Style(use_tex=False), rows=2)
    (ax1, ax2) = axs
    # P_el
    ax1.set_ylabel("$P_{el}$ in kW")
    results["SimAgent"]["room"]["P_el"].plot(ax=ax1, color=mpcplot.EBCColors.green)
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

    x_ticks = np.arange(0, 3600 * 6 + 1, 3600)
    x_tick_labels = [int(tick / 3600) for tick in x_ticks]
    ax2.set_xticks(x_ticks)
    ax2.set_xticklabels(x_tick_labels)
    ax2.set_xlabel("Time in hours")
    for ax in axs:
        mpcplot.make_grid(ax)
        ax.set_xlim(0, 3600 * 6)

    # save the figure
    plt.savefig(f"plots/plots_{offer_type}/predictions.svg", format='svg')
    plt.close()

    # flexibility
    # get only the first prediction time of each time step
    ind_res = results["FlexibilityIndicator"]["FlexibilityIndicator"]
    energy_flex_neg = ind_res.xs("energyflex_neg", axis=1).droplevel(1).dropna()
    energy_flex_pos = ind_res.xs("energyflex_pos", axis=1).droplevel(1).dropna()
    fig, axs = mpcplot.make_fig(style=mpcplot.Style(use_tex=False), rows=1)
    ax1 = axs[0]
    ax1.set_ylabel("$\epsilon$ in kWh")
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

    # save the figure
    plt.savefig(f"plots/plots_{offer_type}/flexibility.svg", format='svg')
    plt.close()


def clear_files(bClear_plots: bool = False, bClear_flex_files: bool = False, bClear_results: bool = False) -> None:
    print(">> Starting deletion of old generated files")

    rootPath = os.path.join(os.getcwd().split("flexquant", 1)[0], "flexquant")

    folders: list[str] = ["created_flex_files",
                          "plots",
                          "results"]

    if bClear_plots:
        folders.remove("plots")
    if bClear_results:
        folders.remove("results")
    if bClear_flex_files:
        folders.remove("created_flex_files")

    files: list[str] = ["nlp_hess_l.casadi"]

    bDebug: bool = True

    for folder in folders:
        path: str = os.path.join(rootPath, "Examples", "OneRoom_SimpleMPC", folder)
        if os.path.exists(path):
            if bDebug: print(f"{'deleting:':>14} {path}")
            shutil.rmtree(path=path)

    for file in files:
        path: str = os.path.join(rootPath, "Examples", "OneRoom_SimpleMPC", file)
        if os.path.exists(path):
            if bDebug: print(f"{'deleting:':>14} {path}")
            os.remove(path=path)

    print(">> successfully finished")


if __name__ == "__main__":
    bClearFiles: bool = True
    if bClearFiles:
        clear_files(bClear_plots=True, bClear_results=True, bClear_flex_files=False)

    offer_types: list[str] = ["neg", "pos", "average"]
    for offer_type in offer_types:
        print(f'\n{"":-^50}')
        print(f'{f" Starting simulation with {offer_type} ":-^50}')
        print(f'{"":-^50}\n')

        # Edit market.json
        file_path: str = os.path.join("flex_configs", "market.json")
        with open(file_path, "r") as market_file:
            market_data = json.load(market_file)
            market_file.close()

        market_data["agent_config"]["modules"][1]["market_specs"]["options"]["event_type"] = offer_type
        market_data["agent_config"]["modules"][1]["market_specs"]["options"]["results_file_offer"] = f"results/flex_offer_{offer_type}.csv"
        market_data["agent_config"]["modules"][1]["results_file"] = f"results/flexibility_market_{offer_type}.csv"

        # delete old json file
        os.remove(file_path)

        # create new market.json file
        with open(file_path, "w") as market_file:
            json.dump(market_data, market_file, indent=4)

        # Edit flexibility_agent_config.json
        file_path: str = os.path.join("flex_configs", "flexibility_agent_config.json")
        with open(file_path, "r") as flex_agent_config:
            flex_agent_data = json.load(flex_agent_config)
            flex_agent_config.close()

        flex_agent_data["indicator_config"]["agent_config"]["modules"][1]["results_file"] = f"results/flexibility_indicator_{offer_type}.csv"

        # delete old json file
        os.remove(file_path)

        # create new flexiblity_agent_config.json
        with open(file_path, "w") as flex_agent_config:
            json.dump(flex_agent_data, flex_agent_config, indent=4)

        run_example(until=until, offer_type=offer_type)

        print(f'\n{"":-^50}')
        print(f'{f" Finished simulation with {offer_type} ":-^50}')
        print(f'{"":-^50}\n')
