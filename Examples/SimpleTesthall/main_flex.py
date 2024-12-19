import logging
from pathlib import Path
import pandas as pd
from agentlib.utils.multi_agent_system import LocalMASAgency
from agentlib_mpc.utils.plotting import interactive
import ast
import numpy as np
import json
import sys
import os
import csv
import pickle
import logging
from copy import copy
from scipy import integrate
from flexibility_quantification.generate_flex_agents import FlexAgentGenerator
from flexibility_quantification.data_structures.flex_offer import OfferStatus
import shutil
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from agentlib_mpc.utils.analysis import mpc_at_time_step
import agentlib_mpc.utils.plotting.basic as mpcplot

pd.set_option("display.max_rows", None)

_CONVERSION_MAP = {"seconds": 1, "minutes": 60, "hours": 3600, "days": 86400}


def run_example(offer_type=None) -> None:
    # change this to switch between different cases
    mpc_config: str = "model/local/mpc/config.json"
    predictor_config: str = "model/local/predictor/config.json"
    flex_config: str = "flex_configs/flexibility_agent_config.json"

    # Config parameters:
    varying_price_signal: int = 2
    start_day: int = 2
    duration: int = 1
    use_case: str = "Winter"
    fmu: bool = False

    agent_configs, env_config, \
    initial_time, until, time_step = get_configs(predictor_config=predictor_config,
                                                 mpc_config=mpc_config,
                                                 flex_config=flex_config,
                                                 varying_price_signal=varying_price_signal,
                                                 start_day=start_day,
                                                 duration=duration,
                                                 use_case=use_case,
                                                 fmu=fmu)

    # Set the log-level
    logging.basicConfig(level=logging.INFO)

    mas = LocalMASAgency(
        agent_configs=agent_configs, env=env_config, variable_logging=False,
    )

    mas.run(until=until)

    # Write Warnings
    solver_stats = write_solver_warning()
    results = mas.get_results(cleanup=False)
    if results is None:
        sys.exit()

    # create the folder to store the figure
    Path("plots").mkdir(parents=True, exist_ok=True)
    Path(f"plots/plots_{offer_type}").mkdir(parents=True, exist_ok=True)

    # room temp
    fig, axs = mpcplot.make_fig(style=mpcplot.Style(use_tex=False), rows=1)
    ax1 = axs[0]
    fig.set_figwidth(13)
    # T out
    ax1.set_ylabel("$T_{room}$ in K")
    results["SimAgent"]["SimTestHall"]["T_upper"].plot(ax=ax1, color="0.5")
    results["SimAgent"]["SimTestHall"]["T_lower"].plot(ax=ax1, color="0.5")
    results["SimAgent"]["SimTestHall"]["T_out"].plot(ax=ax1, color=mpcplot.EBCColors.dark_grey)

    mpc_at_time_step(
        data=results["NegFlexMPC"]["NegFlexMPC"], time_step=initial_time + 9000, variable="T_out"
    ).plot(ax=ax1, label="neg", linestyle="--", color=mpcplot.EBCColors.red)
    mpc_at_time_step(
        data=results["PosFlexMPC"]["PosFlexMPC"], time_step=initial_time + 9000, variable="T_out"
    ).plot(ax=ax1, label="pos", linestyle="--", color=mpcplot.EBCColors.blue)
    mpc_at_time_step(
        data=results["myMPCAgent"]["myMPC"], time_step=initial_time + 9900, variable="T_out"
    ).plot(ax=ax1, label="base", linestyle="--", color=mpcplot.EBCColors.dark_grey)

    ax1.legend()
    ax1.vlines(initial_time + 9000, ymin=0, ymax=500, colors="black")
    ax1.vlines(initial_time + 9900, ymin=0, ymax=500, colors="black")
    ax1.vlines(initial_time + 10800, ymin=0, ymax=500, colors="black")
    ax1.vlines(initial_time + 18000, ymin=0, ymax=500, colors="black")

    ax1.set_ylim(284, 301)
    x_ticks = np.arange(initial_time, until, 3600)  # maybe also add 1 to until
    x_tick_labels = [int(tick / 3600) for tick in x_ticks]
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(x_tick_labels)
    ax1.set_xlabel("Time in hours")
    for ax in axs:
        mpcplot.make_grid(ax)
        ax.set_xlim(initial_time, until)

    # save the figure
    plt.savefig(f"plots/plots_{offer_type}/room_temp.svg", format='svg')
    plt.close()

    # predictions
    fig, axs = mpcplot.make_fig(style=mpcplot.Style(use_tex=False), rows=2)
    (ax1, ax2) = axs
    fig.set_figwidth(13)
    # P_el
    ax1.set_ylabel("$P_{el}$ in kW")
    results["SimAgent"]["SimTestHall"]["P_el_c"].plot(ax=ax1, color=mpcplot.EBCColors.green)
    mpc_at_time_step(
        data=results["NegFlexMPC"]["NegFlexMPC"], time_step=initial_time + 9000, variable="P_el_c"
    ).ffill().plot(
        ax=ax1,
        drawstyle="steps-post",
        label="neg",
        linestyle="--",
        color=mpcplot.EBCColors.red,
    )
    mpc_at_time_step(
        data=results["PosFlexMPC"]["PosFlexMPC"], time_step=initial_time + 9000, variable="P_el_c"
    ).ffill().plot(
        ax=ax1,
        drawstyle="steps-post",
        label="pos",
        linestyle="--",
        color=mpcplot.EBCColors.blue,
    )
    mpc_at_time_step(
        data=results["myMPCAgent"]["myMPC"], time_step=initial_time + 9900, variable="P_el_c"
    ).ffill().plot(
        ax=ax1,
        drawstyle="steps-post",
        label="base",
        linestyle="--",
        color=mpcplot.EBCColors.dark_grey,
    )

    ax1.legend()
    ax1.vlines(initial_time + 9000, ymin=-1000, ymax=5000, colors="black")
    ax1.vlines(initial_time + 9900, ymin=-1000, ymax=5000, colors="black")
    ax1.vlines(initial_time + 10800, ymin=-1000, ymax=5000, colors="black")
    ax1.vlines(initial_time + 18000, ymin=-1000, ymax=5000, colors="black")
    ax1.set_ylim(0, 2500)

    # Q_Ahu
    ax2.set_ylabel("Q_Ahu in W")
    results["SimAgent"]["SimTestHall"]["Q_Ahu"].plot(ax=ax2, color=mpcplot.EBCColors.dark_grey)
    mpc_at_time_step(
        data=results["NegFlexMPC"]["NegFlexMPC"], time_step=initial_time + 9000, variable="Q_Ahu"
    ).ffill().plot(
        ax=ax2,
        drawstyle="steps-post",
        label="neg",
        linestyle="--",
        color=mpcplot.EBCColors.red,
    )
    mpc_at_time_step(
        data=results["PosFlexMPC"]["PosFlexMPC"], time_step=initial_time + 9000, variable="Q_Ahu"
    ).ffill().plot(
        ax=ax2,
        drawstyle="steps-post",
        label="pos",
        linestyle="--",
        color=mpcplot.EBCColors.blue,
    )
    mpc_at_time_step(
        data=results["myMPCAgent"]["myMPC"], time_step=initial_time + 9900, variable="Q_Ahu"
    ).ffill().plot(
        ax=ax2,
        drawstyle="steps-post",
        label="base",
        linestyle="--",
        color=mpcplot.EBCColors.dark_grey,
    )

    ax2.legend()
    ax2.vlines(initial_time + 9000, ymin=0, ymax=500, colors="black")
    ax2.vlines(initial_time + 9900, ymin=0, ymax=500, colors="black")
    ax2.vlines(initial_time + 10800, ymin=0, ymax=500, colors="black")
    ax2.vlines(initial_time + 18000, ymin=0, ymax=500, colors="black")

    ax2.set_ylim(500, 3000)

    x_ticks = np.arange(initial_time, until, 3600)  # maybe also add 1 to until
    x_tick_labels = [int(tick / 3600) for tick in x_ticks]
    ax2.set_xticks(x_ticks)
    ax2.set_xticklabels(x_tick_labels)
    ax2.set_xlabel("Time in hours")
    for ax in axs:
        mpcplot.make_grid(ax)
        ax.set_xlim(initial_time, until)

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
    fig.set_figwidth(13)
    ax1.set_ylabel("$\epsilon$ in kWh")
    energy_flex_neg.plot(ax=ax1, label="neg")
    energy_flex_pos.plot(ax=ax1, label="pos")
    energy_flex_neg.plot(ax=ax1, label="neg", color=mpcplot.EBCColors.red)
    energy_flex_pos.plot(ax=ax1, label="pos", color=mpcplot.EBCColors.blue)
    ax1.yaxis.set_major_formatter(FormatStrFormatter("%.4f"))

    ax1.legend()

    x_ticks = np.arange(initial_time, until, 3600)  # maybe also add 1 to until
    x_tick_labels = [int(tick / 3600) for tick in x_ticks]
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(x_tick_labels)
    ax1.set_xlabel("Time in hours")
    for ax in axs:
        mpcplot.make_grid(ax)
        ax.set_xlim(initial_time, until)

    # save the figure
    plt.savefig(f"plots/plots_{offer_type}/flexibility.svg", format='svg')
    plt.close()


def get_series_from_predictions(series, convert_to="seconds", fname=None, return_first=False, index_of_return=0):
    actual_values: dict[float, float] = {}
    if fname is not None:
        f = open(fname, "w+")
    for i, (time, prediction) in enumerate(series.groupby(level=0)):
        time = time / _CONVERSION_MAP[convert_to]
        prediction: pd.Series = prediction.dropna().droplevel(0)
        if return_first:
            if i == index_of_return:
                return prediction
            else:
                continue
        if fname is not None:
            f.write(f"{time} {prediction} \n")
        actual_values[time] = prediction.iloc[0]
        prediction.index = (prediction.index + time) / _CONVERSION_MAP[convert_to]
    if fname is not None:
        f.close()
    return pd.Series(actual_values)


# Write Warnings
def write_solver_warning():
    """
    Returns a warning, if solver ist not successful
    Result-file paths are add manually

    Args:
        None
    Returns:
        Warning which indicates which MPC is not successful at which time
    """
    file_paths = {
        'results/stats_mpc_simple_building_local_broadcast_base.csv': 'MPC',
        'results/stats_mpc_simple_building_local_broadcast_neg_flex.csv': 'Max MPC',
        'results/stats_mpc_simple_building_local_broadcast_pos_flex.csv': 'Min MPC'
    }
    ret = []
    for file_path, solver_name in file_paths.items():
        with open(file_path) as file:
            csv_reader = csv.reader(file, delimiter=',')
            next(csv_reader)
            data = list(csv_reader)

            success = []
            for row in data:
                successful = row[11]
                success.append((row[0], successful, row[10]))

                if successful == 'False':
                    print('\033[91mWarning: Solver of', solver_name, 'not successful at time_step:', row[0], row[10],
                          file=sys.stderr)
            ret.append(success)
    return ret


def calculate_violation(results, fmu=False):
    total_violation_prediction = 0
    for time, flex_res in results["FlexibilityMarket"]["FlexibilityMarket"].groupby(level=0):
        if flex_res["Status"].iloc[0] == OfferStatus.accepted_positive.value:
            total_violation_prediction += flex_res["Positive Comfort Violation"][time].iloc[0]
        elif flex_res["Status"].iloc[0] == OfferStatus.accepted_negative.value:
            total_violation_prediction += flex_res["Negative Comfort Violation"][time].iloc[0]

        elif flex_res["Status"].iloc[0] == OfferStatus.not_accepted.value:
            continue

    def integral(arr, filt):
        arr = copy(arr)
        arr[~filt] = 0
        return integrate.trapezoid(arr.dropna())

    if not fmu:
        total_violation = integral(
            results["SimAgent"]["SimTestHall"]["T_out"] - results["SimAgent"]["SimTestHall"]["T_upper"],
            results["SimAgent"]["SimTestHall"]["T_upper"] < results["SimAgent"]["SimTestHall"]["T_out"])
        total_violation += integral(
            results["SimAgent"]["SimTestHall"]["T_lower"] - results["SimAgent"]["SimTestHall"]["T_out"],
            results["SimAgent"]["SimTestHall"]["T_lower"] > results["SimAgent"]["SimTestHall"]["T_out"])

        max_upper_violation = max(
            (results["SimAgent"]["SimTestHall"]["T_out"] - results["SimAgent"]["SimTestHall"]["T_upper"]).dropna())
        max_lower_violation = max(
            (results["SimAgent"]["SimTestHall"]["T_lower"] - results["SimAgent"]["SimTestHall"]["T_out"]).dropna())
    else:
        raise NotImplementedError

    return total_violation_prediction, total_violation, max_upper_violation, max_lower_violation


def set_mean_values(arr) -> list[float]:
    def count_false_after_true(lst) -> int:
        count = 0
        found_true = False
        for item in lst:
            if item:
                if found_true:
                    break
                found_true = True
            elif found_true:
                count += 1
        return count

    missing_indices = np.isnan(arr)
    m = count_false_after_true(missing_indices)
    result = []
    values = arr.values[:-1]

    for i in range(0, len(values), m + 1):
        if np.isnan(values[i]):
            data = values[i:i + m + 1]
            non_nan_values = np.nan_to_num(data, nan=0)
            mean_value = np.sum(non_nan_values) / m
            result.append(mean_value)
            result.extend(data[1:])
        else:
            result.extend(arr[i:i + m + 1])

    return result


def get_configs(predictor_config, mpc_config, flex_config, varying_price_signal, start_day, duration, use_case, fmu):
    agent_configs = FlexAgentGenerator(flex_config=flex_config, mpc_agent_config=mpc_config).generate_flex_agents()
    agent_configs.extend([predictor_config, mpc_config])

    if fmu:
        agent_configs.append("model/local/fmu/config.json")
    else:
        agent_configs.append("model/local/mpc/ca_simu.json")

    match use_case:
        case "Summer":
            initial_time = 16502400 + 86400 * start_day

        case "Winter":
            initial_time = 172800 + 86400 * start_day

        case _:
            print("Wrong use_case selected ! Exiting program ...")
            sys.exit(1)

    with open(mpc_config) as f:
        mpc_conf = json.load(f)
        with open(predictor_config) as f2:
            pred_conf = json.load(f2)
            for key in ("time_step", "prediction_horizon"):
                in_params = False
                for i, param in enumerate(pred_conf["modules"][1]["parameters"]):
                    if param["name"] == key:
                        if key == "time_step":
                            time_step = mpc_conf["modules"][1][key]
                        pred_conf["modules"][1]["parameters"][i]["value"] = mpc_conf["modules"][1][key]
                        in_params = True
                    if param["name"] == "varying_price_signal":
                        pred_conf["modules"][1]["parameters"][i]["value"] = varying_price_signal

                if not in_params:
                    pred_conf["modules"][1]["parameters"].append(
                        {key: mpc_conf["modules"][1][key]}
                    )

    with open(predictor_config, "w+") as f:
        json.dump(pred_conf, f, indent=4)

    until = initial_time + 86400 * duration * (0.25)

    env_config = {"rt": False, "t_sample": 100, "offset": initial_time}
    return agent_configs, env_config, initial_time, until, time_step


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

    # offer_types: list[str] = ["neg", "pos", "average"]
    offer_types: list[str] = ["neg"]
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
        market_data["agent_config"]["modules"][1]["market_specs"]["options"][
            "results_file_offer"] = f"results/flex_offer_{offer_type}.csv"
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

        flex_agent_data["indicator_config"]["agent_config"]["modules"][1][
            "results_file"] = f"results/flexibility_indicator_{offer_type}.csv"

        # delete old json file
        os.remove(file_path)

        # create new flexiblity_agent_config.json
        with open(file_path, "w") as flex_agent_config:
            json.dump(flex_agent_data, flex_agent_config, indent=4)

        run_example(offer_type=offer_type)

        print(f'\n{"":-^50}')
        print(f'{f" Finished simulation with {offer_type} ":-^50}')
        print(f'{"":-^50}\n')
