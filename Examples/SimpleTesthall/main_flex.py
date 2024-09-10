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

pd.set_option("display.max_rows", None)

_CONVERSION_MAP = {"seconds": 1, "minutes": 60, "hours": 3600, "days": 86400}
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
            else: continue
        if fname is not None:
            f.write(f"{time} {prediction} \n")
        actual_values[time] = prediction.iloc[0]
        prediction.index = (prediction.index + time) / _CONVERSION_MAP[convert_to]
    if fname is not None:
        f.close()
    return pd.Series(actual_values)


#Write Warnings
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
                    print('\033[91mWarning: Solver of', solver_name, 'not successful at time_step:', row[0], row[10], file=sys.stderr)
            ret.append(success)
    return ret


def run_example(agent_configs, env_config, with_plots=False, log_level=logging.INFO, initial_time=1728000, until=86400, time_step=900):
    # Set the log-level
    logging.basicConfig(level=log_level)
    mas = LocalMASAgency(
        agent_configs=agent_configs,
        env=env_config,
        variable_logging=False,
    )
    mas.run(until=until)

    # Write Warnings
    solver_stats = write_solver_warning()
    results = mas.get_results(cleanup=True)
    return results

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
        total_violation = integral(results["SimAgent"]["SimTestHall"]["T_out"] - results["SimAgent"]["SimTestHall"]["T_upper"], results["SimAgent"]["SimTestHall"]["T_upper"] < results["SimAgent"]["SimTestHall"]["T_out"])
        total_violation += integral(results["SimAgent"]["SimTestHall"]["T_lower"] - results["SimAgent"]["SimTestHall"]["T_out"], results["SimAgent"]["SimTestHall"]["T_lower"] > results["SimAgent"]["SimTestHall"]["T_out"])
        
        max_upper_violation = max((results["SimAgent"]["SimTestHall"]["T_out"] - results["SimAgent"]["SimTestHall"]["T_upper"]).dropna())
        max_lower_violation = max((results["SimAgent"]["SimTestHall"]["T_lower"] - results["SimAgent"]["SimTestHall"]["T_out"]).dropna())
    else:
        raise NotImplementedError
    
    return total_violation_prediction, total_violation, max_upper_violation, max_lower_violation


def set_mean_values(arr):
    def count_false_after_true(lst):
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
            mean_value = np.sum(non_nan_values)/m
            result.append(mean_value)
            result.extend(data[1:])
        else:
            result.extend(arr[i:i + m + 1])

    return result


def get_configs(predictor_config, mpc_config, flex_config, varying_price_signal=2, 
                start_day=2, duration=1, usecase="Winter",fmu=False):
    agent_configs = FlexAgentGenerator(flex_config=flex_config,  mpc_agent_config=mpc_config).generate_flex_agents()
    agent_configs.extend([predictor_config, mpc_config])
    if fmu:
        agent_configs.append("Model//local//fmu//config.json")
    else:
        agent_configs.append("Model//local//mpc//ca_simu.json")

    if "Summer" == usecase:
        initial_time = 16502400 + 86400 * start_day
    elif usecase == "Winter":
        initial_time = 172800 + 86400 * start_day

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


    until = initial_time + 86400 * duration

    env_config = {"rt": False, "t_sample": 100, "offset": initial_time}
    return agent_configs, env_config, initial_time, until, time_step

from matplotlib import pyplot as plt
if __name__ == "__main__":
    # change this to switch between different cases
    predictor_config = "Model//local//predictor//config.json"
    mpc_config = f"Model//local//mpc//config.json"
    flex_config = f"flexibility_agent_config.json"
    fname = None
    
    agent_configs, env_config, initial_time, until, time_step = get_configs(predictor_config, mpc_config, flex_config)
    results = run_example(agent_configs, env_config, initial_time=initial_time, until=until, time_step=time_step)
    if fname is not None:
        with open(fname, "w+b") as f:
            pickle.dump(results, f)


