import matplotlib.pyplot as plt
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
from plot_results import plot_results_2
import pickle
import sys


# Set the log-level
logging.basicConfig(level=logging.WARN)
until = 2*21600

ENV_CONFIG = {"rt": False, "factor": 0.01, "t_sample": 60}


def run_example(until=until, offer_type="real"):
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
    results['until'] = until

    if results is None:
        sys.exit()

    with open(f'results/results_file_{offer_type}.pkl', 'wb') as results_file:
        pickle.dump(results, results_file)

    plot_results_2(results=results, offer_type=offer_type, until=until)


def clear_files(bClear_plots: bool = False, bClear_flex_files: bool = False, bClear_results: bool = False) -> None:
    print(">> Starting deletion of old generated files")

    rootPath = os.path.join(os.getcwd().split("flexquant", 1)[0], "flexquant")

    folders: list[str] = ["created_flex_files",
                          "plots",
                          "results"]

    if not bClear_plots:
        folders.remove("plots")
    if not bClear_results:
        folders.remove("results")
    if not bClear_flex_files:
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
        clear_files(bClear_plots=True, bClear_results=True, bClear_flex_files=True)

    offer_types: list[str] = ["neg", "pos", "real"]
    # offer_types: list[str] = ["neg"]
    for offer_type in offer_types:
        print(f'\n{"":-^50}')
        print(f'{f" Starting simulation with {offer_type} ":-^50}')
        print(f'{"":-^50}\n')

        # read market.json
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

        # read flexibility_agent_config.json
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

        # read simple_model.json
        file_path: str = os.path.join("mpc_and_sim", "simple_model.json")
        with open(file_path, "r") as model_config:
            model_config_data = json.load(model_config)
            model_config.close()

        model_config_data["modules"][1]["optimization_backend"]["results_file"] = f"results/mpc_{offer_type}.csv"

        # delete old json file
        os.remove(file_path)

        # create new simple_model.json
        with open(file_path, "w") as model_config:
            json.dump(model_config_data, model_config, indent=4)

        # read simple_sim.json
        file_path: str = os.path.join("mpc_and_sim", "simple_sim.json")
        with open(file_path, "r") as sim_config:
            sim_config_data = json.load(sim_config)
            sim_config.close()

        sim_config_data["modules"][1]["result_filename"] = f"results/sim_room_{offer_type}.csv"

        # delete old json file
        os.remove(file_path)

        # create new simple_sim.json
        with open(file_path, "w") as sim_config:
            json.dump(sim_config_data, sim_config, indent=4)

        run_example(until=until, offer_type=offer_type)

        print(f'\n{"":-^50}')
        print(f'{f" Finished simulation with {offer_type} ":-^50}')
        print(f'{"":-^50}\n')
