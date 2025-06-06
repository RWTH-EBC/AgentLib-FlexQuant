import os
import tempfile

import json
import logging
from copy import deepcopy

from agentlib.utils.multi_agent_system import LocalMASAgency

from flexibility_quantification.generate_flex_agents import FlexAgentGenerator
from flexibility_quantification.utils.interactive import Dashboard, CustomBound


logging.basicConfig(level=logging.WARN)
until = 3600 * 24 

ENV_CONFIG = {"rt": False, "factor": 0.002, "t_sample": 1} 

def update_configs(flex_config, flex_event_duration):
    """Loads and modifies a flex config, returns path to updated temporary config file."""
    
    # load config and create a copy
    with open(flex_config, "r") as f:
        flex_config = json.load(f)
    flex_config_copy = deepcopy(flex_config)
    
    # modify parameter and set custom base path for generated flex files
    flex_config_copy["flex_event_duration"] = flex_event_duration
    flex_config_copy["flex_base_directory_path"] = (
        f"C:/Users/fse-ksu/flex_test/run_with_flex_event_duration_{flex_event_duration}"
    )
    generated_flex_files_base_path = flex_config_copy["flex_base_directory_path"]
    
    # dump temp modified flex config for further processing
    temp_file = tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".json")
    json.dump(flex_config_copy, temp_file, indent=2)
    temp_file_path = temp_file.name
    temp_file.close()

    return temp_file_path, generated_flex_files_base_path

def plot_results(generated_flex_files_base_path, sim_config, flex_config):
    """Plots results for each simulation with specified flex event duration."""

    Dashboard( 
            flex_config=flex_config,
            simulator_agent_config=sim_config,
            generated_flex_files_base_path=generated_flex_files_base_path
        ).show(
            custom_bounds=CustomBound(
                for_variable="T_zone",
                lb_name="T_lower",
                ub_name="T_upper"
            )
        )

def run_example(flex_event_duration, until=until):
    """Runs MAS simulation with specified flex event duration."""

    mpc_config = "mpc_and_sim/simple_model.json"
    sim_config = "mpc_and_sim/fmu_config.json" 
    predictor_config = "predictor/predictor_config.json"
    flex_config = "flex_configs/flexibility_agent_config.json"

    updated_flex_config_path, generated_flex_files_base_path = update_configs(
        flex_config, flex_event_duration)

    generator = FlexAgentGenerator(
        flex_config=updated_flex_config_path, 
        mpc_agent_config=mpc_config
    )

    config_list = generator.generate_flex_agents()
    sim_config = generator.adapt_sim_results_path(sim_config)

    agent_configs = [sim_config, predictor_config, *config_list]

    mas = LocalMASAgency(
        agent_configs=agent_configs, env=ENV_CONFIG, variable_logging=False
    )
    mas.run(until=until)

    plot_results(generated_flex_files_base_path, sim_config, updated_flex_config_path)

    if os.path.exists(updated_flex_config_path):
            os.remove(updated_flex_config_path)

if __name__ == "__main__":
    flex_event_durations = [7200, 8100, 6300]
    for flex_event_duration in flex_event_durations:
        # press ctrl + C to close plot and start next run
        run_example(flex_event_duration, until)