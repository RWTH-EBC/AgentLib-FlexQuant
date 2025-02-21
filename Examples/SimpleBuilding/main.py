import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from pathlib import Path
import logging
from flexibility_quantification.generate_flex_agents import FlexAgentGenerator
from agentlib.utils.multi_agent_system import LocalMASAgency
from flexibility_quantification.utils.interactive import Dashboard, CustomBound

logging.basicConfig(level=logging.WARN)
until = 3600 * 24 

ENV_CONFIG = {"rt": False, "factor": 0.002, "t_sample": 1} 

def run_example(until=until):

    os.chdir(Path(__file__).parent)

    mpc_config = "mpc_and_sim/simple_model.json"
    sim_config = "mpc_and_sim/fmu_config.json" 
    predictor_config = "predictor/predictor_config.json"
    flex_config = "flex_configs/flexibility_agent_config.json"

    config_list = FlexAgentGenerator(
        flex_config=flex_config, mpc_agent_config=mpc_config
    ).generate_flex_agents()

    agent_configs = [sim_config, predictor_config]
    agent_configs.extend(config_list)

    mas = LocalMASAgency(
        agent_configs=agent_configs, env=ENV_CONFIG, variable_logging=False
    )
    mas.run(until=until)

    results = []
    results = mas.get_results(cleanup=False)
    
    Dashboard( 
        flex_config=flex_config,
        simulator_agent_config=sim_config,
        results=results
    ).show(
        custom_bounds=CustomBound(
            for_variable="T_zone",
            lb_name="T_lower",
            ub_name="T_upper"
        )
    )    

if __name__ == "__main__":
    run_example(until)