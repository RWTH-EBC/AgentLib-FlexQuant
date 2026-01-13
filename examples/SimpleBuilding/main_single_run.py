import logging
from agentlib_flexquant.generate_flex_agents import FlexAgentGenerator
from agentlib.utils.multi_agent_system import LocalMASAgency

logging.basicConfig(level=logging.WARN)
until = 3600 * 24 

ENV_CONFIG = {"rt": False, "factor": 0.002, "t_sample": 1}
sim_config = "mpc_and_sim/fmu_config.json"
mpc_config = "mpc_and_sim/simple_building.json"
predictor_config = "predictor/predictor_config.json"
flex_config = "flex_configs/flexibility_agent_config.json"


def run_example(until=until):

    generator = FlexAgentGenerator(
        flex_config=flex_config, mpc_agent_config=mpc_config
    )

    config_list = generator.generate_flex_agents()
    sim_config_new = generator.adapt_sim_results_path(sim_config, save_name_suffix="_temp")

    agent_configs = [sim_config_new, predictor_config]
    agent_configs.extend(config_list)

    mas = LocalMASAgency(
        agent_configs=agent_configs, env=ENV_CONFIG, variable_logging=False
    )
    mas.run(until=until)  
    results = mas.get_results(cleanup=False)
    return results


if __name__ == "__main__":
    # Here the simulation is run once, 
    # generated files are stored in --> the current working directory
    # For an example with multiple runs, see: examples\SimpleBuilding\main_multi_run.py
    # For plotting of results generated from this main file, 
    # see: examples\SimpleBuilding\plot_results_single.py
    run_example(until)
