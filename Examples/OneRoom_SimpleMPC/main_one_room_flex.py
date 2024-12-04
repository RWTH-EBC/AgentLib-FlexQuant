import logging
from flexibility_quantification.generate_flex_agents import FlexAgentGenerator
from agentlib.utils.multi_agent_system import LocalMASAgency
from flexibility_quantification.utils.interactive import Dashboard

# Set the log-level
logging.basicConfig(level=logging.WARN)
until = 21600

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

    Dashboard(
        flex_config="flex_configs/flexibility_agent_config.json",
        simulator_agent_config="mpc_and_sim/simple_sim.json",
        results=results
    ).show(
        temperature_var_name="T",
        ub_comfort_var_name="T_upper",
        lb_comfort_var_name="T_lower",
    )


if __name__ == "__main__":
    run_example(until)
