import logging
from flexibility_quantification.generate_flex_agents import FlexAgentGenerator
from agentlib.utils.multi_agent_system import LocalMASAgency
from flexibility_quantification.utils.interactive import show_flex_dashboard

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

    show_flex_dashboard(agent_configs_paths=agent_configs, results=results, timescale="hours")


if __name__ == "__main__":
   run_example(until)
   # show_flex_dashboard(
   #     # one way to get the agent config paths is by printing agent_configs after generating the agents
   #     agent_configs_paths=['mpc_and_sim/simple_sim.json', 'predictor/predictor_config.json', 'created_flex_files\\baseline.json', 'created_flex_files\\pos_flex.json', 'created_flex_files\\neg_flex.json', 'created_flex_files\\indicator.json', 'created_flex_files\\flexibility_market.json'],
   #     results=r"C:\Users\fwu-pkr\PycharmProjects\flexquant\Examples\OneRoom_SimpleMPC\results",
   #     timescale="hours"
   # )
