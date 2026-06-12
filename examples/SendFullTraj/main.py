import logging

from agentlib.utils.ray_broker import RayBrokerWrapper

from agentlib_flexquant.generate_flex_agents import FlexAgentGenerator
from agentlib_flexquant.utils.interactive import Dashboard, CustomBound
from agentlib.utils.multi_agent_system import LocalMASAgency, RayMAS

# Set the log-level
logging.basicConfig(level=logging.WARN)
until = 27900

ENV_CONFIG = {"rt": False, "factor": 0.01, "t_sample": 60}


def run_example(until=until):
    ray = False

    baseline_config = "Building_1_flex/baseline.json"
    pos_flex_config = "Building_1_flex/pos_flex.json"
    neg_flex_config = "Building_1_flex/neg_flex.json"
    indicator_config = "Building_1_flex/indicator.json"
    predictor_config = "Building_1_flex/predictor.json"
    user_config = "Building_1_flex/user.json"
    sim_config = "Building_1_flex/simulator.json"

    agent_configs = [baseline_config, pos_flex_config, neg_flex_config, indicator_config, predictor_config, user_config, sim_config]

    if ray:
        import ray
        ray.init()
        broker = RayBrokerWrapper()

        mas = RayMAS(
            agent_configs=agent_configs, env=ENV_CONFIG, variable_logging=False,
            broker=broker
        )
    else:
        mas = LocalMASAgency(
            agent_configs=agent_configs, env=ENV_CONFIG, variable_logging=False
        )

    mas.run(until=until)
    results = mas.get_results(cleanup=False)
    return results


if __name__ == "__main__":
    run_example(until)