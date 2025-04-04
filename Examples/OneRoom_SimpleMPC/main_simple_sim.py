import logging
import matplotlib.pyplot as plt
from agentlib.utils.multi_agent_system import LocalMASAgency

# Set the log-level
logging.basicConfig(level=logging.WARN)
until = 43200

ENV_CONFIG = {"rt": False, "factor": 0.01, "t_sample": 60}


def run_example(until=until):
    results = []
    sim_config = "mpc_and_sim/simple_sim.json"
    predictor_config = "predictor/predictor_config.json"
    agent_configs = [sim_config, predictor_config]

    mas = LocalMASAgency(
        agent_configs=agent_configs, env=ENV_CONFIG, variable_logging=False
    )

    mas.run(until=until)
    results = mas.get_results(cleanup=False)

    return results['SimAgent']['room']


def plot_sim(res):
    xlb = 0
    xub = res.index[-1]
    dt = res.index[1]-res.index[0]
    fig, ax = plt.subplots(3, 1)

    # plot results
    res['mDot'].plot(ax=ax[0], label=r"$\dot{m}$", legend=True, xlabel="", xticks=[])
    res['T_out'].plot(ax=ax[1], label="$T_{out}$", legend=True, xlabel="", xticks=[])
    res['P_el'].plot(ax=ax[2], label="$P_{el}$", legend=True)

    # set x limit
    for a in ax:
        a.set(xlim=(xlb, xub))

    # set title of the figure
    ax[0].set_title(f'Resolution dt={dt}')

    plt.show()


if __name__ == "__main__":
    result_df = run_example(until)
    plot_sim(result_df)
