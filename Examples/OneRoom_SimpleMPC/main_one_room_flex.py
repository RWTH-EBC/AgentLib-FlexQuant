import matplotlib.pyplot as plt

from agentlib.utils.multi_agent_system import LocalMASAgency
import sys
import os
os.chdir(os.path.dirname(__file__))
sys.path.append(os.path.join("..",".."))
import agentlib_mpc.modules
from FlexibilityQuantification import generate_flex_agents

import pandas as pd


until = 5000

ENV_CONFIG = {"rt": False, "factor": 0.01, "t_sample": 60}
_CONVERSION_MAP = {"seconds": 1, "minutes": 60, "hours": 3600, "days": 86400}

## TODO:
def get_series_from_predictions(series, convert_to="seconds", fname=None, return_first=False, index_of_return=1):
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

def run_example(until=until, T_set=295):
    results = []
    mpc_config = "simple_model.json"
    flex_config = "flexibility_agent_config.json"
    mpc_config_pos, mpc_config_neg, mpc_config, indicator_config, provisor_config = generate_flex_agents.generate_flex_agents(mpc_config, flex_config)
    sim_config = "simple_sim.json"
    mas = LocalMASAgency(
        agent_configs=[
            mpc_config, mpc_config_pos, mpc_config_neg, sim_config, indicator_config, provisor_config],
        env=ENV_CONFIG,
        variable_logging=False
    )

    mas.run(until=until)
    results = mas.get_results()
    fig, ax = plt.subplots(3, 1, sharex=True)
    ax[0].plot(results["SimAgent"]["room"]["T_out"], color="black")
    ax[0].set_ylabel("T_out")

    ax[1].plot(results["SimAgent"]["room"]["mDot"], color="black")
    ax[1].set_ylabel("mDot")

    
    ax[2].plot(results["SimAgent"]["room"]["P_el"], color="black")
    ax[2].set_ylabel("P_el")

    ax[0].axhline(T_set, color="grey", linestyle="--", label="Set Point")
 
    get_series_from_predictions(results["FlexModel"]["FlexMPC"]["variable"]["T_out"]).plot(ax=ax[0])
    get_series_from_predictions(results["FlexModel"]["FlexMPC"]["variable"]["mDot"]).plot(ax=ax[1])
    get_series_from_predictions(results["FlexModel"]["FlexMPC"]["variable"]["P_el"]).plot(ax=ax[2])
    ax[0].legend(["Simulation", None, "Optimisation"])

    fig, ax = plt.subplots(4, 1, sharex=True)
    fig.suptitle(f"Simple MPC Model\nMPC Predictions made in t={3}ts")

    baseline = get_series_from_predictions(results["FlexModel"]["FlexMPC"]["variable"]["P_el"], return_first=True, index_of_return=2)

    pos_flex = (get_series_from_predictions(results["PosFlexMPC"]["PosFlexMPC"]["variable"]["P_el"], return_first=True, index_of_return=2) - baseline) 

    neg_flex = (get_series_from_predictions(results["NegFlexMPC"]["NegFlexMPC"]["variable"]["P_el"], return_first=True, index_of_return=2) - baseline)
    (baseline - baseline).plot(ax=ax[0], color="black")
    pos_flex.plot(ax=ax[0], color="red")
    neg_flex.plot(ax=ax[0], color="blue")

    # print(results[1]["switch_test"])
    get_series_from_predictions(results["FlexModel"]["FlexMPC"]["variable"]["mDot"], return_first=True, index_of_return=2).plot(ax=ax[1], color="black")
    get_series_from_predictions(results["PosFlexMPC"]["PosFlexMPC"]["variable"]["mDot"], return_first=True, index_of_return=2).plot(ax=ax[1], color="red")
    get_series_from_predictions(results["NegFlexMPC"]["NegFlexMPC"]["variable"]["mDot"], return_first=True, index_of_return=2).plot(ax=ax[1], color="blue")
    get_series_from_predictions(results["FlexModel"]["FlexMPC"]["variable"]["T"], return_first=True, index_of_return=2).plot(ax=ax[2], color="black")
    get_series_from_predictions(results["PosFlexMPC"]["PosFlexMPC"]["variable"]["T"], return_first=True, index_of_return=2).plot(ax=ax[2], color="red")
    get_series_from_predictions(results["NegFlexMPC"]["NegFlexMPC"]["variable"]["T"], return_first=True, index_of_return=2).plot(ax=ax[2], color="blue")
    
    baseline.plot(ax=ax[3], color="black")
    get_series_from_predictions(results["PosFlexMPC"]["PosFlexMPC"]["variable"]["P_el"], return_first=True, index_of_return=2).plot(ax=ax[3], color="red")
    get_series_from_predictions(results["NegFlexMPC"]["NegFlexMPC"]["variable"]["P_el"], return_first=True, index_of_return=2).plot(ax=ax[3], color="blue")

    # get_series_from_predictions(results["FlexModel"]["PosFlexModel"]["variable"]["switch_test"]).plot(ax=ax[4], color="blue")
    # fig.suptitle("Regelungsergebnisse")

    ax[0].legend(["Baseline", "Pos. Flex.", "Neg. Flex."])
    ax[2].set_xlabel("Simulation Time [sec]")
    ax[0].set_ylabel("Flexibility Prediction [W]")
    ax[1].set_ylabel("Mass Stream [kg/s]")
    ax[2].set_ylabel("Temperatur [K]")
    ax[3].set_ylabel("Leistung [W]")
    plt.show()


results = run_example(until, T_set=295)
plt.tight_layout()
input()
