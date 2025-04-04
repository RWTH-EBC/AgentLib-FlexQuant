import ast
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import agentlib_mpc.utils.plotting.basic as mpcplot
from agentlib_mpc.utils.analysis import mpc_at_time_step
from agentlib_mpc.utils.analysis import load_sim, load_mpc, load_mpc_stats
from matplotlib.pyplot import xticks
from tensorflow.python.platform.tf_logging import log_every_n

from flexibility_quantification.data_structures.flex_results import Results
from flexibility_quantification.utils.data_handling import strip_multi_index

results_wo_CasadiSimulator = "00_result_wo_casadimodel"
results_w_CasadiSimulator = "03_result_w_casadimodel_dt_300"
flex_config="flex_configs/flexibility_agent_config.json"
simulator_agent_config="mpc_and_sim/simple_sim.json"
dt=int(results_w_CasadiSimulator[-3:])

# res_wo_sim = Results(flex_config=flex_config,simulator_agent_config=simulator_agent_config,results=results_wo_CasadiSimulator)
# res_w_sim = Results(flex_config=flex_config,simulator_agent_config=simulator_agent_config,results=results_w_CasadiSimulator)
P_EL = 'P_el'
BASE_SIM = 'mpc_sim_base.csv'
POS_SIM = 'mpc_sim_pos_flex.csv'
NEG_SIM = 'mpc_sim_neg_flex.csv'
W_SIM_KEY = [BASE_SIM, POS_SIM, NEG_SIM]
INTERP_W_SIM_KEY = ['interp_' + key for key in W_SIM_KEY]
BASE_COLL = 'mpc_base.csv'
POS_COLL = 'mpc_pos_flex.csv'
NEG_COLL = 'mpc_neg_flex.csv'
WO_SIM_KEY = [BASE_COLL, POS_COLL, NEG_COLL]

csv_files_w_sim = [os.path.join(results_w_CasadiSimulator, f) for f in os.listdir(results_w_CasadiSimulator) if f.endswith(".csv")]
res_w_sim = {os.path.basename(f): pd.read_csv(f) for f in csv_files_w_sim}

csv_files_wo_sim = [os.path.join(results_wo_CasadiSimulator, f) for f in os.listdir(results_wo_CasadiSimulator) if f.endswith(".csv")]
res_wo_sim = {os.path.basename(f): pd.read_csv(f, header=[1]) for f in csv_files_wo_sim}

def re_index(df):
    index_tuples = [ast.literal_eval(idx) for idx in df.iloc[:,0]]
    multi_index = pd.MultiIndex.from_tuples(index_tuples, names=('time_step', 'time'))
    df_reindex = df.set_index(multi_index)
    df_reindex.rename(columns={df_reindex.columns[0]: 'multiindex'}, inplace=True)
    return df_reindex

for key in W_SIM_KEY:
    res_w_sim[key] = re_index(res_w_sim[key])

for key in WO_SIM_KEY:
    res_wo_sim[key] = re_index(res_wo_sim[key])

def interpolate_high_res(res_w_sim: pd.DataFrame, res_wo_sim: pd.DataFrame) -> pd.DataFrame:
    """Interpolate high resolution flex results."""
    high_res_time = mpc_at_time_step(res_w_sim[POS_SIM], time_step=0).index.tolist()
    low_res_time = mpc_at_time_step(res_wo_sim[POS_COLL], time_step=0).index.tolist()

    for k in [BASE_SIM, POS_SIM, NEG_SIM]:
        res_w_sim['interp_'+ k] = pd.DataFrame(index=res_wo_sim[POS_COLL].index, columns=res_w_sim[k].columns[1:])
        for outer_idx in np.unique(res_w_sim['interp_'+ k].index.get_level_values(0).tolist()):
            for inner_idx in range(len(low_res_time)):
                for col in res_w_sim['interp_'+ k].columns:
                    res_w_sim['interp_' + k].loc[(outer_idx, low_res_time[inner_idx]), col] = np.interp(low_res_time[inner_idx], high_res_time, mpc_at_time_step(res_w_sim[k], time_step=outer_idx)[col])

# interpolate high resolution to get corresponding value at low resolution
interpolate_high_res(res_w_sim, res_wo_sim)

def plot_all(variable:str, lb:float, ub:float, ylabel: str, dt:int):
    x_ub = len(res_wo_sim[POS_COLL][variable])
    fig, ax = plt.subplots(3, 1)
    ax[0].set_title(f'High resolution dt={dt}')
    
    # plot Baseline MPC
    res_w_sim['interp_'+BASE_SIM][variable].plot(ax=ax[0], label="$SIM_{base}$", legend=True)
    res_wo_sim[BASE_COLL][variable].plot(ax=ax[0], label="$COLL_{base}$", legend=True)
    
    # plot positive flexibility MPC
    res_w_sim['interp_'+POS_SIM][variable].plot(ax=ax[1], label="$SIM_{pos}$", legend=True)
    res_wo_sim[POS_COLL][variable].plot(ax=ax[1], label="$COLL_{pos}$", legend=True)
    
    # plot negative flexibility MPC
    res_w_sim['interp_'+NEG_SIM][variable].plot(ax=ax[2], label="$SIM_{neg}$", legend=True)
    res_wo_sim[NEG_COLL][variable].plot(ax=ax[2], label="$COLL_{neg}$", legend=True)
    
    for k, axis in zip(WO_SIM_KEY, ax):
        axis.set(xlim=(0, x_ub), xlabel="", xticks=[])
        axis.set(ylabel=ylabel, ylim=(lb, ub))
        # Get unique first-level index values
        unique_groups = res_wo_sim[k][variable].index.get_level_values(0).unique()
        # Find positions where each Group starts
        group_positions = [res_wo_sim[k][variable].index.get_level_values(0).tolist().index(group) for group in unique_groups]
        # Add vertical lines at group start positions
        for pos in group_positions:
            axis.axvline(x=pos, color='green', linestyle='--', linewidth=0.8)  # Adjust for spacing

def plot_one_step(variable: str, time_step:float, lb: float, ub: float, ylabel: str, dt:int):
    x_ub = mpc_at_time_step(res_wo_sim[BASE_COLL], time_step=0).index.tolist()[-1]
    t_mc = 900
    t_prep = t_mc+900
    t_event = t_prep+7200
    fig, ax = plt.subplots(3, 1)
    ax[0].set_title(f'High resolution dt={dt}, time_step={time_step}')

    # plot Baseline MPC
    res_w_sim['interp_' + BASE_SIM].loc[time_step,variable].plot(ax=ax[0], label="$SIM_{base}$", legend=True)
    res_wo_sim[BASE_COLL].loc[time_step,variable].plot(ax=ax[0], label="$COLL_{base}$", legend=True)
    ax[0].set(xlabel="", xticks=[])

    # # plot positive flexibility MPC
    res_w_sim['interp_' + POS_SIM].loc[time_step,variable].plot(ax=ax[1], label="$SIM_{pos}$", legend=True)
    res_wo_sim[POS_COLL].loc[time_step,variable].plot(ax=ax[1], label="$COLL_{pos}$", legend=True)
    ax[1].set(xlabel="", xticks=[])

    # # plot negative flexibility MPC
    res_w_sim['interp_' + NEG_SIM].loc[time_step,variable].plot(ax=ax[2], label="$SIM_{neg}$", legend=True)
    res_wo_sim[NEG_COLL].loc[time_step,variable].plot(ax=ax[2], label="$COLL_{neg}$", legend=True)

    for k, axis in zip(WO_SIM_KEY, ax):
        axis.set(ylabel=ylabel, ylim=(lb, ub))
        axis.set(xlim=(0, x_ub))
        # Add vertical lines at group start positions
        for t in [t_mc, t_prep, t_event]:
            axis.axvline(x=t, color='green', linestyle='--', linewidth=0.8)

plot_all('P_el', lb=-0.2, ub=1, ylabel="$P_{el}$ / kW", dt=dt)
plot_all('T_out', lb=280, ub=900, ylabel="$T_{out}$ / K", dt=dt)

one_step_time = 900
plot_one_step(variable='P_el', lb=-0.2, ub=1, ylabel="$P_{el}$ / kW", time_step=one_step_time, dt=dt)
plot_one_step(variable='T_out', lb=280, ub=300, ylabel="$T_{out}$ / K", time_step=one_step_time, dt=dt)

plt.show()