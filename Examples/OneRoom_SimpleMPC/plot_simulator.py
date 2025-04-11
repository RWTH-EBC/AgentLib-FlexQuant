import ast
import copy
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from agentlib_mpc.utils.analysis import mpc_at_time_step

# define the path and extract time resolution
results_path = "001_result_w_casadimodel_dt_10"
dt = re.search(r'(\d+)$', results_path).group(1)
time_steps = 48
horizon_length = 43200

# load the results
BASE_SIM = 'mpc_sim_base.csv'
POS_SIM = 'mpc_sim_pos_flex.csv'
NEG_SIM = 'mpc_sim_neg_flex.csv'
W_SIM_KEY = [BASE_SIM, POS_SIM, NEG_SIM]
INTERP_W_SIM_KEY = ['interp_' + key for key in W_SIM_KEY]

BASE_COLL = 'mpc_base.csv'
POS_COLL = 'mpc_pos_flex.csv'
NEG_COLL = 'mpc_neg_flex.csv'
WO_SIM_KEY = [BASE_COLL, POS_COLL, NEG_COLL]

csv_files_w_sim = [os.path.join(results_path, f) for f in os.listdir(results_path) if f.endswith(".csv") and any(KEY in f for KEY in W_SIM_KEY)]
res_w_sim = {os.path.basename(f): pd.read_csv(f) for f in csv_files_w_sim}

csv_files_wo_sim = [os.path.join(results_path, f) for f in os.listdir(results_path) if f.endswith(".csv") and any(KEY in f for KEY in WO_SIM_KEY)]
res_wo_sim = {os.path.basename(f): pd.read_csv(f, header=[1]) for f in csv_files_wo_sim}

def re_index(df):
    """ set multiindex to the dataframe index """
    index_tuples = [ast.literal_eval(idx) for idx in df.iloc[:,0]]
    multi_index = pd.MultiIndex.from_tuples(index_tuples, names=('time_step', 'time'))
    df_reindex = df.set_index(multi_index)
    df_reindex.rename(columns={df_reindex.columns[0]: 'multiindex'}, inplace=True)
    return df_reindex

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

def plot_all(variable: str, lb: float, ub: float, ylabel: str, dt: int, interpolation: bool):
    fig, ax = plt.subplots(3, 1, sharex=True)
    ax[0].set_title(f'High resolution dt={dt}')

    if interpolation:
        sim_keys_for_plot = INTERP_W_SIM_KEY
    else:
        sim_keys_for_plot = W_SIM_KEY

    my_res_w_sim = copy.deepcopy(res_w_sim)
    my_res_wo_sim = copy.deepcopy(res_wo_sim)

    for key in sim_keys_for_plot:
        my_res_w_sim[key].index = [i[0]*time_steps+i[1] for i in my_res_w_sim[key].index]
    for key in WO_SIM_KEY:
        my_res_wo_sim[key].index = [i[0]*time_steps+i[1] for i in my_res_wo_sim[key].index]

    x_ub = int(my_res_w_sim[sim_keys_for_plot[0]].index[-1])

    # plot Baseline MPC
    my_res_w_sim[sim_keys_for_plot[0]][variable].plot(ax=ax[0], label="$SIM_{base}$", legend=True)
    my_res_wo_sim[BASE_COLL][variable].plot(ax=ax[0], label="$COLL_{base}$", legend=True)
    
    # plot positive flexibility MPC
    my_res_w_sim[sim_keys_for_plot[1]][variable].plot(ax=ax[1], label="$SIM_{pos}$", legend=True)
    my_res_wo_sim[POS_COLL][variable].plot(ax=ax[1], label="$COLL_{pos}$", legend=True)
    
    # plot negative flexibility MPC
    my_res_w_sim[sim_keys_for_plot[2]][variable].plot(ax=ax[2], label="$SIM_{neg}$", legend=True)
    my_res_wo_sim[NEG_COLL][variable].plot(ax=ax[2], label="$COLL_{neg}$", legend=True)
    
    for k, axis in zip(WO_SIM_KEY, ax):
        axis.set(xlim=(0, x_ub), xlabel="", xticks=[])
        axis.tick_params(axis='x', which='both', bottom=False, top=False)
        axis.set(ylabel=ylabel, ylim=(lb, ub))
        group_positions = list(range(0, horizon_length+x_ub, horizon_length))
        # Add vertical lines at group start positions
        for pos in group_positions:
            axis.axvline(x=pos, color='green', linestyle='--', linewidth=0.8)

        ax[2].set(xlabel="time / s", xticks=group_positions, xticklabels=[gp/time_steps for gp in group_positions])

def plot_one_step(variable: str, time_step: float, lb: float, ub: float, ylabel: str, dt: int, interpolation: bool):
    x_ub = mpc_at_time_step(res_wo_sim[BASE_COLL], time_step=0).index.tolist()[-1]
    t_mc = 900
    t_prep = t_mc+900
    t_event = t_prep+7200
    fig, ax = plt.subplots(3, 1, sharex=True)
    ax[0].set_title(f'High resolution dt={dt}, time_step={time_step}')

    if interpolation:
        keys_for_plot = INTERP_W_SIM_KEY
    else:
        keys_for_plot = W_SIM_KEY

    # plot Baseline MPC
    res_w_sim[keys_for_plot[0]].loc[time_step,variable].plot(ax=ax[0], label="$SIM_{base}$", legend=True)
    res_wo_sim[BASE_COLL].loc[time_step,variable].plot(ax=ax[0], label="$COLL_{base}$", legend=True)
    ax[0].set(xlabel="", xticks=[])

    # plot positive flexibility MPC
    res_w_sim[keys_for_plot[1]].loc[time_step,variable].plot(ax=ax[1], label="$SIM_{pos}$", legend=True)
    res_wo_sim[POS_COLL].loc[time_step,variable].plot(ax=ax[1], label="$COLL_{pos}$", legend=True)
    ax[1].set(xlabel="", xticks=[])

    # plot negative flexibility MPC
    res_w_sim[keys_for_plot[2]].loc[time_step,variable].plot(ax=ax[2], label="$SIM_{neg}$", legend=True)
    res_wo_sim[NEG_COLL].loc[time_step,variable].plot(ax=ax[2], label="$COLL_{neg}$", legend=True)
    ax[2].set(xlabel="time / s", xticks=[])

    for k, axis in zip(WO_SIM_KEY, ax):
        axis.set(ylabel=ylabel, ylim=(lb, ub))
        axis.set(xlim=(0, x_ub))
        axis.tick_params(axis='x', which='both', bottom=False, top=False)
        # Add vertical lines at group start positions
        for t in [t_mc, t_prep, t_event]:
            axis.axvline(x=t, color='green', linestyle='--', linewidth=0.8)

# re index the results dataframe
for key in W_SIM_KEY:
    res_w_sim[key] = re_index(res_w_sim[key])

for key in WO_SIM_KEY:
    res_wo_sim[key] = re_index(res_wo_sim[key])

# interpolate high resolution to get corresponding value at low resolution
interpolate_high_res(res_w_sim, res_wo_sim)

plot_all('P_el', lb=-0.2, ub=1, ylabel="$P_{el}$ / kW", dt=dt, interpolation=False)
plot_all('T_out', lb=290, ub=300, ylabel="$T_{out}$ / K", dt=dt, interpolation=False)

time_step_for_plot = 900
plot_one_step(variable='P_el', lb=-0.2, ub=1, ylabel="$P_{el}$ / kW", time_step=time_step_for_plot, dt=dt, interpolation=False)
plot_one_step(variable='T_out', lb=290, ub=300, ylabel="$T_{out}$ / K", time_step=time_step_for_plot, dt=dt, interpolation=False)

plt.show()