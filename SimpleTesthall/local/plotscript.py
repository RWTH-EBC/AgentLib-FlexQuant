import itertools
import pandas as pd
import numpy as np
from RWTHColors import ColorManager
import matplotlib.pyplot as plt
import warnings
from local.utils import create_series
import os
from matplotlib.ticker import FuncFormatter, MultipleLocator
import tikzplotlib as tikz

warnings.filterwarnings("ignore", message="The label '_nolegend_' of <matplotlib.lines.Line2D object at")


def seconds_to_days(min_time: 0):
    def formatter(x, pos):
        return '{:.0f}'.format((x - min_time) / 86400)

    return formatter

def seconds_to_hours(min_time: 0):
    def formatter(x, pos):
        return '{:.0f}'.format((x - min_time) / 3600)

    return formatter


def create_mpc_series(results_mpc: pd.DataFrame, variable_mpc: str, value_type: str):
    res_mpc = results_mpc.copy()
    value = res_mpc[value_type][variable_mpc]
    series = value[value.index.get_level_values(1) == 0]
    series = series.reset_index(level=1, drop=True)
    return series


def create_mpc_one_predition_series(results_mpc: pd.DataFrame, variable_mpc: str, value_type: str, initial):
    res_mpc = results_mpc.copy()
    value = res_mpc[value_type][variable_mpc]
    series = value[value.index.get_level_values(0) == initial]
    series = series.reset_index(level=0, drop=True)
    return series


def create_pel_series_one_predition(results_mpc: pd.DataFrame, variable_mpc: str, initial):
    res_mpc = results_mpc.copy()
    value = res_mpc[variable_mpc]
    series = value[value.index.get_level_values(0) == initial]
    series = series.reset_index(level=0, drop=True)
    return series


def mpc_at_time_step(results: pd.DataFrame, type, var: str, time_step):
    data = results.xs(time_step, level=0)  # get values at each timestep
    if type == 'variable':
        selected_var = data.variable[var]
    elif type == 'parameter':
        selected_var = data.parameter[var]
    else:
        raise ValueError("Not known type")

    new_index = time_step + data.index.values

    df = selected_var.reset_index(drop=True).set_axis(new_index, axis=0)

    return df


# Colors
cm = ColorManager()
green1 = cm.RWTHGruen.p(100)
green2 = cm.RWTHGruen.p(75)
green3 = cm.RWTHGruen.p(50)
green4 = cm.RWTHGruen.p(25)
red = "#DD402D"  #EBC Rot
red_h = "#EB8C81"  #EBC rot heller
red_d1 = "#AC2B1C"  #EBC rot dunkler 1
red_d2 = "#721D13"  #EBC rot dunkler2
black1 = cm.RWTHSchwarz.p(100)
black2 = cm.RWTHSchwarz.p(75)
black3 = cm.RWTHSchwarz.p(50)
black4 = cm.RWTHSchwarz.p(25)
blau1 = cm.RWTHBlau.p(100)
blau3 = cm.RWTHBlau.p(50)
color_red = [red_d2, red_d1, red, red_h]
color_red_cycle = itertools.cycle(color_red)
color_green = [green1, green2, green3, green4]
color_green_cycle = itertools.cycle(color_green)


# Plots General
def t_zone_fmu(t_lower_series, t_upper_series, results, initial, until):
    """
    plot and compare room temperature T_Air from MPC and FMU
    Args:
        t_lower_series:
        t_upper_series:
        results:

    Returns:

    """
    plt.figure(figsize=(10, 4))
    plt.plot(t_lower_series, drawstyle="steps-post", label="Komfortgrenze", color=black4)
    plt.plot(t_upper_series, drawstyle="steps-post", label="_nolegend_", color=black4)
    t_zone_air_predict = create_mpc_series(results_mpc=results["myMPCAgent"]['myMPC'], variable_mpc="T_Air",
                                           value_type="variable")

    #plt.plot(results["SimAgent"]["SimTestHall"]["TAirRoom"], label="Simulation", color=red)
    plt.plot(t_zone_air_predict, label="Prediction", color=blau1)
    plt.ylabel(r"T$_{zone}$  in K")
    plt.xlabel('Zeit in Tagen')
    # plt.savefig('plots/t_zone_air.svg', format='svg')
    # tikz.clean_figure()
    # tikz.save("plots/t_zone_air.tex")

    # 设置 x 轴为天数显示，刻度间隔为1天
    #num_days = int((until - initial) / 86400) + 1  # 计算总天数
    #xticks_pos = [initial + day * 86400 for day in range(num_days)]  # 每天的秒数位置
    #xticks_labels = [f"Tag {day + 1}" for day in range(num_days)]  # 从Tag 1开始

    #plt.xticks(xticks_pos, xticks_labels)  # 设置x轴的刻度位置和标签

    plt.legend()
    if not os.path.exists('plots/MPC'):
        os.makedirs('plots/MPC', exist_ok=True)

    plt.savefig("plots/MPC/MPC_T_Air.svg", format='svg')
    tikz.save("plots/MPC/MPC_T_Air.tex")
    plt.show()


def t_parts_fmu(results, parts: str):
    """
    compare and plot one part temperature from mpc and fmu
    Args:
        results:
        parts:

    Returns:

    """
    plt.figure(figsize=(8, 6))
    t_zone_air_predict = create_mpc_series(results_mpc=results["myMPCAgent"]['myMPC'], variable_mpc=f"T_{parts}",
                                           value_type="variable")

    plt.plot(results["SimAgent"]["SimTestHall"][f"T_{parts}"], label="Simulation", color=red)
    plt.plot(t_zone_air_predict, label="Prediction", color=blau1)
    plt.ylabel(r"T$_{parts}$  in K")
    plt.xlabel('Zeit in Tagen')
    # plt.savefig('plots/t_zone_air.svg', format='svg')
    # tikz.clean_figure()
    # tikz.save("plots/t_zone_air.tex")
    plt.legend()
    plt.show()


def t_zone_casadi(results, initial):
    """
    compare results T_room from FMU simulator and casadi simulator
    Args:
        results:
        initial:

    Returns:

    """
    plt.figure(figsize=(8, 6))

    T_Air_SimAgent = results["SimAgent"]["SimTestHall"]["TAirRoom"]
    T_Air_Casadi = results["CasadiSimAgent"]["SimZone"]["T_Air"]

    merged_df = pd.merge(T_Air_SimAgent, T_Air_Casadi, left_index=True, right_index=True, how='outer')
    mae = np.mean(np.abs(merged_df["TAirRoom"] - merged_df["T_Air"]))
    mae_formatted = f'{mae:.5f} K'

    rmse = np.sqrt(np.mean((merged_df["TAirRoom"] - merged_df["T_Air"]) ** 2))
    rmse_formatted = f'{rmse:.5f} K'

    plt.plot(T_Air_SimAgent, label="fmu", color=blau1)
    plt.plot(T_Air_Casadi, label="casadi", color=red)

    plt.text(merged_df.index[-1], merged_df["TAirRoom"].iloc[-1], f'MAE: {mae_formatted}\nRMSE: {rmse_formatted}',
             verticalalignment='bottom',
             horizontalalignment='right')

    plt.ylabel(r"T$_{zone}$  in K")
    plt.xlabel('Zeit in Tagen')
    plt.legend()

    #plt.gca().xaxis.set_major_locator(MultipleLocator(3600))  # 每1h显示一个刻度
    #plt.gca().xaxis.set_major_formatter(FuncFormatter(seconds_to_hours(initial)))

    if not os.path.exists('plots'):
        os.makedirs('plots')

    # Saving the plot using a relative path
    plt.savefig(f"plots/T_Air.svg", format='svg')

    plt.show()


def t_parts(results, partsFMU, partsCasadi, initial):
    """
    compare and plot parts' temperature from FMU and Casadi
    Args:
        results:
        parts:

    Returns:

    """
    plt.figure(figsize=(8, 6))
    T_parts_SimAgent = results["SimAgent"]["SimTestHall"][partsFMU]
    T_parts_Casadi = results["CasadiSimAgent"]["SimZone"][partsCasadi]

    merged_df = pd.merge(T_parts_SimAgent, T_parts_Casadi, left_index=True, right_index=True, how='outer')
    mae = np.mean(np.abs(merged_df.iloc[:, 0] - merged_df.iloc[:, 1]))
    mae_formatted = f'{mae:.5f} K'

    plt.plot(T_parts_SimAgent, label=f"fmu {partsFMU}", color=blau1)
    plt.plot(T_parts_Casadi, label=f"casadi {partsCasadi}", color=red)
    plt.text(merged_df.index[-1], merged_df.iloc[-1, 0], f'MAE: {mae_formatted}',
             verticalalignment='bottom',
             horizontalalignment='right')

    plt.ylabel(r"T$_{part}$  in K")
    plt.xlabel('Zeit in Tagen')
    plt.legend()

    # change sec to day in x-axis
    plt.gca().xaxis.set_major_locator(MultipleLocator(86400))  # 每1天显示一个刻度
    plt.gca().xaxis.set_major_formatter(FuncFormatter(seconds_to_days(initial)))

    # Ensure the 'plots' directory exists
    if not os.path.exists('plots'):
        os.makedirs('plots')

    # Saving the plot using a relative path
    plt.savefig(f"plots/{partsFMU}.svg", format='svg')
    plt.show()


def Q_parts(results, partsFMU: str, partsCasadi: str, initial: int):
    plt.figure(figsize=(8, 6))
    # Get data
    Q_parts_SimAgent = results["SimAgent"]["SimTestHall"][partsFMU]
    Q_parts_Casadi = results["CasadiSimAgent"]["SimZone"][partsCasadi]
    if "Q_conv" in partsFMU or "Q_rad" in partsFMU:
        Q_parts_SimAgent = -Q_parts_SimAgent

    merged_df = pd.merge(Q_parts_SimAgent, Q_parts_Casadi, left_index=True, right_index=True, how='outer')
    mae = np.mean(np.abs(merged_df.iloc[:, 0] - merged_df.iloc[:, 1]))
    mae_formatted = f'{mae:.5f} W'

    plt.plot(Q_parts_SimAgent, label=f"fmu {partsFMU}", color=blau1)
    plt.plot(Q_parts_Casadi, label=f"casadi {partsCasadi}", color=red)
    plt.text(merged_df.index[-1], merged_df.iloc[-1, 0], f'MAE: {mae_formatted}',
             verticalalignment='bottom',
             horizontalalignment='right')

    plt.ylabel(r"Q$_{part}$  in W")
    plt.xlabel('Zeit in Tagen')
    plt.legend()

    # change sec to day in x-axis
    plt.gca().xaxis.set_major_locator(MultipleLocator(86400))  # 每1天显示一个刻度
    plt.gca().xaxis.set_major_formatter(FuncFormatter(seconds_to_days(initial)))
    # Ensure the 'plots' directory exists
    if not os.path.exists('plots'):
        os.makedirs('plots')

    # Saving the plot using a relative path
    plt.savefig(f"plots/{partsFMU}.svg", format='svg')
    plt.show()


def input_output_temp(results, initial, ts, until):
    plt.figure(figsize=(10, 4))
    t_flow_in_series = create_mpc_series(results_mpc=results["myMPCAgent"]['myMPC'], variable_mpc="T_flow_in",
                                         value_type="variable")
    T_flow_out_baseline_series = create_series.process_collocation_points(
        results_mpc=results["myMPCAgent"]['myMPC'],
        variable_mpc='T_flow_out_baseline',
        value_type='variable', ts=ts)
    plt.plot(t_flow_in_series, label="input temperature of radiator", color=red)
    plt.plot(T_flow_out_baseline_series, label="output temperature of radiator", color=black1)
    plt.ylabel(fr"T in K")
    plt.xlabel('Zeit in Tagen')
    plt.legend()

    # 设置 x 轴为天数显示，刻度间隔为1天
    num_days = int((until - initial) / 86400) + 1  # 计算总天数
    xticks_pos = [initial + day * 86400 for day in range(num_days)]  # 每天的秒数位置
    xticks_labels = [f"Tag {day + 1}" for day in range(num_days)]  # 从Tag 1开始

    plt.xticks(xticks_pos, xticks_labels)  # 设置x轴的刻度位置和标签

    # Ensure the 'plots' directory exists
    if not os.path.exists('plots/MPC'):
        os.makedirs('plots/MPC')

    # Saving the plot using a relative path
    plt.savefig(f"plots/MPC/MPC_SET_T_flow_in_and_out.svg", format='svg')
    tikz.save("plots/MPC/MPC_SET_T_flow_in_and_out.tex")
    plt.show()


def ambient(results, until, initial, time_step):
    """
    plot disturbance ambient temperature and solar radiation
    Args:
        results:
        until:
        initial:

    Returns:

    """
    fig, ax1 = plt.subplots(figsize=(10, 4))
    t = time_step
    n = int((until - initial) / t) - 1
    t_amb_mpc_list = [mpc_at_time_step(results=results["myMPCAgent"]['myMPC'], type='parameter', var='T_amb',
                                       time_step=initial + t * (i + 1)).ffill() for i in range(n)]
    q_rad_mpc_list = [mpc_at_time_step(results=results["myMPCAgent"]['myMPC'], type='parameter', var='Q_RadSol',
                                       time_step=initial + t * (i + 1)).ffill() for i in range(n)]
    t_amb_mpc = pd.Series({t_amb.index[0]: t_amb.iloc[0] for t_amb in t_amb_mpc_list})
    q_rad_mpc = pd.Series({q_rad.index[0]: q_rad.iloc[0] for q_rad in q_rad_mpc_list})

    # Plot ambient temperature on the primary y-axis
    ax1.plot(t_amb_mpc, label='Umgebungstemperatur', color=red)
    ax1.set_ylabel(r"T$_{amb}$ in K")
    ax1.tick_params(axis='y')
    # Create secondary y-axis for solar radiation
    ax2 = ax1.twinx()
    ax2.plot(q_rad_mpc, label='Solare Einstrahlung', color=blau1)
    ax2.set_ylabel('Solare Einstrahlung in W')
    ax2.tick_params(axis='y')

    # 设置x轴为天数显示，刻度间隔为1天
    num_days = int((until - initial) / 86400) + 1  # 计算总天数
    xticks_pos = [initial + day * 86400 for day in range(num_days)]  # 每天的秒数位置
    xticks_labels = [f"Tag {day + 1}" for day in range(num_days)]  # 从Tag 1开始

    plt.xticks(xticks_pos, xticks_labels)  # 设置x轴的刻度位置和标签

    # plt.xlim([(initial + 86400), until])
    # num_days = int(until / 86400) + 1
    # xticks_pos = [initial + day * 86400 for day in range(num_days)]
    # xticks_labels = [f"Tag{day + 1}" for day in range(num_days)]
    # plt.xticks(xticks_pos, xticks_labels)
    #tikz.clean_figure()
    #plt.savefig('plots/ambient.svg', format='svg')
    #tikz.save("plots/ambient.tex")
    # Set legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

    # Ensure the 'plots' directory exists
    if not os.path.exists('plots/MPC'):
        os.makedirs('plots/MPC')

    # Saving the plot using a relative path
    plt.savefig("plots/MPC/disturbance.svg", format='svg')
    tikz.save("plots/MPC/disturbance.tex")
    plt.show()


def qset_price(results, until, initial):
    fig, ax1 = plt.subplots(figsize=(8, 6))
    t = 900
    n = int((until - initial) / t) - 1

    Q_set_series = create_mpc_series(results_mpc=results["myMPCAgent"]['myMPC'], variable_mpc="QFlowSet",
                                     value_type="variable")
    pel_price_list = [mpc_at_time_step(results=results["myMPCAgent"]['myMPC'], type="parameter",
                                       var='r_pel', time_step=initial + t * (i + 1)).ffill() for i in range(n)]

    ax2 = ax1.twinx()
    ax1.plot(Q_set_series, label='QFlow$_{set}$', color="black", drawstyle="steps-post")
    #ax1.plot(p_el_series_maxpel, label='P$_{el,max}$', color=colors[("black", 25)])
    #ax1.plot(p_el_series_minpel, label='P$_{el,min}$', color=colors[("black", 25)])
    ax1.set_ylabel('Qflow_set in W')

    #ax2.plot(rpel, label='Power Price', color=colors[("blue",50)])
    #ax2.plot(price_list[0], label='Power Price Pred', color=blau1)
    #for i in range(1, n):
    #   ax2.plot(price_list[i], label='_nolegend_', color=blau1)
    #ax2.plot(price, label='Power Price Pred', color=blau1, drawstyle="steps-post")
    #for pel_price_gen in pel_price_list:
    #   ax2.plot(pel_price_gen, label='_nolegend_', color=blau1, drawstyle="steps-post", ls='--')
    ax2.plot(pel_price_list[0], label='el price', color=blau1, drawstyle="steps-post", ls='--')
    for pel_price_gen in pel_price_list:
        ax2.plot(pel_price_gen, label='_nolegend_', color=blau1, drawstyle="steps-post", ls='--')

    ax2.set_ylabel('el. prices in ct/kWh')

    # plt.xlim([(initial + 86400), until])
    # num_days = int((until-initial) / 86400) + 1
    # xticks_pos = [initial + day * 86400 for day in range(num_days)]
    # xticks_labels = [f"Tag{day + 1}" for day in range(num_days)]
    # plt.xticks(xticks_pos, xticks_labels)
    #plt.savefig('plots/p_el_price.svg', format='svg')
    #tikz.clean_figure()
    #tikz.save("plots/p_el_price.tex")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

    plt.show()


def p_el_price(p_el_series, r_pel_series, until, initial):
    """
    plot power Pel and electricity price in one fig.(baseline MPC)
    Args:
        results:
        p_el_series:
        until:
        initial:

    Returns:

    """
    fig, ax1 = plt.subplots(figsize=(10, 4))
    t = 900
    ax2 = ax1.twinx()

    ax1.plot(p_el_series, label='P$_{el}$', color=black1, drawstyle="steps-post")
    ax1.set_ylabel('Baseline Pel in W')

    ax2.plot(r_pel_series, label='Power Price', color=blau1)
    #ax2.plot(pel_price_list, label='Power Price Pred', color=blau1, drawstyle="steps-post")
    #ax2.plot(pel_price_list[0], label='Power Price Pred', color=blau1, drawstyle="steps-post", ls='--')
    #for pel_price_gen in pel_price_list:
    #    ax2.plot(pel_price_gen, label='_nolegend_', color=blau1, drawstyle="steps-post", ls='--')
    # for pel_price_gen in pel_price_list:
    #     ax2.plot(pel_price_gen, label='_nolegend_', color=colors[("blue", 25)], drawstyle="steps-post", ls='--')
    ax2.set_ylabel('Strompreis in ct/kWh')
    #plt.legend()
    # plt.xlim([(initial + 86400), until])
    # num_days = int((until-initial) / 86400) + 1
    # xticks_pos = [initial + day * 86400 for day in range(num_days)]
    # xticks_labels = [f"Tag{day + 1}" for day in range(num_days)]
    # plt.xticks(xticks_pos, xticks_labels)
    #plt.savefig('plots/p_el_price.svg', format='svg')

    num_days = int((until - initial) / 86400) + 1  # 计算总天数
    xticks_pos = [initial + day * 86400 for day in range(num_days)]  # 每天的秒数位置
    xticks_labels = [f"Tag {day + 1}" for day in range(num_days)]  # 从Tag 1开始
    plt.xticks(xticks_pos, xticks_labels)  # 设置x轴的刻度位置和标签

    ax1.set_xlabel('Zeit in Tagen')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
    #tikz.clean_figure()
    #tikz.save("plots/p_el_price.tex")
    if not os.path.exists('plots/MPC'):
        os.makedirs('plots/MPC', exist_ok=True)

    plt.savefig("plots/MPC/el_price.svg", format='svg')
    tikz.save("plots/MPC/el_price.tex")
    plt.show()


def t_rad_debug(t_rad_series, results, initial, until):
    """
    plot power Pel and electricity price in one fig.(baseline MPC)
    Args:
        results:
        p_el_series:
        until:
        initial:

    Returns:

    """
    plt.figure(figsize=(10, 4))

    plt.plot(results["SimAgent"]["SimTestHall"]["TRad"], label="Simulation", color=red)
    plt.plot(t_rad_series, label="Prediction", color=blau1)
    plt.ylabel(r"T$_{zone}$  in K")
    plt.xlabel('Zeit in Tagen')
    # plt.savefig('plots/t_zone_air.svg', format='svg')
    # tikz.clean_figure()
    # tikz.save("plots/t_zone_air.tex")
    plt.legend()

    # 设置x轴为天数显示，刻度间隔为1天
    num_days = int((until - initial) / 86400) + 1  # 计算总天数
    xticks_pos = [initial + day * 86400 for day in range(num_days)]  # 每天的秒数位置
    xticks_labels = [f"Tag {day + 1}" for day in range(num_days)]  # 从Tag 1开始

    plt.xticks(xticks_pos, xticks_labels)  # 设置x轴的刻度位置和标签

    plt.savefig("plots\MPC_T_rad.svg", format='svg')
    plt.show()

def t_out_debug(t_baseline_series, t_PF_series, t_NF_series, initial, until):

    plt.figure(figsize=(10, 4))

    plt.plot(t_PF_series, label="PF-MPC", color=blau1)
    plt.plot(t_NF_series, label="NF-MPC", color=red)
    plt.plot(t_baseline_series, label="Baseline", color=black1)
    plt.ylabel(r"T$_{flow_out}$  in K")
    plt.xlabel('Zeit in Tagen')
    # plt.savefig('plots/t_zone_air.svg', format='svg')
    # tikz.clean_figure()
    # tikz.save("plots/t_zone_air.tex")
    plt.legend()

    # 设置x轴为天数显示，刻度间隔为1天
    num_days = int((until - initial) / 86400) + 1  # 计算总天数
    xticks_pos = [initial + day * 86400 for day in range(num_days)]  # 每天的秒数位置
    xticks_labels = [f"Tag {day + 1}" for day in range(num_days)]  # 从Tag 1开始

    plt.xticks(xticks_pos, xticks_labels)  # 设置x轴的刻度位置和标签

    if not os.path.exists('plots/MPC'):
        os.makedirs('plots/MPC', exist_ok=True)

    plt.savefig("plots/MPC/MPC_T_flow_out.svg", format='svg')
    tikz.save("plots/MPC/MPC_T_flow_out.tex")
    plt.show()


def pel_all_mpcs(p_el_series, p_el_series_maxpel, p_el_series_minpel, price, until, initial):
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax2 = ax1.twinx()

    ax2.plot(p_el_series, label='P$_{el}$', color=black1, drawstyle="steps-post")
    ax2.plot(p_el_series_maxpel, label='P$_{el,max}$', color=red, drawstyle="steps-post")
    ax2.plot(p_el_series_minpel, label='P$_{el,min}$', color=blau1, drawstyle="steps-post")
    ax2.set_ylabel(r"Elektrische Leistung  in W")
    ax1.plot(price, label='Strompreis', color=black4)
    ax1.set_ylabel('Strompreis in ct/kWh')
    ax1.set_xlabel('Zeit in Tagen')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

    #num_days = int((until - initial) / 86400) + 1  # 计算总天数
    #xticks_pos = [initial + day * 86400 for day in range(num_days)]  # 每天的秒数位置
    #xticks_labels = [f"Tag {day + 1}" for day in range(num_days)]  # 从Tag 1开始
    #plt.xticks(xticks_pos, xticks_labels)

    if not os.path.exists('plots/flexs'):
        os.makedirs('plots/flexs', exist_ok=True)

    plt.savefig("plots/flexs/pel_min_max.svg", format='svg')
    tikz.save("plots/flexs/pel_min_max.tex")

    plt.show()


def setpoints(results, t_ahu_set_series, q_tabs_set_series, initial):
    plt.plot(results["SimAgent"]["SimTestHall"]["genericAHU1.genericAHUBus.TSupMea"], label="Istwert", color=blau1)
    #plt.plot(results["SimAgent"]["SimTestHall"]["TAhuSet"], label="Sollwert", ls='--', color=red_h)
    plt.plot(t_ahu_set_series, label="Sollwert", drawstyle="steps-post", color=red)
    plt.ylabel("T$_{RLT}$ in K")
    plt.xlabel("Zeit in Tagen")
    #plt.legend()
    tikz.clean_figure()
    tikz.save("plots/setAhu.tex")
    # plt.savefig('plots/setAhu.svg', format='svg')
    plt.show()

    plt.plot(results["SimAgent"]["AgentLogger"]["Q_Tabs_fmu"], label="Istwert", color=blau1)
    #plt.plot(results["SimAgent"]["SimTestHall"]["QFlowTabsSet"], label="Sollwert", ls='--', color=red_h)
    plt.plot(q_tabs_set_series / 1000, label="Sollwert", drawstyle="steps-post", color=red)
    plt.ylabel("Q$_{BKT}$ in kW")
    plt.xlabel("Zeit in Tagen")
    #plt.legend()
    # plt.savefig('plots/setBkt.svg', format='svg')
    tikz.clean_figure()
    tikz.save("plots/setBkt.tex")
    plt.show()


def power_distribution(results, until, initial):
    plt.plot(results["SimAgent"]["SimTestHall"]["Q_Tabs"], label="tabs", color=red)
    plt.plot(results["SimAgent"]["SimTestHall"]["Q_Ahu"], label="ahu", color=blau3)
    plt.ylabel("Leistung in kW")
    plt.xlabel("Zeit in Tagen")
    tikz.clean_figure()
    # plt.savefig('plots/tabs_ahu.svg', format='svg')
    tikz.save("plots/tabs_ahu.tex")
    plt.show()

    QFlowCold = [-x for x in results["SimAgent"]["SimTestHall"]["QFlowCold"]]
    QColdBKT = [-x for x in results["SimAgent"]["SimTestHall"]["coolEnergyCalc.y3"]]
    plt.plot(QFlowCold, label="total_cool", color=blau1)
    plt.plot(QColdBKT, label="cool_tabs", color=blau3)
    plt.xlabel("Zeit in Tagen")
    plt.ylabel("Leistung in kW")
    tikz.clean_figure()
    # plt.savefig('plots/cool.svg', format='svg')
    tikz.save("plots/cool.tex")
    plt.show()

    plt.plot(results["SimAgent"]["SimTestHall"]["QFlowHeat"], label="total_heat", color=red)
    plt.plot(results["SimAgent"]["SimTestHall"]["hotEnergyCalc.y3"], label="heat_tabs", color=blau3)
    # hotEnergyCalc.y2: heat_ahu
    plt.xlabel("Zeit in Tagen")
    plt.ylabel("Leistung in kW")
    tikz.clean_figure()
    # plt.savefig('plots/heat.svg', format='svg')
    tikz.save("plots/heat.tex")
    plt.show()


def plot_pel_pred(results, time_step, initial, until):
    plt.figure(figsize=(10, 4))
    t = time_step
    n = int((until - initial) / t) - 1
    pel = [mpc_at_time_step(results=results["myMPCAgent"]['myMPC'], type='variable', var="P_el_alg",
                            time_step=initial + t * (i + 1)) for i in range(n)]
    pel_max = [mpc_at_time_step(results=results["myMPCAgent_positive"]['myPFMPC'], type='variable', var="P_el_min_alg",
                                time_step=initial + t * (i + 1)) for i in range(n)]
    pel_min = [mpc_at_time_step(results=results["myMPCAgent_negative"]['myNFMPC'], type='variable', var="P_el_max_alg",
                                time_step=initial + t * (i + 1)) for i in range(n)]

    for i in range(n):
        plt.plot(pel[i], label='_nolegend_', color='black', drawstyle="steps-post")
        plt.plot(pel_max[i], label='_nolegend_', color=red, drawstyle="steps-post")
        plt.plot(pel_min[i], label='_nolegend_', color=blau1, drawstyle="steps-post")

    plt.xlabel('Zeit in Tagen')
    plt.ylabel('Leistung in kW')
    plt.show()


def t_zone_all_mpc(t_lower_series, t_upper_series, results, until, initial):
    """
        plot and compare room temperature T_Air from MPC and FMU
        Args:
            t_lower_series:
            t_upper_series:
            results:

        Returns:

        """
    plt.figure(figsize=(10, 4))
    plt.plot(t_lower_series, drawstyle="steps-post", label="Komortgrenze", color=black4)
    plt.plot(t_upper_series, drawstyle="steps-post", label="_nolegend_", color=black4)
    t_zone_air_base = create_mpc_series(results_mpc=results["myMPCAgent"]['myMPC'], variable_mpc="T_Air",
                                        value_type="variable")
    t_zone_air_nf = create_mpc_series(results_mpc=results['myMPCAgent_negative']['myNFMPC'], variable_mpc="T_Air",
                                      value_type="variable")
    t_zone_air_pf = create_mpc_series(results_mpc=results['myMPCAgent_positive']['myPFMPC'], variable_mpc="T_Air",
                                      value_type="variable")
    plt.plot(t_zone_air_base, label="Baseline", color=black1)
    plt.plot(t_zone_air_nf, label="NF-MPC", color=red)
    plt.plot(t_zone_air_pf, label="PF-MPC", color=blau1)

    plt.ylabel(r"T$_{zone}$  in K")
    plt.xlabel('Zeit in Tagen')
    # plt.savefig('plots/t_zone_air.svg', format='svg')
    # tikz.clean_figure()
    # tikz.save("plots/t_zone_air.tex")
    # 设置 x 轴为天数显示，刻度间隔为1天
    num_days = int((until - initial) / 86400) + 1  # 计算总天数
    xticks_pos = [initial + day * 86400 for day in range(num_days)]  # 每天的秒数位置
    xticks_labels = [f"Tag {day + 1}" for day in range(num_days)]  # 从Tag 1开始

    plt.xticks(xticks_pos, xticks_labels)  # 设置x轴的刻度位置和标签

    plt.legend()
    if not os.path.exists('plots/flexs'):
        os.makedirs('plots/flexs', exist_ok=True)

    plt.savefig("plots/flexs/T_Airroom_all_mpcs.svg", format='svg')
    tikz.save("plots/flexs/T_Airroom_all_mpcs.tex")
    plt.show()


def input_temp_all_mpc(results, initial, until):
    plt.figure(figsize=(10, 4))
    t_flow_in_series = create_mpc_series(results_mpc=results["myMPCAgent"]['myMPC'], variable_mpc="T_flow_in",
                                         value_type="variable")
    t_flow_in_min_series = create_mpc_series(results_mpc=results["myMPCAgent_positive"]['myPFMPC'],
                                             variable_mpc="T_flow_in_min",
                                             value_type="variable")
    t_flow_in_max_series = create_mpc_series(results_mpc=results["myMPCAgent_negative"]['myNFMPC'],
                                             variable_mpc="T_flow_in_max",
                                             value_type="variable")
    plt.plot(t_flow_in_series, label="Baseline", color=black1)
    plt.plot(t_flow_in_max_series, label="NF-MPC", color=red)
    plt.plot(t_flow_in_min_series, label="PF-MPC", color=blau1)

    plt.ylabel(fr"T$_input$  in K")
    plt.xlabel('Zeit in Tagen')
    plt.legend()

    # 设置 x 轴为天数显示，刻度间隔为1天
    num_days = int((until - initial) / 86400) + 1  # 计算总天数
    xticks_pos = [initial + day * 86400 for day in range(num_days)]  # 每天的秒数位置
    xticks_labels = [f"Tag {day + 1}" for day in range(num_days)]  # 从Tag 1开始

    plt.xticks(xticks_pos, xticks_labels)  # 设置x轴的刻度位置和标签

    # Ensure the 'plots' directory exists
    if not os.path.exists('plots/flexs'):
        os.makedirs('plots/flexs')

    # Saving the plot using a relative path
    plt.savefig(f"plots/flexs/T_flow_in.svg", format='svg')
    tikz.save("plots/flexs/T_flow_in.tex")
    plt.show()


# Plots KPI
def plot_energyflexibility_costs_bar(eps_neg, eps_pos, costs_neg, costs_pos, ts, initial_time, until):
    """
    plot positive and negative energy flex. and absolute cost separately.
    Args:
        eps_neg:
        eps_pos:
        costs_neg:
        costs_pos:
        ts:
        initial_time:

    Returns:

    """
    # Negative Powerflexibility
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    left_neg = ts / 2

    # Add the label only once by using a flag
    bar_label_added = False
    for i in range(len(eps_neg)):
        if not bar_label_added:
            ax1.bar(x=left_neg, height=eps_neg.values[i], width=ts, color=black4, label=r'$\epsilon_{neg}$')
            bar_label_added = True
        else:
            ax1.bar(x=left_neg, height=eps_neg.values[i], width=ts, color=black4)
        left_neg += ts
    ax1.set_xlabel("Zeit in Tagen")
    ax1.set_ylabel(r"$\epsilon_{neg}$ in kWh")
    # Create secondary y-axis for costs
    ax1_1 = ax1.twinx()
    ax1_1.plot(costs_neg, drawstyle="steps-post", label="$C_{abs,neg}$", color=red)
    ax1_1.set_ylabel(r"$C_{abs}$ in ct")

    # todo: only for test, delete the following line after all
    # ax1_1.set_ylim([250, 650])

    # Adding legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax1_1.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

    num_days = int((until - initial_time) / 86400) + 1  # 计算总天数
    xticks_pos = [0 + day * 86400 for day in range(num_days)]  # 每天的秒数位置
    xticks_labels = [f"Tag {day + 1}" for day in range(num_days)]  # 从Tag 1开始
    plt.xticks(xticks_pos, xticks_labels)

    if not os.path.exists('plots/flexs'):
        os.makedirs('plots/flexs', exist_ok=True)

    plt.savefig("plots/flexs/energ_cost_neg.svg", format='svg')
    tikz.save("plots/flexs/energ_cost_neg.tex")

    # Save or show the plot
    # plt.savefig('plots/energyflex_costs_neg.svg', format='svg')
    # tikz.save("plots/energyflex_costs_neg.tex")
    plt.show()

    # Positive Powerflexibility
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    left_pos = ts / 2
    # Add the label only once by using a flag
    bar_label_added = False
    for i in range(len(eps_pos)):
        if not bar_label_added:
            ax2.bar(x=left_pos, height=eps_pos.values[i], width=ts, color=black4, label=r'$\epsilon_{pos}$')
            bar_label_added = True
        else:
            ax2.bar(x=left_pos, height=eps_pos.values[i], width=ts, color=black4)
        left_pos += ts

    ax2.set_xlabel("Zeit in Tagen")
    ax2.set_ylabel(r"$\epsilon_{pos}$ in kWh")
    # Create secondary y-axis for costs
    ax2_2 = ax2.twinx()
    ax2_2.plot(costs_pos, drawstyle="steps-post", label="$C_{abs,pos}$", color=red)
    ax2_2.set_ylabel(r"$C_{abs}$ in ct")

    # 设置x轴为天数显示，刻度间隔为1天
    plt.xticks(xticks_pos, xticks_labels)  # 使用相同的刻度位置和标签

    # Adding legends
    lines_3, labels_3 = ax2.get_legend_handles_labels()
    lines_4, labels_4 = ax2_2.get_legend_handles_labels()
    ax2.legend(lines_3 + lines_4, labels_3 + labels_4, loc='upper left')

    if not os.path.exists('plots/flexs'):
        os.makedirs('plots/flexs', exist_ok=True)

    plt.savefig("plots/flexs/energ_cost_pos.svg", format='svg')
    tikz.save("plots/flexs/energ_cost_pos.tex")
    # Save or show the plot
    # plt.savefig('plots/energyflex_costs_pos.svg', format='svg')
    # tikz.save("plots/energyflex_costs_pos.tex")
    plt.show()


def plot_energyflexibility_costs(eps_neg, eps_pos, costs_neg, costs_pos, ts, initial_time):
    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax2 = ax1.twinx()
    #ax1.plot(eps_neg, drawstyle="steps-post", label='Negative Energieflexibilität', color=colors[("black", 25)])
    ax1.plot(costs_neg, marker='x', linestyle='None', label='_nolegend_', color=red)

    #ax1.set_ylabel("$\Delta$E in kWh")
    ax1.set_xlabel('Zeit in Tagen')
    #plt.savefig('plots/energyflex_costs_neg.svg', format='svg')
    #tikz.clean_figure()
    #tikz.save("plots/energyflex_costs_neg.tex")
    plt.show()

    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax2 = ax1.twinx()
    #ax1.plot(eps_pos, drawstyle="steps-post", label='Positive Energieflexibilität', color=colors[("black", 25)])
    ax1.plot(costs_pos, marker='x', linestyle='None', color=red)
    #ax1.set_ylabel("$\Delta$E in kWh")
    ax1.set_xlabel('Zeit in Tagen')
    #plt.savefig('plots/energyflex_costs_pos.svg', format='svg')
    #tikz.clean_figure()
    #tikz.save("plots/energyflex_costs_pos.tex")
    plt.show()


def plot_poweravg_min_max(pi_neg, pi_pos, pi_neg_max, pi_neg_min, pi_pos_max, pi_pos_min, until, initial_time):
    # Fig: 1 powerflex_avg
    fig, ax = plt.subplots(figsize=(10, 4))
    # Negative Werte umdrehen
    pi_neg = pi_neg.apply(lambda x: x * -1)
    # Durchschnittliche negative und positive Leistung plotten
    ax.plot(pi_neg, drawstyle="steps-post", label=r'$\pi_{neg, avg}$', color=red)
    ax.plot(pi_pos, drawstyle="steps-post", label=r'$\pi_{pos, avg}$', color=blau1)
    # Horizontale Linie bei y = 0
    ax.axhline(y=0, xmin=0, xmax=(until - initial_time), color=black2, linewidth=0.5)
    ax.set_ylabel(r"$\pi_{avg}$ in kW")
    ax.set_xlabel('Zeit in Tagen')
    ax.legend(loc='upper right')
    # Y-Achse begrenzen
    #y_max = max(pi_pos.values) + max(pi_pos.values) * 0.1
    #y_min = min(pi_neg.values) - min(pi_neg.values) * 0.1
    #ax.set_ylim(y_min, y_max)  # Korrigiert
    # X-Achse in Tagen anzeigen
    num_days = int((until - initial_time) / 86400) + 1  # 计算总天数
    xticks_pos = [0 + day * 86400 for day in range(num_days)]  # 每天的秒数位置
    xticks_labels = [f"Tag {day + 1}" for day in range(num_days)]  # 从Tag 1开始
    plt.xticks(xticks_pos, xticks_labels)

    if not os.path.exists('plots/flexs'):
        os.makedirs('plots/flexs', exist_ok=True)

    plt.savefig("plots/flexs/powerflex_avg.svg", format='svg')
    tikz.save("plots/flexs/powerflex_avg.tex")

    plt.show()

    #plt.hlines(y=0, xmin=0, xmax=until, color=colors[("black", 75)])
    # y_max = max(pi_pos.values)+max(pi_pos.values)*0.1
    # y_min = min(pi_neg.values)-min(pi_neg.values)*0.1
    # plt.ylim(y_min, y_max)
    #plt.savefig('plots/powerflex_avg.svg', format='svg')
    #tikz.clean_figure()
    #tikz.save("plots/powerflex_avg.tex")

    #Fig.2: powerflex_max_min
    fig, ax = plt.subplots(figsize=(10, 4))
    pi_neg_max = pi_neg_max.apply(lambda x: x * -1)
    #pi_neg_min = pi_neg_min.apply(lambda x: x * -1)
    ax.plot(pi_neg_max, label=r'$\pi_{neg, max}$', color=red)
    #ax.plot(pi_neg_min, marker='x', label=r'$\pi_{neg, min}$',color=red)
    ax.plot(pi_pos_max, label=r'$\pi_{pos, max}$', color=blau1)
    #ax.plot(pi_pos_min, marker='x', label=r'$\pi_{pos, min}$',color=blau1)
    ax.axhline(y=0, xmin=0, xmax=(until - initial_time), color=black2, linewidth=0.5)
    ax.set_ylabel(r"$\pi_{max}$ in kW")
    ax.set_xlabel('Zeit in Tagen')
    ax.legend(loc='upper right')
    #y_max = max(pi_pos_max.values) + max(pi_pos_max.values) * 0.2
    #y_min = min(pi_neg_max.values) - min(pi_neg_max.values) * 0.2
    #ax.set_ylim(y_min, y_max)
    plt.xticks(xticks_pos, xticks_labels)  # 设置x轴的刻度位置和标签

    if not os.path.exists('plots/flexs'):
        os.makedirs('plots/flexs', exist_ok=True)

    plt.savefig("plots/flexs/powerflex_max.svg", format='svg')
    tikz.save("plots/flexs/powerflex_max.tex")

    plt.show()
    #plt.hlines(y=0, xmin=0, xmax=until, color=colors[("black", 75)])
    # plt.ylim(y_min, y_max)
    #plt.savefig('plots/powerflex_min_max.svg', format='svg')
    #tikz.clean_figure()
    #tikz.save("plots/powerflex_min_max.tex")


def plot_energyflex_neg_pos(eps_neg, eps_pos, until, initial):
    """
    plot positive and negative energy flexibility
    Args:
        eps_neg:
        eps_pos:
        until:
        initial_time:

    Returns:

    """
    fig, ax = plt.subplots(figsize=(10, 4))
    eps_neg = eps_neg.apply(lambda x: x * -1)
    ax.plot(eps_neg, label=r'$\epsilon_{neg}$', color=red)
    ax.plot(eps_pos, label=r'$\epsilon_{pos}$', color=blau1)
    #plt.hlines(y=0, xmin=initial_time, xmax=until, color=colors[("black", 75)])
    ax.axhline(y=0, xmin=0, xmax=(until - initial), color=black2, linewidth=0.5)
    ax.set_ylabel(r"$\epsilon$ in kWh")
    ax.set_xlabel('Zeit in Tagen')
    ax.legend(loc='upper right')

    num_days = int((until - initial) / 86400) + 1  # 计算总天数
    xticks_pos = [0 + day * 86400 for day in range(num_days)]  # 每天的秒数位置
    xticks_labels = [f"Tag {day + 1}" for day in range(num_days)]  # 从Tag 1开始
    plt.xticks(xticks_pos, xticks_labels)  # 设置x轴的刻度位置和标签

    if not os.path.exists('plots/flexs'):
        os.makedirs('plots/flexs', exist_ok=True)

    plt.savefig("plots/flexs/energyflex.svg", format='svg')
    tikz.save("plots/flexs/energyflex.tex")
    # y_max = max(eps_pos.values) + max(eps_pos.values) * 0.1
    # y_min = min(eps_neg.values) - min(eps_neg.values) * 0.1
    # plt.ylim(y_min, y_max)
    #plt.savefig('plots/energyflex.svg', format='svg')
    #tikz.clean_figure()
    #tikz.save("plots/energyflex.tex")
    plt.show()


def plot_powerflex_neg_pos(results, until, initial):
    """
    plot positive and negative power flexibility
    Args:
        results:
        until:
        initial:

    Returns:

    """
    plt.figure()
    t = 900
    n = int((until - initial) / t) - 1
    #TODO: check plotting
    powerflex_neg = [
        create_series.create_flex_series(
            results=results["myFlexibilityAgent"]["myFlexibility"]["powerflex_flex_neg"],
            time_step=initial + t * (i + 1)
        ).ffill() * -1 for i in range(n)
    ]
    powerflex_pos = [
        create_series.create_flex_series(results=results["myFlexibilityAgent"]["myFlexibility"]["powerflex_flex_pos"],
                                         time_step=initial + t * (i + 1)).ffill() for i in range(n)]

    plt.plot(powerflex_pos[0], label='postive Leistungsflexibilität', color=red_d1, drawstyle="steps-post", ls='--')
    for pi_gen in powerflex_pos:
        plt.plot(pi_gen, label='_nolegend_', color=red_d1, drawstyle="steps-post", ls='--')

    plt.plot(powerflex_neg[0], label='negative Leistungsflexibilität', color=blau1, drawstyle="steps-post", ls='--')
    for pi_gen in powerflex_neg:
        plt.plot(pi_gen, label='_nolegend_', color=blau1, drawstyle="steps-post", ls='--')

    plt.ylabel('$\Delta$P in kW')
    plt.xlabel('Zeit in Tagen')

    plt.show()


def plot_timeflex(tau_neg, tau_pos):
    plt.plot(tau_neg, marker='x', linestyle='None', color=red)
    plt.plot(tau_pos, marker='x', linestyle='None', color=blau1)
    plt.ylabel("Zeitflexibilität in s")
    plt.xlabel('Zeit in Tagen')
    tikz.clean_figure()
    tikz.save("plots/timeflex.tex")
    plt.show()


def plot_costs_rel(costs_neg_rel, costs_pos_rel):
    plt.figure()
    plt.plot(costs_neg_rel, color=red, drawstyle="steps-post")
    plt.plot(costs_pos_rel, color=blau1, drawstyle="steps-post")
    plt.ylabel(r"Relative Kosten in ct/kWh")
    plt.xlabel('Zeit in Tagen')
    #tikz.clean_figure()
    #tikz.save("plots/costs_rel.tex")
    plt.show()


def plot_costs_price(costs_neg_rel, costs_pos_rel, r_pel_series, until, initial):
    r_pel_series.index = r_pel_series.index - initial
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(costs_neg_rel, label="$C_{rel,neg}$", color=red)
    ax1.plot(r_pel_series, label='Strompreis', color=black4)
    ax1.set_ylabel("Kosten in ct/kWh")
    ax1.set_xlabel('Zeit in Tagen')
    ax1.legend(loc='upper right')

    num_days = int((until - initial) / 86400) + 1  # 计算总天数
    xticks_pos = [0 + day * 86400 for day in range(num_days)]  # 每天的秒数位置
    xticks_labels = [f"Tag {day + 1}" for day in range(num_days)]  # 从Tag 1开始
    plt.xticks(xticks_pos, xticks_labels)  # 设置x轴的刻度位置和标签

    if not os.path.exists('plots/flexs'):
        os.makedirs('plots/flexs', exist_ok=True)

    plt.savefig(f"plots/flexs/costs_price_neg.svg", format='svg')
    tikz.save("plots/flexs/costs_price_neg.tex")
    plt.show()

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(costs_pos_rel, label="$C_{rel,pos}$", color=blau1)
    ax2.plot(r_pel_series, label='Strompreis', color=black4)
    ax2.set_ylabel("Kosten in ct/kWh")
    ax2.set_xlabel('Zeit in Tagen')
    ax2.legend(loc='upper right')

    plt.xticks(xticks_pos, xticks_labels)  # 设置x轴的刻度位置和标签

    plt.savefig("plots/flexs/costs_price_pos.svg", format='svg')
    tikz.save("plots/flexs/costs_price_pos.tex")

    plt.show()


def t_zone_all_mpc(t_lower_series, t_upper_series, results, until, initial):
    """
        plot and compare room temperature T_Air from MPC and FMU
        Args:
            t_lower_series:
            t_upper_series:
            results:

        Returns:

        """
    plt.figure(figsize=(10, 4))
    plt.plot(t_lower_series, drawstyle="steps-post", label="Komortgrenze", color=black4)
    plt.plot(t_upper_series, drawstyle="steps-post", label="_nolegend_", color=black4)
    t_zone_air_base = create_mpc_series(results_mpc=results["myMPCAgent"]['myMPC'], variable_mpc="T_Air",
                                        value_type="variable")
    t_zone_air_nf = create_mpc_series(results_mpc=results['myMPCAgent_negative']['myNFMPC'], variable_mpc="T_Air",
                                      value_type="variable")
    t_zone_air_pf = create_mpc_series(results_mpc=results['myMPCAgent_positive']['myPFMPC'], variable_mpc="T_Air",
                                      value_type="variable")
    plt.plot(t_zone_air_base, label="Baseline", color=black1)
    plt.plot(t_zone_air_nf, label="NF-MPC", color=red)
    plt.plot(t_zone_air_pf, label="PF-MPC", color=blau1)

    plt.ylabel(r"T$_{zone}$  in K")
    plt.xlabel('Zeit in Tagen')
    # plt.savefig('plots/t_zone_air.svg', format='svg')
    # tikz.clean_figure()
    # tikz.save("plots/t_zone_air.tex")
    # 设置 x 轴为天数显示，刻度间隔为1天
    num_days = int((until - initial) / 86400) + 1  # 计算总天数
    xticks_pos = [initial + day * 86400 for day in range(num_days)]  # 每天的秒数位置
    xticks_labels = [f"Tag {day + 1}" for day in range(num_days)]  # 从Tag 1开始

    plt.xticks(xticks_pos, xticks_labels)  # 设置x轴的刻度位置和标签

    plt.legend()
    if not os.path.exists('plots/flexs'):
        os.makedirs('plots/flexs', exist_ok=True)

    plt.savefig("plots/flexs/T_Airroom_all_mpcs.svg", format='svg')
    tikz.save("plots/flexs/T_Airroom_all_mpcs.tex")
    plt.show()


def t_zone_one_prediction(results, initial, time_step):
    """
    plot and compare room temperature T_Air from MPC and FMU
    Args:
        t_lower_series:
        t_upper_series:
        results:

    Returns:

    """
    plt.figure(figsize=(10, 4))
    t_upper_series = create_mpc_one_predition_series(results_mpc=results["myMPCAgent"]['myMPC'],
                                                     variable_mpc="T_upper", value_type="parameter",
                                                     initial=initial + time_step)
    t_lower_series = create_mpc_one_predition_series(results_mpc=results["myMPCAgent"]['myMPC'],
                                                     variable_mpc="T_lower", value_type="parameter",
                                                     initial=initial + time_step)
    t_zone_air_predict = create_mpc_one_predition_series(results_mpc=results["myMPCAgent"]['myMPC'],
                                                         variable_mpc="T_Air",
                                                         value_type="variable", initial=initial + time_step)
    t_zone_max = create_mpc_one_predition_series(results_mpc=results["myMPCAgent_negative"]['myNFMPC'],
                                                 variable_mpc="T_Air",
                                                 value_type="variable", initial=initial + time_step)
    t_zone_min = create_mpc_one_predition_series(results_mpc=results["myMPCAgent_positive"]['myPFMPC'],
                                                 variable_mpc="T_Air",
                                                 value_type="variable", initial=initial + time_step)
    t_lower_series = t_lower_series.dropna()
    t_upper_series = t_upper_series.dropna()
    plt.plot(t_lower_series, drawstyle="steps-post", label="Komortgrenze", color=black4)
    plt.plot(t_upper_series, drawstyle="steps-post", label="_nolegend_", color=black4)
    plt.plot(t_zone_air_predict, label="Baseline", color=black1)
    plt.plot(t_zone_min, label="PF-MPC", color=blau1)
    plt.plot(t_zone_max, label="NF-MPC", color=red)

    plt.ylabel(r"T$_{zone}$  in K")
    plt.xlabel('Zeit in Stunden')
    limit_time = t_zone_air_predict.index[-1]

    num_hours = int(limit_time / 3600) + 1  # 计算总天数
    xticks_pos = [0 + hours * 3600 for hours in range(num_hours)]  # 每天的秒数位置
    xticks_labels = [f"{hours}h" for hours in range(num_hours)]  # 从Tag 1开始

    plt.xticks(xticks_pos, xticks_labels)  # 设置x轴的刻度位置和标签

    plt.legend()
    if not os.path.exists('plots/flexs'):
        os.makedirs('plots/flexs', exist_ok=True)

    plt.savefig("plots/flexs/T_Air_one_prediction.svg", format='svg')
    tikz.save("plots/flexs/T_Air_one_prediction.tex")
    plt.show()


def p_el_one_prediction(results, initial, time_step):
    """
    plot and compare room temperature T_Air from MPC and FMU
    Args:
        t_lower_series:
        t_upper_series:
        results:

    Returns:

    """
    plt.figure(figsize=(10, 4))
    # = create_mpc_one_predition_series(results_mpc=results["myMPCAgent"]['myMPC'],
    # variable_mpc="T_upper", value_type="parameter",initial=initial+time_step)
    #t_lower_series = create_mpc_one_predition_series(results_mpc=results["myMPCAgent"]['myMPC'],
    #  variable_mpc="T_lower", value_type="parameter",initial=initial+time_step)

    pel = create_pel_series_one_predition(results_mpc=results["myFlexibilityAgent"]['myFlexibility'],
                                          variable_mpc="P_el_alg",
                                          initial=time_step)
    pel_max = create_pel_series_one_predition(results_mpc=results["myFlexibilityAgent"]['myFlexibility'],
                                              variable_mpc="P_el_max_alg",
                                              initial=time_step)
    pel_min = create_pel_series_one_predition(results_mpc=results["myFlexibilityAgent"]['myFlexibility'],
                                              variable_mpc="P_el_min_alg",
                                              initial=time_step)

    plt.plot(pel, label="Baseline", color=black1,drawstyle="steps-post")
    plt.plot(pel_min, label="PF-MPC", color=blau1,drawstyle="steps-post")
    plt.plot(pel_max, label="NF-MPC", color=red,drawstyle="steps-post")

    plt.ylabel(r"P$_{el}$  in W")
    plt.xlabel('Zeit in Stunden')
    limit_time = pel.index[-1]

    num_hours = int(limit_time / 3600) + 1  # num of hours
    xticks_pos = [0 + hours * 3600 for hours in range(num_hours)]
    xticks_labels = [f"{hours}h" for hours in range(num_hours)]  # 从Tag 1开始

    plt.xticks(xticks_pos, xticks_labels)  # 设置x轴的刻度位置和标签

    plt.legend()
    if not os.path.exists('plots/flexs'):
        os.makedirs('plots/flexs', exist_ok=True)

    plt.savefig("plots/flexs/pel_one_prediction.svg", format='svg')
    tikz.save("plots/flexs/pel_one_prediction.tex")
    plt.show()


def input_temp_all_mpc(results, initial, until):
    plt.figure(figsize=(10, 4))
    t_flow_in_series = create_mpc_series(results_mpc=results["myMPCAgent"]['myMPC'], variable_mpc="T_flow_in",
                                         value_type="variable")
    t_flow_in_min_series = create_mpc_series(results_mpc=results["myMPCAgent_positive"]['myPFMPC'],
                                             variable_mpc="T_flow_in_min",
                                             value_type="variable")
    t_flow_in_max_series = create_mpc_series(results_mpc=results["myMPCAgent_negative"]['myNFMPC'],
                                             variable_mpc="T_flow_in_max",
                                             value_type="variable")
    plt.plot(t_flow_in_series, label="Baseline", color=black1)
    plt.plot(t_flow_in_max_series, label="NF-MPC", color=red)
    plt.plot(t_flow_in_min_series, label="PF-MPC", color=blau1)

    plt.ylabel(fr"T$_input$  in K")
    plt.xlabel('Zeit in Tagen')
    plt.legend()

    # 设置 x 轴为天数显示，刻度间隔为1天
    num_days = int((until - initial) / 86400) + 1  # 计算总天数
    xticks_pos = [initial + day * 86400 for day in range(num_days)]  # 每天的秒数位置
    xticks_labels = [f"Tag {day + 1}" for day in range(num_days)]  # 从Tag 1开始

    plt.xticks(xticks_pos, xticks_labels)  # 设置x轴的刻度位置和标签

    # Ensure the 'plots' directory exists
    if not os.path.exists('plots/flexs'):
        os.makedirs('plots/flexs')

    # Saving the plot using a relative path
    plt.savefig(f"plots/flexs/T_flow_in.svg", format='svg')
    tikz.save("plots/flexs/T_flow_in.tex")
    plt.show()
