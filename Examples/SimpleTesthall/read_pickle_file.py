import pickle
from pathlib import Path
import agentlib_mpc.utils.plotting.basic as mpcplot
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')

dStartDay = 2
dDays = 3
bDebugText = True
dFigureWidth = 20*dDays
bFMU = True


def create_plots(offer_type, pickle_file) -> None:
    with open(pickle_file, 'rb') as file:
        results = pickle.load(file)
        file.close()

    initial_time = 172800.0 + 86400.0 * dStartDay
    until = initial_time + 86400.0 * dDays

    offer_Time_steps: list = []
    Plot_files = list(Path(f"plots/plots_{offer_type}").glob('flex_offer_*_flexEnvelope.svg'))
    for iIdx in range(len(Plot_files)):
        strDigits = ''.join(char for char in Plot_files[iIdx].name if char.isdigit())
        offer_Time_steps.append(int(strDigits) / 10)
    offer_Time_steps.sort()

    # create the folder to store the figure
    Path("plots_REGEN").mkdir(parents=True, exist_ok=True)
    Path(f"plots_REGEN/plots_{offer_type}").mkdir(parents=True, exist_ok=True)

    if not bFMU:
        casadi_plots(results=results, initial_time=initial_time, until=until, offer_type=offer_type, offer_Time_steps=offer_Time_steps)
    else:
        fmu_plots(results=results, initial_time=initial_time, until=until, offer_type=offer_type, offer_Time_steps=offer_Time_steps)


def casadi_plots(results, initial_time, until, offer_type, offer_Time_steps) -> None:
    # room temp
    if bDebugText:
        print('Starting room temp plot ...')

    fig, axs = mpcplot.make_fig(style=mpcplot.Style(use_tex=False), rows=1)
    ax1 = axs[0]
    fig.set_figwidth(dFigureWidth)
    # T out
    ax1.set_ylabel("$T_{room}$ in K")
    results["SimAgent"]["SimTestHall"]["T_upper"].plot(ax=ax1, color="0.5")
    results["SimAgent"]["SimTestHall"]["T_lower"].plot(ax=ax1, color="0.5")
    results["SimAgent"]["SimTestHall"]["T_Air"].plot(ax=ax1, color=mpcplot.EBCColors.dark_grey)

    simData = results["SimAgent"]["SimTestHall"]["T_Air"]
    ax1.plot(simData.index, simData.values, 'g-', label='sim')
    for iIdx in range(len(offer_Time_steps)):
        negFlexData = results["NegFlexMPC"]["NegFlexMPC"]['variable']['T_Air'][offer_Time_steps[iIdx]].head(31)
        ax1.plot(negFlexData.index + offer_Time_steps[iIdx], negFlexData.ffill().values, 'r--', label='neg')
        posFlexData = results["PosFlexMPC"]["PosFlexMPC"]['variable']['T_Air'][offer_Time_steps[iIdx]].head(31)
        ax1.plot(posFlexData.index + offer_Time_steps[iIdx], posFlexData.ffill().values, 'b--', label='pos')
        #baseFlexData = results["myMPCAgent"]["Baseline"]['variable']['T_Air'][offer_Time_steps[iIdx] + 1800]#.head(31)
        #ax1.plot(baseFlexData.index + (offer_Time_steps[iIdx] + 1800), baseFlexData.ffill().values, 'k--', label='base')

        if iIdx == 0:
            ax1.legend()

        ax1.vlines(offer_Time_steps[iIdx], ymin=290, ymax=300, colors="black")
        ax1.vlines(offer_Time_steps[iIdx] + 1800, ymin=290, ymax=300, colors="black")
        ax1.vlines(offer_Time_steps[iIdx] + 1800 + 1800, ymin=290, ymax=300, colors="black")
        ax1.vlines(offer_Time_steps[iIdx] + 1800 + 1800 + 14400, ymin=290, ymax=300, colors="black")

    ax1.set_ylim(290, 299.30)
    x_ticks = np.arange(initial_time, until, 3600)  # maybe also add 1 to until
    x_tick_labels = [int(tick / 3600) for tick in x_ticks]
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(x_tick_labels)
    ax1.set_xlabel("Time in hours")
    for ax in axs:
        mpcplot.make_grid(ax)
        ax.set_xlim(initial_time, until)

    # save the figure
    plt.savefig(f"plots_REGEN/plots_{offer_type}/room_temp.svg", format='svg')
    plt.close()

    if bDebugText:
        print('Finished room temp plot')

    # predictions
    if bDebugText:
        print('Starting predictions plot ...')

    fig, axs = mpcplot.make_fig(style=mpcplot.Style(use_tex=False), rows=3)
    (ax1, ax2, ax3) = axs
    fig.set_figwidth(dFigureWidth)
    # P_el
    ax1.set_ylabel("$P_{el}$ in W")

    simData = results["SimAgent"]["SimTestHall"]["P_el_c"]
    ax1.plot(simData.index, simData.values, 'g-', label='sim')
    for iIdx in range(len(offer_Time_steps)):
        negFlexData = results["NegFlexMPC"]["NegFlexMPC"]['variable']['P_el_c'][offer_Time_steps[iIdx]].head(31)
        ax1.plot(negFlexData.index + offer_Time_steps[iIdx], negFlexData.ffill().values, 'r--', label='neg')
        posFlexData = results["PosFlexMPC"]["PosFlexMPC"]['variable']['P_el_c'][offer_Time_steps[iIdx]].head(31)
        ax1.plot(posFlexData.index + offer_Time_steps[iIdx], posFlexData.ffill().values, 'b--', label='pos')
        #baseFlexData = results["myMPCAgent"]["Baseline"]['variable']['P_el_c'][offer_Time_steps[iIdx] + 1800].head(31)
        #ax1.plot(baseFlexData.index + (offer_Time_steps[iIdx] + 1800), baseFlexData.ffill().values, 'k--', label='base')

        if iIdx == 0:
            ax1.legend()

        ax1.vlines(offer_Time_steps[iIdx], ymin=0, ymax=2.7, colors="black")
        ax1.vlines(offer_Time_steps[iIdx] + 1800, ymin=0, ymax=2.7, colors="black")
        ax1.vlines(offer_Time_steps[iIdx] + 1800 + 1800, ymin=0, ymax=2.7, colors="black")
        ax1.vlines(offer_Time_steps[iIdx] + 1800 + 1800 + 14400, ymin=0, ymax=2.7, colors="black")

    #ax1.set_ylim(0, 2700)

    # Q_Ahu
    ax2.set_ylabel("Q_Ahu in W")

    simData = results["SimAgent"]["SimTestHall"]["Q_Ahu"]
    ax2.plot(simData.index, simData.values, 'g-', label='sim')
    for iIdx in range(len(offer_Time_steps)):
        negFlexData = results["NegFlexMPC"]["NegFlexMPC"]['variable']['Q_Ahu'][offer_Time_steps[iIdx]].head(31)
        ax2.plot(negFlexData.index + offer_Time_steps[iIdx], negFlexData.ffill().values, 'r--', label='neg')
        posFlexData = results["PosFlexMPC"]["PosFlexMPC"]['variable']['Q_Ahu'][offer_Time_steps[iIdx]].head(31)
        ax2.plot(posFlexData.index + offer_Time_steps[iIdx], posFlexData.ffill().values, 'b--', label='pos')
        #baseFlexData = results["myMPCAgent"]["Baseline"]['variable']['Q_Ahu'][offer_Time_steps[iIdx] + 1800].head(31)
        #ax2.plot(posFlexData.index + (offer_Time_steps[iIdx] + 1800), baseFlexData.ffill().values, 'k--', label='base')

        if iIdx == 0:
            ax2.legend()

        ax2.vlines(offer_Time_steps[iIdx], ymin=-2000, ymax=2000, colors="black")
        ax2.vlines(offer_Time_steps[iIdx] + 1800, ymin=-2000, ymax=2000, colors="black")
        ax2.vlines(offer_Time_steps[iIdx] + 1800 + 1800, ymin=-2000, ymax=2000, colors="black")
        ax2.vlines(offer_Time_steps[iIdx] + 1800 + 1800 + 14400, ymin=-2000, ymax=2000, colors="black")

    #ax2.set_ylim(0, 3000)

    # Q_Tabs (BKT)
    ax3.set_ylabel("Q_Tabs in W")

    simData = results["SimAgent"]["SimTestHall"]["Q_Tabs_set"]
    ax3.plot(simData.index, simData.values, 'g-', label='sim')
    for iIdx in range(len(offer_Time_steps)):
        negFlexData = results["NegFlexMPC"]["NegFlexMPC"]['variable']['Q_Tabs_set'][offer_Time_steps[iIdx]].head(31)
        ax3.plot(negFlexData.index + offer_Time_steps[iIdx], negFlexData.ffill().values, 'r--', label='neg')
        posFlexData = results["PosFlexMPC"]["PosFlexMPC"]['variable']['Q_Tabs_set'][offer_Time_steps[iIdx]].head(31)
        ax3.plot(posFlexData.index + offer_Time_steps[iIdx], posFlexData.ffill().values, 'b--', label='pos')
        #baseFlexData = results["myMPCAgent"]["Baseline"]['variable']['Q_Tabs_set'][offer_Time_steps[iIdx] + 1800].head(31)
        #ax3.plot(posFlexData.index + (offer_Time_steps[iIdx] + 1800), baseFlexData.ffill().values, 'k--', label='base')

        if iIdx == 0:
            ax3.legend()

        ax3.vlines(offer_Time_steps[iIdx], ymin=0, ymax=5000, colors="black")
        ax3.vlines(offer_Time_steps[iIdx] + 1800, ymin=0, ymax=5000, colors="black")
        ax3.vlines(offer_Time_steps[iIdx] + 1800 + 1800, ymin=0, ymax=5000, colors="black")
        ax3.vlines(offer_Time_steps[iIdx] + 1800 + 1800 + 14400, ymin=0, ymax=5000, colors="black")

    #ax3.set_ylim(-100, 5000)

    x_ticks = np.arange(initial_time, until, 3600)  # maybe also add 1 to until
    x_tick_labels = [int(tick / 3600) for tick in x_ticks]
    ax3.set_xticks(x_ticks)
    ax3.set_xticklabels(x_tick_labels)
    ax3.set_xlabel("Time in hours")
    for ax in axs:
        mpcplot.make_grid(ax)
        ax.set_xlim(initial_time, until)

    # save the figure
    plt.savefig(f"plots_REGEN/plots_{offer_type}/predictions.svg", format='svg')
    plt.close()

    if bDebugText:
        print('Finished predictions plot')

    # Temp Predictions
    if bDebugText:
        print('Starting temp predictions plot ...')

    fig, axs = mpcplot.make_fig(style=mpcplot.Style(use_tex=False), rows=20)
    (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12, ax13, ax14, ax15, ax16, ax17, ax18, ax19, ax20) = axs
    fig.set_figwidth(dFigureWidth)
    fig.set_figheight(100)

    simData = results["SimAgent"]["SimTestHall"]["T_Air"]
    simData_T_upper = results["SimAgent"]["SimTestHall"]["T_upper"]
    simData_T_lower = results["SimAgent"]["SimTestHall"]["T_lower"]
    simData_T_out = results["SimAgent"]["SimTestHall"]["T_out"]

    iIdxOffset = 0
    for iIdx, ax in enumerate(axs):
        simData_T_upper.plot(ax=ax, color="0.5")
        simData_T_lower.plot(ax=ax, color="0.5")
        simData_T_out.plot(ax=ax, color=mpcplot.EBCColors.dark_grey)
        ax.set_ylabel("$T_{room}$ in K")
        ax.plot(simData.index, simData.values, 'g-', label='sim')

        baseFlexData = results["myMPCAgent"]["Baseline"]['variable']['T_Air'][initial_time + (3600 * (iIdx + iIdxOffset))]  # .head(31)
        ax.plot(baseFlexData.index + initial_time + (3600 * (iIdx + iIdxOffset)), baseFlexData.ffill().values, 'k--', label='base')

        if ax == axs[-1]:
            x_ticks = np.arange(initial_time, until, 3600)  # maybe also add 1 to until
            x_tick_labels = [int(tick / 3600) for tick in x_ticks]
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_tick_labels)
            ax.set_xlabel("Time in hours")

    for ax in axs:
        mpcplot.make_grid(ax)
        ax.set_xlim(initial_time, until)

    # save the figure
    plt.savefig(f"plots_REGEN/plots_{offer_type}/TEMP_predictions.svg", format='svg')
    plt.close()

    if bDebugText:
        print('Finished temp predictions plot')

    # Flex Temp Predictions
    if bDebugText:
        print('Starting flex temp predictions plot ...')

    fig, axs = mpcplot.make_fig(style=mpcplot.Style(use_tex=False), rows=20)
    (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12, ax13, ax14, ax15, ax16, ax17, ax18, ax19, ax20) = axs
    fig.set_figwidth(dFigureWidth)
    fig.set_figheight(100)

    simData = results["SimAgent"]["SimTestHall"]["T_Air"]
    simData_T_upper = results["SimAgent"]["SimTestHall"]["T_upper"]
    simData_T_lower = results["SimAgent"]["SimTestHall"]["T_lower"]
    simData_T_out = results["SimAgent"]["SimTestHall"]["T_out"]

    iIdxOffset = 0
    for iIdx, ax in enumerate(axs):
        simData_T_upper.plot(ax=ax, color="0.5")
        simData_T_lower.plot(ax=ax, color="0.5")
        simData_T_out.plot(ax=ax, color=mpcplot.EBCColors.dark_grey)
        ax.set_ylabel("$T_{room}$ in K")
        ax.plot(simData.index, simData.values, 'g-', label='sim')

        baseFlexData = results["myMPCAgent"]["Baseline"]['variable']['T_Air'][initial_time + (3600 * (iIdx + iIdxOffset))]  # .head(31)
        ax.plot(baseFlexData.index + initial_time + (3600 * (iIdx + iIdxOffset)), baseFlexData.ffill().values, 'k--', label='base')
        if not results["myMPCAgent"]["Baseline"]['parameter']['in_provision'][initial_time + (3600 * (iIdx + iIdxOffset))][0]:
            negFlexData = results["NegFlexMPC"]["NegFlexMPC"]['variable']['T_Air'][initial_time + (3600 * (iIdx + iIdxOffset))]  # .head(31)
            ax.plot(negFlexData.index + initial_time + (3600 * (iIdx + iIdxOffset)), negFlexData.ffill().values, 'r--', label='neg')
            posFlexData = results["PosFlexMPC"]["PosFlexMPC"]['variable']['T_Air'][initial_time + (3600 * (iIdx + iIdxOffset))]  # .head(31)
            ax.plot(posFlexData.index + initial_time + (3600 * (iIdx + iIdxOffset)), posFlexData.ffill().values, 'b--', label='pos')

        ax.legend()

        if ax == axs[-1]:
            x_ticks = np.arange(initial_time, until, 3600)  # maybe also add 1 to until
            x_tick_labels = [int(tick / 3600) for tick in x_ticks]
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_tick_labels)
            ax.set_xlabel("Time in hours")

    for ax in axs:
        mpcplot.make_grid(ax)
        ax.set_xlim(initial_time, until)

    # save the figure
    plt.savefig(f"plots_REGEN/plots_{offer_type}/FLEX_Temp_predictions.svg", format='svg')
    plt.close()

    if bDebugText:
        print('Finished flex temp predictions plot')

    # Flex Pel Predictions
    if bDebugText:
        print('Starting flex pel predictions plot ...')

    fig, axs = mpcplot.make_fig(style=mpcplot.Style(use_tex=False), rows=20)
    (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12, ax13, ax14, ax15, ax16, ax17, ax18, ax19, ax20) = axs
    fig.set_figwidth(dFigureWidth)
    fig.set_figheight(100)

    simData = results["SimAgent"]["SimTestHall"]["P_el_c"]

    iIdxOffset = 0
    for iIdx, ax in enumerate(axs):
        ax.set_ylabel("$P_{el,c}$ in W")
        ax.plot(simData.index, simData.values, 'g-', label='sim')

        baseFlexData = results["myMPCAgent"]["Baseline"]['variable']['P_el_c'][initial_time + (3600 * (iIdx + iIdxOffset))]  # .head(31)
        ax.plot(baseFlexData.index + initial_time + (3600 * (iIdx + iIdxOffset)), baseFlexData.ffill().values, 'k--', label='base')
        if not results["myMPCAgent"]["Baseline"]['parameter']['in_provision'][initial_time + (3600 * (iIdx + iIdxOffset))][0]:
            negFlexData = results["NegFlexMPC"]["NegFlexMPC"]['variable']['P_el_c'][initial_time + (3600 * (iIdx + iIdxOffset))]  # .head(31)
            ax.plot(negFlexData.index + initial_time + (3600 * (iIdx + iIdxOffset)), negFlexData.ffill().values, 'r--', label='neg')
            posFlexData = results["PosFlexMPC"]["PosFlexMPC"]['variable']['P_el_c'][initial_time + (3600 * (iIdx + iIdxOffset))]  # .head(31)
            ax.plot(posFlexData.index + initial_time + (3600 * (iIdx + iIdxOffset)), posFlexData.ffill().values, 'b--', label='pos')

        ax.legend()

        if ax == axs[-1]:
            x_ticks = np.arange(initial_time, until, 3600)  # maybe also add 1 to until
            x_tick_labels = [int(tick / 3600) for tick in x_ticks]
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_tick_labels)
            ax.set_xlabel("Time in hours")

    for ax in axs:
        mpcplot.make_grid(ax)
        ax.set_xlim(initial_time, until)

    # save the figure
    plt.savefig(f"plots_REGEN/plots_{offer_type}/FLEX_Pel_predictions.svg", format='svg')
    plt.close()

    if bDebugText:
        print('Finished flex pel predictions plot')

    # Flex TahuSet Predictions
    if bDebugText:
        print('Starting flex TahuSet predictions plot ...')

    fig, axs = mpcplot.make_fig(style=mpcplot.Style(use_tex=False), rows=20)
    (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12, ax13, ax14, ax15, ax16, ax17, ax18, ax19, ax20) = axs
    fig.set_figwidth(dFigureWidth)
    fig.set_figheight(100)

    simData = results["SimAgent"]["SimTestHall"]["T_ahu_set"]

    iIdxOffset = 0
    for iIdx, ax in enumerate(axs):
        ax.set_ylabel("$T_{ahu,set}$ in K")
        ax.plot(simData.index, simData.values, 'g-', label='sim')

        baseFlexData = results["myMPCAgent"]["Baseline"]['variable']['T_ahu_set'][initial_time + (3600 * (iIdx + iIdxOffset))]  # .head(31)
        ax.plot(baseFlexData.index + initial_time + (3600 * (iIdx + iIdxOffset)), baseFlexData.ffill().values, 'k--', label='base')
        if not results["myMPCAgent"]["Baseline"]['parameter']['in_provision'][initial_time + (3600 * (iIdx + iIdxOffset))][0]:
            negFlexData = results["NegFlexMPC"]["NegFlexMPC"]['variable']['T_ahu_set'][initial_time + (3600 * (iIdx + iIdxOffset))]  # .head(31)
            ax.plot(negFlexData.index + initial_time + (3600 * (iIdx + iIdxOffset)), negFlexData.ffill().values, 'r--', label='neg')
            posFlexData = results["PosFlexMPC"]["PosFlexMPC"]['variable']['T_ahu_set'][initial_time + (3600 * (iIdx + iIdxOffset))]  # .head(31)
            ax.plot(posFlexData.index + initial_time + (3600 * (iIdx + iIdxOffset)), posFlexData.ffill().values, 'b--', label='pos')

        ax.legend()

        if ax == axs[-1]:
            x_ticks = np.arange(initial_time, until, 3600)  # maybe also add 1 to until
            x_tick_labels = [int(tick / 3600) for tick in x_ticks]
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_tick_labels)
            ax.set_xlabel("Time in hours")

    for ax in axs:
        mpcplot.make_grid(ax)
        ax.set_xlim(initial_time, until)

    # save the figure
    plt.savefig(f"plots_REGEN/plots_{offer_type}/FLEX_TahuSet_predictions.svg", format='svg')
    plt.close()

    if bDebugText:
        print('Finished flex TahuSet predictions plot')

    # Flex QTabsSet Predictions
    if bDebugText:
        print('Starting flex QTabsSet predictions plot ...')

    fig, axs = mpcplot.make_fig(style=mpcplot.Style(use_tex=False), rows=20)
    (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12, ax13, ax14, ax15, ax16, ax17, ax18, ax19, ax20) = axs
    fig.set_figwidth(dFigureWidth)
    fig.set_figheight(100)

    simData = results["SimAgent"]["SimTestHall"]["Q_Tabs_set"]

    iIdxOffset = 0
    for iIdx, ax in enumerate(axs):
        ax.set_ylabel("$Q_{tabs,set}$ in W")
        ax.plot(simData.index, simData.values, 'g-', label='sim')

        baseFlexData = results["myMPCAgent"]["Baseline"]['variable']['Q_Tabs_set'][initial_time + (3600 * (iIdx + iIdxOffset))]  # .head(31)
        ax.plot(baseFlexData.index + initial_time + (3600 * (iIdx + iIdxOffset)), baseFlexData.ffill().values, 'k--', label='base')
        if not results["myMPCAgent"]["Baseline"]['parameter']['in_provision'][initial_time + (3600 * (iIdx + iIdxOffset))][0]:
            negFlexData = results["NegFlexMPC"]["NegFlexMPC"]['variable']['Q_Tabs_set'][initial_time + (3600 * (iIdx + iIdxOffset))]  # .head(31)
            ax.plot(negFlexData.index + initial_time + (3600 * (iIdx + iIdxOffset)), negFlexData.ffill().values, 'r--', label='neg')
            posFlexData = results["PosFlexMPC"]["PosFlexMPC"]['variable']['Q_Tabs_set'][initial_time + (3600 * (iIdx + iIdxOffset))]  # .head(31)
            ax.plot(posFlexData.index + initial_time + (3600 * (iIdx + iIdxOffset)), posFlexData.ffill().values, 'b--', label='pos')

        ax.legend()

        if ax == axs[-1]:
            x_ticks = np.arange(initial_time, until, 3600)  # maybe also add 1 to until
            x_tick_labels = [int(tick / 3600) for tick in x_ticks]
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_tick_labels)
            ax.set_xlabel("Time in hours")

    for ax in axs:
        mpcplot.make_grid(ax)
        ax.set_xlim(initial_time, until)

    # save the figure
    plt.savefig(f"plots_REGEN/plots_{offer_type}/FLEX_QTabsSet_predictions.svg", format='svg')
    plt.close()

    if bDebugText:
        print('Finished flex QTabsSet predictions plot')

    # Flex Temp Tamb Predictions
    if bDebugText:
        print('Starting flex temp Tamb predictions plot ...')

    fig, axs = mpcplot.make_fig(style=mpcplot.Style(use_tex=False), rows=20)
    (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12, ax13, ax14, ax15, ax16, ax17, ax18, ax19, ax20) = axs
    fig.set_figwidth(dFigureWidth)
    fig.set_figheight(100)

    simData = results["SimAgent"]["SimTestHall"]["T_Air"]
    simData_T_upper = results["SimAgent"]["SimTestHall"]["T_upper"]
    simData_T_lower = results["SimAgent"]["SimTestHall"]["T_lower"]
    simData_T_out = results["SimAgent"]["SimTestHall"]["T_out"]
    simData_T_amb = results["SimAgent"]["SimTestHall"]["T_amb"]
    simData_Q_Sol = results["SimAgent"]["SimTestHall"]["Q_RadSol"]

    iIdxOffset = 0
    for iIdx, ax in enumerate(axs):
        simData_T_upper.plot(ax=ax, color="0.5")
        simData_T_lower.plot(ax=ax, color="0.5")
        simData_T_out.plot(ax=ax, color=mpcplot.EBCColors.dark_grey)

        axTamb = ax.twinx()
        simData_T_amb.plot(ax=axTamb, color=mpcplot.EBCColors.blue)
        axQsol = ax.twinx()
        axQsol.spines.right.set_position(("axes", 0.5))
        simData_Q_Sol.plot(ax=axQsol, color=mpcplot.EBCColors.light_grey)

        ax.set_ylabel("$T_{room}$ in K")
        ax.plot(simData.index, simData.values, 'g-', label='sim')

        baseFlexData = results["myMPCAgent"]["Baseline"]['variable']['T_Air'][initial_time + (3600 * (iIdx + iIdxOffset))]  # .head(31)
        ax.plot(baseFlexData.index + initial_time + (3600 * (iIdx + iIdxOffset)), baseFlexData.ffill().values, 'k--', label='base')
        if not results["myMPCAgent"]["Baseline"]['parameter']['in_provision'][initial_time + (3600 * (iIdx + iIdxOffset))][0]:
            negFlexData = results["NegFlexMPC"]["NegFlexMPC"]['variable']['T_Air'][initial_time + (3600 * (iIdx + iIdxOffset))]  # .head(31)
            ax.plot(negFlexData.index + initial_time + (3600 * (iIdx + iIdxOffset)), negFlexData.ffill().values, 'r--', label='neg')
            posFlexData = results["PosFlexMPC"]["PosFlexMPC"]['variable']['T_Air'][initial_time + (3600 * (iIdx + iIdxOffset))]  # .head(31)
            ax.plot(posFlexData.index + initial_time + (3600 * (iIdx + iIdxOffset)), posFlexData.ffill().values, 'b--', label='pos')

        ax.legend()

        if ax == axs[-1]:
            x_ticks = np.arange(initial_time, until, 3600)  # maybe also add 1 to until
            x_tick_labels = [int(tick / 3600) for tick in x_ticks]
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_tick_labels)
            ax.set_xlabel("Time in hours")

    for ax in axs:
        mpcplot.make_grid(ax)
        ax.set_xlim(initial_time, until)

    # save the figure
    plt.savefig(f"plots_REGEN/plots_{offer_type}/FLEX_Temp_Tamb_predictions.svg", format='svg')
    plt.close()

    if bDebugText:
        print('Finished flex temp Tamb predictions plot')

    # Flex Pel Tamb Predictions
    if bDebugText:
        print('Starting flex Pel Tamb predictions plot ...')

    fig, axs = mpcplot.make_fig(style=mpcplot.Style(use_tex=False), rows=20)
    (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12, ax13, ax14, ax15, ax16, ax17, ax18, ax19,
     ax20) = axs
    fig.set_figwidth(dFigureWidth)
    fig.set_figheight(100)

    simData = results["SimAgent"]["SimTestHall"]["P_el_c"]
    simData_T_amb = results["SimAgent"]["SimTestHall"]["T_amb"]

    iIdxOffset = 0
    for iIdx, ax in enumerate(axs):
        ax.set_ylabel("$P_{el,c}$ in W")
        ax.plot(simData.index, simData.values, 'g-', label='sim')

        axSec = ax.twinx()
        simData_T_amb.plot(ax=axSec, color=mpcplot.EBCColors.blue)

        baseFlexData = results["myMPCAgent"]["Baseline"]['variable']['P_el_c'][initial_time + (3600 * (iIdx + iIdxOffset))]  # .head(31)
        ax.plot(baseFlexData.index + initial_time + (3600 * (iIdx + iIdxOffset)), baseFlexData.ffill().values, 'k--', label='base')
        if not results["myMPCAgent"]["Baseline"]['parameter']['in_provision'][initial_time + (3600 * (iIdx + iIdxOffset))][0]:
            negFlexData = results["NegFlexMPC"]["NegFlexMPC"]['variable']['P_el_c'][initial_time + (3600 * (iIdx + iIdxOffset))]  # .head(31)
            ax.plot(negFlexData.index + initial_time + (3600 * (iIdx + iIdxOffset)), negFlexData.ffill().values, 'r--', label='neg')
            posFlexData = results["PosFlexMPC"]["PosFlexMPC"]['variable']['P_el_c'][initial_time + (3600 * (iIdx + iIdxOffset))]  # .head(31)
            ax.plot(posFlexData.index + initial_time + (3600 * (iIdx + iIdxOffset)), posFlexData.ffill().values, 'b--', label='pos')

        ax.legend()

        if ax == axs[-1]:
            x_ticks = np.arange(initial_time, until, 3600)  # maybe also add 1 to until
            x_tick_labels = [int(tick / 3600) for tick in x_ticks]
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_tick_labels)
            ax.set_xlabel("Time in hours")

    for ax in axs:
        mpcplot.make_grid(ax)
        ax.set_xlim(initial_time, until)

    # save the figure
    plt.savefig(f"plots_REGEN/plots_{offer_type}/FLEX_Pel_Tamb_predictions.svg", format='svg')
    plt.close()

    if bDebugText:
        print('Finished flex Pel Tamb predictions plot')

    # Flex TahuSet Tamb Predictions
    if bDebugText:
        print('Starting flex TahuSet Tamb predictions plot ...')

    fig, axs = mpcplot.make_fig(style=mpcplot.Style(use_tex=False), rows=20)
    (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12, ax13, ax14, ax15, ax16, ax17, ax18, ax19,
     ax20) = axs
    fig.set_figwidth(dFigureWidth)
    fig.set_figheight(100)

    simData = results["SimAgent"]["SimTestHall"]["T_ahu_set"]
    simData_T_amb = results["SimAgent"]["SimTestHall"]["T_amb"]

    iIdxOffset = 0
    for iIdx, ax in enumerate(axs):
        ax.set_ylabel("$T_{ahu,set}$ in K")
        ax.plot(simData.index, simData.values, 'g-', label='sim')

        axSec = ax.twinx()
        simData_T_amb.plot(ax=axSec, color=mpcplot.EBCColors.blue)

        baseFlexData = results["myMPCAgent"]["Baseline"]['variable']['T_ahu_set'][initial_time + (3600 * (iIdx + iIdxOffset))]  # .head(31)
        ax.plot(baseFlexData.index + initial_time + (3600 * (iIdx + iIdxOffset)), baseFlexData.ffill().values, 'k--', label='base')
        if not results["myMPCAgent"]["Baseline"]['parameter']['in_provision'][initial_time + (3600 * (iIdx + iIdxOffset))][0]:
            negFlexData = results["NegFlexMPC"]["NegFlexMPC"]['variable']['T_ahu_set'][initial_time + (3600 * (iIdx + iIdxOffset))]  # .head(31)
            ax.plot(negFlexData.index + initial_time + (3600 * (iIdx + iIdxOffset)), negFlexData.ffill().values, 'r--', label='neg')
            posFlexData = results["PosFlexMPC"]["PosFlexMPC"]['variable']['T_ahu_set'][initial_time + (3600 * (iIdx + iIdxOffset))]  # .head(31)
            ax.plot(posFlexData.index + initial_time + (3600 * (iIdx + iIdxOffset)), posFlexData.ffill().values, 'b--', label='pos')

        ax.legend()

        if ax == axs[-1]:
            x_ticks = np.arange(initial_time, until, 3600)  # maybe also add 1 to until
            x_tick_labels = [int(tick / 3600) for tick in x_ticks]
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_tick_labels)
            ax.set_xlabel("Time in hours")

    for ax in axs:
        mpcplot.make_grid(ax)
        ax.set_xlim(initial_time, until)

    # save the figure
    plt.savefig(f"plots_REGEN/plots_{offer_type}/FLEX_TahuSet_Tamb_predictions.svg", format='svg')
    plt.close()

    if bDebugText:
        print('Finished flex TahuSet Tamb predictions plot')

    # Flex QTabsSet Tamb Predictions
    if bDebugText:
        print('Starting flex QTabsSet Tamb predictions plot ...')

    fig, axs = mpcplot.make_fig(style=mpcplot.Style(use_tex=False), rows=20)
    (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12, ax13, ax14, ax15, ax16, ax17, ax18, ax19,
     ax20) = axs
    fig.set_figwidth(dFigureWidth)
    fig.set_figheight(100)

    simData = results["SimAgent"]["SimTestHall"]["Q_Tabs_set"]
    simData_T_amb = results["SimAgent"]["SimTestHall"]["T_amb"]

    iIdxOffset = 0
    for iIdx, ax in enumerate(axs):
        ax.set_ylabel("$Q_{tabs,set}$ in W")
        ax.plot(simData.index, simData.values, 'g-', label='sim')

        axSec = ax.twinx()
        simData_T_amb.plot(ax=axSec, color=mpcplot.EBCColors.blue)

        baseFlexData = results["myMPCAgent"]["Baseline"]['variable']['Q_Tabs_set'][initial_time + (3600 * (iIdx + iIdxOffset))]  # .head(31)
        ax.plot(baseFlexData.index + initial_time + (3600 * (iIdx + iIdxOffset)), baseFlexData.ffill().values, 'k--', label='base')
        if not results["myMPCAgent"]["Baseline"]['parameter']['in_provision'][initial_time + (3600 * (iIdx + iIdxOffset))][0]:
            negFlexData = results["NegFlexMPC"]["NegFlexMPC"]['variable']['Q_Tabs_set'][initial_time + (3600 * (iIdx + iIdxOffset))]  # .head(31)
            ax.plot(negFlexData.index + initial_time + (3600 * (iIdx + iIdxOffset)), negFlexData.ffill().values, 'r--', label='neg')
            posFlexData = results["PosFlexMPC"]["PosFlexMPC"]['variable']['Q_Tabs_set'][initial_time + (3600 * (iIdx + iIdxOffset))]  # .head(31)
            ax.plot(posFlexData.index + initial_time + (3600 * (iIdx + iIdxOffset)), posFlexData.ffill().values, 'b--', label='pos')

        ax.legend()

        if ax == axs[-1]:
            x_ticks = np.arange(initial_time, until, 3600)  # maybe also add 1 to until
            x_tick_labels = [int(tick / 3600) for tick in x_ticks]
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_tick_labels)
            ax.set_xlabel("Time in hours")

    for ax in axs:
        mpcplot.make_grid(ax)
        ax.set_xlim(initial_time, until)

    # save the figure
    plt.savefig(f"plots_REGEN/plots_{offer_type}/FLEX_QTabsSet_Tamb_predictions.svg", format='svg')
    plt.close()

    if bDebugText:
        print('Finished flex QTabsSet Tamb predictions plot')


def fmu_plots(results, initial_time, until, offer_type, offer_Time_steps) -> None:
    # room temp
    if bDebugText:
        print('Starting room temp plot ...')

    fig, axs = mpcplot.make_fig(style=mpcplot.Style(use_tex=False), rows=1)
    ax1 = axs[0]
    fig.set_figwidth(dFigureWidth)
    # T out
    ax1.set_ylabel("$T_{room}$ in K")
    t_upper_index = results["myMPCAgent"]["Baseline"]["parameter"]["T_upper"].index.levels[0]
    t_upper_data = [results["myMPCAgent"]["Baseline"]["parameter"]["T_upper"][t_upper_index[idx]][0] for idx in range(len(t_upper_index))]
    ax1.plot(t_upper_data, t_upper_index, 'g--', label='T_upper')

    t_lower_index = results["myMPCAgent"]["Baseline"]["parameter"]["T_lower"].index.levels[0]
    t_lower_data = [results["myMPCAgent"]["Baseline"]["parameter"]["T_lower"][t_lower_index[idx]][0] for idx in range(len(t_lower_index))]
    ax1.plot(t_lower_data, t_lower_index, 'g--', label='T_lower')

    results["SimAgent"]["SimTestHall"]["TAirRoom"].plot(ax=ax1, color=mpcplot.EBCColors.dark_grey)

    simData = results["SimAgent"]["SimTestHall"]["TAirRoom"]
    ax1.plot(simData.index, simData.values, 'g-', label='sim')
    for iIdx in range(len(offer_Time_steps)):
        negFlexData = results["NegFlexMPC"]["NegFlexMPC"]['variable']['T_Air'][offer_Time_steps[iIdx]].head(31)
        ax1.plot(negFlexData.index + offer_Time_steps[iIdx], negFlexData.ffill().values, 'r--', label='neg')
        posFlexData = results["PosFlexMPC"]["PosFlexMPC"]['variable']['T_Air'][offer_Time_steps[iIdx]].head(31)
        ax1.plot(posFlexData.index + offer_Time_steps[iIdx], posFlexData.ffill().values, 'b--', label='pos')
        # baseFlexData = results["myMPCAgent"]["Baseline"]['variable']['T_Air'][offer_Time_steps[iIdx] + 1800]#.head(31)
        # ax1.plot(baseFlexData.index + (offer_Time_steps[iIdx] + 1800), baseFlexData.ffill().values, 'k--', label='base')

        if iIdx == 0:
            ax1.legend()

        ax1.vlines(offer_Time_steps[iIdx], ymin=290, ymax=300, colors="black")
        ax1.vlines(offer_Time_steps[iIdx] + 1800, ymin=290, ymax=300, colors="black")
        ax1.vlines(offer_Time_steps[iIdx] + 1800 + 1800, ymin=290, ymax=300, colors="black")
        ax1.vlines(offer_Time_steps[iIdx] + 1800 + 1800 + 14400, ymin=290, ymax=300, colors="black")

    ax1.set_ylim(290, 299.30)
    x_ticks = np.arange(initial_time, until, 3600)  # maybe also add 1 to until
    x_tick_labels = [int(tick / 3600) for tick in x_ticks]
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(x_tick_labels)
    ax1.set_xlabel("Time in hours")
    for ax in axs:
        mpcplot.make_grid(ax)
        ax.set_xlim(initial_time, until)

    # save the figure
    plt.savefig(f"plots_REGEN/plots_{offer_type}/room_temp.svg", format='svg')
    plt.close()

    if bDebugText:
        print('Finished room temp plot')

    # predictions
    if bDebugText:
        print('Starting predictions plot ...')

    fig, axs = mpcplot.make_fig(style=mpcplot.Style(use_tex=False), rows=3)
    (ax1, ax2, ax3) = axs
    fig.set_figwidth(dFigureWidth)

    # Q_Ahu
    ax2.set_ylabel("Q_Ahu in W")

    simData = results["SimAgent"]["SimTestHall"]["Q_Ahu"]
    ax2.plot(simData.index, simData.values, 'g-', label='sim')
    for iIdx in range(len(offer_Time_steps)):
        negFlexData = results["NegFlexMPC"]["NegFlexMPC"]['variable']['Q_Ahu'][offer_Time_steps[iIdx]].head(31)
        ax2.plot(negFlexData.index + offer_Time_steps[iIdx], negFlexData.ffill().values, 'r--', label='neg')
        posFlexData = results["PosFlexMPC"]["PosFlexMPC"]['variable']['Q_Ahu'][offer_Time_steps[iIdx]].head(31)
        ax2.plot(posFlexData.index + offer_Time_steps[iIdx], posFlexData.ffill().values, 'b--', label='pos')
        # baseFlexData = results["myMPCAgent"]["Baseline"]['variable']['Q_Ahu'][offer_Time_steps[iIdx] + 1800].head(31)
        # ax2.plot(posFlexData.index + (offer_Time_steps[iIdx] + 1800), baseFlexData.ffill().values, 'k--', label='base')

        if iIdx == 0:
            ax2.legend()

        ax2.vlines(offer_Time_steps[iIdx], ymin=-2000, ymax=2000, colors="black")
        ax2.vlines(offer_Time_steps[iIdx] + 1800, ymin=-2000, ymax=2000, colors="black")
        ax2.vlines(offer_Time_steps[iIdx] + 1800 + 1800, ymin=-2000, ymax=2000, colors="black")
        ax2.vlines(offer_Time_steps[iIdx] + 1800 + 1800 + 14400, ymin=-2000, ymax=2000, colors="black")

    # ax2.set_ylim(0, 3000)

    # Q_Tabs (BKT)
    ax3.set_ylabel("Q_Tabs in W")

    simData = results["SimAgent"]["SimTestHall"]["Q_Tabs"]
    ax3.plot(simData.index, simData.values, 'g-', label='sim')
    for iIdx in range(len(offer_Time_steps)):
        negFlexData = results["NegFlexMPC"]["NegFlexMPC"]['variable']['Q_Tabs_set'][offer_Time_steps[iIdx]].head(31)
        ax3.plot(negFlexData.index + offer_Time_steps[iIdx], negFlexData.ffill().values, 'r--', label='neg')
        posFlexData = results["PosFlexMPC"]["PosFlexMPC"]['variable']['Q_Tabs_set'][offer_Time_steps[iIdx]].head(31)
        ax3.plot(posFlexData.index + offer_Time_steps[iIdx], posFlexData.ffill().values, 'b--', label='pos')
        # baseFlexData = results["myMPCAgent"]["Baseline"]['variable']['Q_Tabs_set'][offer_Time_steps[iIdx] + 1800].head(31)
        # ax3.plot(posFlexData.index + (offer_Time_steps[iIdx] + 1800), baseFlexData.ffill().values, 'k--', label='base')

        if iIdx == 0:
            ax3.legend()

        ax3.vlines(offer_Time_steps[iIdx], ymin=0, ymax=5000, colors="black")
        ax3.vlines(offer_Time_steps[iIdx] + 1800, ymin=0, ymax=5000, colors="black")
        ax3.vlines(offer_Time_steps[iIdx] + 1800 + 1800, ymin=0, ymax=5000, colors="black")
        ax3.vlines(offer_Time_steps[iIdx] + 1800 + 1800 + 14400, ymin=0, ymax=5000, colors="black")

    # ax3.set_ylim(-100, 5000)

    x_ticks = np.arange(initial_time, until, 3600)  # maybe also add 1 to until
    x_tick_labels = [int(tick / 3600) for tick in x_ticks]
    ax3.set_xticks(x_ticks)
    ax3.set_xticklabels(x_tick_labels)
    ax3.set_xlabel("Time in hours")
    for ax in axs:
        mpcplot.make_grid(ax)
        ax.set_xlim(initial_time, until)

    # save the figure
    plt.savefig(f"plots_REGEN/plots_{offer_type}/predictions.svg", format='svg')
    plt.close()

    if bDebugText:
        print('Finished predictions plot')


if __name__ == "__main__":
    OFFER_TYPES = ["neg"]  # neg, pos, real

    for OFFER_TYPE in OFFER_TYPES:
        PICKLE_FILE = f'D:\\fse-jkl\\GIT_FILES\\flexquant_update\\Examples\\SimpleTesthall\\results\\results_file_{OFFER_TYPE}.pkl'
        create_plots(offer_type=OFFER_TYPE, pickle_file=PICKLE_FILE)
    