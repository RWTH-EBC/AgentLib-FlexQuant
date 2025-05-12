import pickle
from pathlib import Path
import agentlib_mpc.utils.plotting.basic as mpcplot
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

if __name__ == "__main__":
    OFFER_TYPE = "neg"  # neg, pos, real
    PICKLE_FILE = f'D:\\fse-jkl\\GIT_FILES\\flexquant\\Examples\\SimpleTesthall\\results\\results_file_{OFFER_TYPE}.pkl'

    with open(PICKLE_FILE, 'rb') as file:
        results = pickle.load(file)
        initial_time = 345600.0
        until = initial_time + 86400.0

        offer_Time_steps: list = []
        Plot_files = list(Path(f"plots/plots_{OFFER_TYPE}").glob('flex_offer_*_flexEnvelope.svg'))
        for iIdx in range(len(Plot_files)):
            strDigits = ''.join(char for char in Plot_files[iIdx].name if char.isdigit())
            offer_Time_steps.append(int(strDigits) / 10)
        offer_Time_steps.sort()

        # create the folder to store the figure
        Path("plots_REGEN").mkdir(parents=True, exist_ok=True)
        Path(f"plots_REGEN/plots_{OFFER_TYPE}").mkdir(parents=True, exist_ok=True)

        # room temp
        fig, axs = mpcplot.make_fig(style=mpcplot.Style(use_tex=False), rows=1)
        ax1 = axs[0]
        fig.set_figwidth(13)
        # T out
        ax1.set_ylabel("$T_{room}$ in K")
        results["SimAgent"]["SimTestHall"]["T_upper"].plot(ax=ax1, color="0.5")
        results["SimAgent"]["SimTestHall"]["T_lower"].plot(ax=ax1, color="0.5")
        results["SimAgent"]["SimTestHall"]["T_out"].plot(ax=ax1, color=mpcplot.EBCColors.dark_grey)

        baseFlexData = results["myMPCAgent"]["myMPC"]['variable']['T_Air'][initial_time + (3600 * 2)].head(31)
        ax1.plot(baseFlexData.index + initial_time + (3600 * 2), baseFlexData.ffill().values, 'k--',
                 label='base')

        simData = results["SimAgent"]["SimTestHall"]["T_Air"]
        ax1.plot(simData.index, simData.values, 'g-', label='sim')
        for iIdx in range(len(offer_Time_steps)):
            negFlexData = results["NegFlexMPC"]["NegFlexMPC"]['variable']['T_Air'][offer_Time_steps[iIdx]].head(31)
            ax1.plot(negFlexData.index + offer_Time_steps[iIdx], negFlexData.ffill().values, 'r--', label='neg')
            posFlexData = results["PosFlexMPC"]["PosFlexMPC"]['variable']['T_Air'][offer_Time_steps[iIdx]].head(31)
            ax1.plot(posFlexData.index + offer_Time_steps[iIdx], posFlexData.ffill().values, 'b--', label='pos')
            baseFlexData = results["myMPCAgent"]["myMPC"]['variable']['T_Air'][offer_Time_steps[iIdx] + 1800].head(31)
            ax1.plot(baseFlexData.index + (offer_Time_steps[iIdx] + 1800), baseFlexData.ffill().values, 'k--',
                     label='base')

            if iIdx == 0:
                ax1.legend()

            ax1.vlines(offer_Time_steps[iIdx], ymin=0, ymax=500, colors="black")
            ax1.vlines(offer_Time_steps[iIdx] + 1800, ymin=0, ymax=500, colors="black")
            ax1.vlines(offer_Time_steps[iIdx] + 1800 + 1800, ymin=0, ymax=500, colors="black")
            ax1.vlines(offer_Time_steps[iIdx] + 1800 + 1800 + 14400, ymin=0, ymax=500, colors="black")

        ax1.set_ylim(290, 297)
        x_ticks = np.arange(initial_time, until, 3600)  # maybe also add 1 to until
        x_tick_labels = [int(tick / 3600) for tick in x_ticks]
        ax1.set_xticks(x_ticks)
        ax1.set_xticklabels(x_tick_labels)
        ax1.set_xlabel("Time in hours")
        for ax in axs:
            mpcplot.make_grid(ax)
            ax.set_xlim(initial_time, until)

        # save the figure
        plt.savefig(f"plots_REGEN/plots_{OFFER_TYPE}/room_temp.svg", format='svg')
        plt.close()

        # predictions
        fig, axs = mpcplot.make_fig(style=mpcplot.Style(use_tex=False), rows=2)
        (ax1, ax2) = axs
        fig.set_figwidth(13)
        # P_el
        ax1.set_ylabel("$P_{el}$ in W")

        simData = results["SimAgent"]["SimTestHall"]["P_el_c"]
        ax1.plot(simData.index, simData.values, 'g-', label='sim')
        for iIdx in range(len(offer_Time_steps)):
            negFlexData = results["NegFlexMPC"]["NegFlexMPC"]['variable']['P_el_c'][offer_Time_steps[iIdx]].head(31)
            ax1.plot(negFlexData.index + offer_Time_steps[iIdx], negFlexData.ffill().values, 'r--', label='neg')
            posFlexData = results["PosFlexMPC"]["PosFlexMPC"]['variable']['P_el_c'][offer_Time_steps[iIdx]].head(31)
            ax1.plot(posFlexData.index + offer_Time_steps[iIdx], posFlexData.ffill().values, 'b--', label='pos')
            baseFlexData = results["myMPCAgent"]["myMPC"]['variable']['P_el_c'][offer_Time_steps[iIdx] + 1800].head(31)
            ax1.plot(posFlexData.index + (offer_Time_steps[iIdx] + 1800), baseFlexData.ffill().values, 'k--',
                     label='base')

            if iIdx == 0:
                ax1.legend()

            ax1.vlines(offer_Time_steps[iIdx], ymin=0, ymax=3000, colors="black")
            ax1.vlines(offer_Time_steps[iIdx] + 1800, ymin=0, ymax=3000, colors="black")
            ax1.vlines(offer_Time_steps[iIdx] + 1800 + 1800, ymin=0, ymax=3000, colors="black")
            ax1.vlines(offer_Time_steps[iIdx] + 1800 + 1800 + 14400, ymin=0, ymax=3000, colors="black")

        ax1.set_ylim(0, 2500)

        # Q_Ahu
        ax2.set_ylabel("Q_Ahu in W")

        simData = results["SimAgent"]["SimTestHall"]["Q_Ahu"]
        ax2.plot(simData.index, simData.values, 'g-', label='sim')
        for iIdx in range(len(offer_Time_steps)):
            negFlexData = results["NegFlexMPC"]["NegFlexMPC"]['variable']['Q_Ahu'][offer_Time_steps[iIdx]].head(31)
            ax2.plot(negFlexData.index + offer_Time_steps[iIdx], negFlexData.ffill().values, 'r--', label='neg')
            posFlexData = results["PosFlexMPC"]["PosFlexMPC"]['variable']['Q_Ahu'][offer_Time_steps[iIdx]].head(31)
            ax2.plot(posFlexData.index + offer_Time_steps[iIdx], posFlexData.ffill().values, 'b--', label='pos')
            baseFlexData = results["myMPCAgent"]["myMPC"]['variable']['Q_Ahu'][offer_Time_steps[iIdx] + 1800].head(31)
            ax2.plot(posFlexData.index + (offer_Time_steps[iIdx] + 1800), baseFlexData.ffill().values, 'k--',
                     label='base')

            if iIdx == 0:
                ax2.legend()

            ax2.vlines(offer_Time_steps[iIdx], ymin=0, ymax=3000, colors="black")
            ax2.vlines(offer_Time_steps[iIdx] + 1800, ymin=0, ymax=3000, colors="black")
            ax2.vlines(offer_Time_steps[iIdx] + 1800 + 1800, ymin=0, ymax=3000, colors="black")
            ax2.vlines(offer_Time_steps[iIdx] + 1800 + 1800 + 14400, ymin=0, ymax=3000, colors="black")

        ax2.set_ylim(500, 3000)

        x_ticks = np.arange(initial_time, until, 3600)  # maybe also add 1 to until
        x_tick_labels = [int(tick / 3600) for tick in x_ticks]
        ax2.set_xticks(x_ticks)
        ax2.set_xticklabels(x_tick_labels)
        ax2.set_xlabel("Time in hours")
        for ax in axs:
            mpcplot.make_grid(ax)
            ax.set_xlim(initial_time, until)

        # save the figure
        plt.savefig(f"plots_REGEN/plots_{OFFER_TYPE}/predictions.svg", format='svg')
        plt.close()

        # flexibility
        # get only the first prediction time of each time step
        ind_res = results["FlexibilityIndicator"]["FlexibilityIndicator"]
        energy_flex_neg = ind_res.xs("energyflex_neg", axis=1).droplevel(1).dropna()
        energy_flex_pos = ind_res.xs("energyflex_pos", axis=1).droplevel(1).dropna()
        fig, axs = mpcplot.make_fig(style=mpcplot.Style(use_tex=False), rows=1)
        ax1 = axs[0]
        fig.set_figwidth(13)
        ax1.set_ylabel("$\epsilon$ in kWh")
        energy_flex_neg.plot(ax=ax1, label="neg")
        energy_flex_pos.plot(ax=ax1, label="pos")
        energy_flex_neg.plot(ax=ax1, label="neg", color=mpcplot.EBCColors.red)
        energy_flex_pos.plot(ax=ax1, label="pos", color=mpcplot.EBCColors.blue)
        ax1.yaxis.set_major_formatter(FormatStrFormatter("%.4f"))

        ax1.legend()

        x_ticks = np.arange(initial_time, until, 3600)  # maybe also add 1 to until
        x_tick_labels = [int(tick / 3600) for tick in x_ticks]
        ax1.set_xticks(x_ticks)
        ax1.set_xticklabels(x_tick_labels)
        ax1.set_xlabel("Time in hours")
        for ax in axs:
            mpcplot.make_grid(ax)
            ax.set_xlim(initial_time, until)

        # save the figure
        plt.savefig(f"plots_REGEN/plots_{OFFER_TYPE}/flexibility.svg", format='svg')
        plt.close()
