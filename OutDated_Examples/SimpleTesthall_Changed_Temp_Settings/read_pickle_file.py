import pickle
from agentlib_mpc.utils.analysis import mpc_at_time_step
import agentlib_mpc.utils.plotting.basic as mpcplot
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

if __name__ == "__main__":

    PICKLE_FILE = 'D:\\fse-jkl\\GIT_FILES\\flexquant\\Examples\\SimpleTesthall\\results\\results_file_neg.pkl'

    with open(PICKLE_FILE, 'rb') as file:
        results = pickle.load(file)
        initial_time = 345600.0
        until = initial_time + 86400.0

        # room temp
        fig, axs = mpcplot.make_fig(style=mpcplot.Style(use_tex=False), rows=1)
        ax1 = axs[0]
        fig.set_figwidth(13)
        # T out
        ax1.set_ylabel("$T_{room}$ in K")
        results["SimAgent"]["SimTestHall"]["TAirRoom"].plot(ax=ax1, color=mpcplot.EBCColors.dark_grey)

        mpc_at_time_step(
            data=results["NegFlexMPC"]["NegFlexMPC"], time_step=initial_time + 9000, variable="T_Air"
        ).plot(ax=ax1, label="neg", linestyle="--", color=mpcplot.EBCColors.red)
        mpc_at_time_step(
            data=results["PosFlexMPC"]["PosFlexMPC"], time_step=initial_time + 9000, variable="T_Air"
        ).plot(ax=ax1, label="pos", linestyle="--", color=mpcplot.EBCColors.blue)
        mpc_at_time_step(
            data=results["myMPCAgent"]["myMPC"], time_step=initial_time + 9900, variable="T_Air"
        ).plot(ax=ax1, label="base", linestyle="--", color=mpcplot.EBCColors.dark_grey)

        ax1.legend()
        ax1.vlines(initial_time + 9000, ymin=0, ymax=500, colors="black")
        ax1.vlines(initial_time + 9900, ymin=0, ymax=500, colors="black")
        ax1.vlines(initial_time + 10800, ymin=0, ymax=500, colors="black")
        ax1.vlines(initial_time + 18000, ymin=0, ymax=500, colors="black")

        ax1.set_ylim(284, 301)
        x_ticks = np.arange(initial_time, until, 3600)  # maybe also add 1 to until
        x_tick_labels = [int(tick / 3600) for tick in x_ticks]
        ax1.set_xticks(x_ticks)
        ax1.set_xticklabels(x_tick_labels)
        ax1.set_xlabel("Time in hours")
        for ax in axs:
            mpcplot.make_grid(ax)
            ax.set_xlim(initial_time, until)

        plt.show()

        # predictions
        fig, axs = mpcplot.make_fig(style=mpcplot.Style(use_tex=False), rows=2)
        (ax1, ax2) = axs
        fig.set_figwidth(13)
        # P_el
        ax1.set_ylabel("$P_{el}$ in W")

        # simData = results["SimAgent"]["SimTestHall"]["P_el_c"]
        # ax1.plot(simData.index, simData.values, 'g-', label='sim')
        negFlexData = results["NegFlexMPC"]["NegFlexMPC"]['variable']['P_el_c'][initial_time + 9000.0]
        ax1.plot(negFlexData.index + (initial_time + 9000.0), negFlexData.ffill().values, 'r--', label='neg')
        posFlexData = results["PosFlexMPC"]["PosFlexMPC"]['variable']['P_el_c'][initial_time + 9000.0]
        ax1.plot(posFlexData.index + (initial_time + 9000.0), posFlexData.ffill().values, 'b--', label='pos')
        baseFlexData = results["myMPCAgent"]["myMPC"]['variable']['P_el_c'][initial_time + 10800.0]
        ax1.plot(posFlexData.index + (initial_time + 10800.0), posFlexData.ffill().values, 'k--', label='base')

        ax1.legend()
        ax1.vlines(initial_time + 9000, ymin=-1000, ymax=5000, colors="black")
        ax1.vlines(initial_time + 10800, ymin=-1000, ymax=5000, colors="black")
        ax1.vlines(initial_time + 12600, ymin=-1000, ymax=5000, colors="black")
        ax1.vlines(initial_time + 27000, ymin=-1000, ymax=5000, colors="black")
        ax1.set_ylim(0, 2500)

        # Q_Ahu
        ax2.set_ylabel("Q_Ahu in W")
        results["SimAgent"]["SimTestHall"]["Q_Ahu"].plot(ax=ax2, color=mpcplot.EBCColors.dark_grey)
        mpc_at_time_step(
            data=results["NegFlexMPC"]["NegFlexMPC"], time_step=initial_time + 9000, variable="Q_Ahu"
        ).ffill().plot(
            ax=ax2,
            drawstyle="steps-post",
            label="neg",
            linestyle="--",
            color=mpcplot.EBCColors.red,
        )
        mpc_at_time_step(
            data=results["PosFlexMPC"]["PosFlexMPC"], time_step=initial_time + 9000, variable="Q_Ahu"
        ).ffill().plot(
            ax=ax2,
            drawstyle="steps-post",
            label="pos",
            linestyle="--",
            color=mpcplot.EBCColors.blue,
        )
        mpc_at_time_step(
            data=results["myMPCAgent"]["myMPC"], time_step=initial_time + 10800, variable="Q_Ahu"
        ).ffill().plot(
            ax=ax2,
            drawstyle="steps-post",
            label="base",
            linestyle="--",
            color=mpcplot.EBCColors.dark_grey,
        )

        ax2.legend()
        ax2.vlines(initial_time + 9000, ymin=0, ymax=500, colors="black")
        ax2.vlines(initial_time + 10800, ymin=0, ymax=500, colors="black")
        ax2.vlines(initial_time + 12600, ymin=0, ymax=500, colors="black")
        ax2.vlines(initial_time + 27000, ymin=0, ymax=500, colors="black")

        ax2.set_ylim(500, 3000)

        x_ticks = np.arange(initial_time, until, 3600)  # maybe also add 1 to until
        x_tick_labels = [int(tick / 3600) for tick in x_ticks]
        ax2.set_xticks(x_ticks)
        ax2.set_xticklabels(x_tick_labels)
        ax2.set_xlabel("Time in hours")
        for ax in axs:
            mpcplot.make_grid(ax)
            ax.set_xlim(initial_time, until)

        plt.show()

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

        plt.show()
