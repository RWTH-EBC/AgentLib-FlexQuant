import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import agentlib_mpc.utils.plotting.basic as mpcplot
from agentlib_mpc.utils.analysis import mpc_at_time_step
from agentlib_flexquant.data_structures.flex_results import Results
matplotlib.use('TkAgg')


def plot_results(results_data: dict = None):
    """
    Example how plotting with matplotlib and mpcplot from agentlib_mpc works
    """
    if results_data is None:
        res = Results(
            flex_config="Building_1_flex/flex_config.json",
            simulator_agent_config="Building_1_flex/simulator.json",
            results="Building_1_flex"
        )
    else:
        res = Results(
            flex_config="Building_1_flex/flex_config.json",
            simulator_agent_config="Building_1_flex/simulator.json",
            results=results_data
        )

    fig, axs = mpcplot.make_fig(style=mpcplot.Style(use_tex=False), rows=2)
    (ax1, ax2) = axs
    # load
    ax1.set_ylabel(r"$T_{Air}$ in K")
    res.df_simulation["T_Air"].dropna().plot(ax=ax1)
    # T_in
    ax2.set_ylabel("$T_{amb}$ in K")
    res.df_simulation["T_amb"].dropna().plot(ax=ax2)
    mpc_at_time_step(
        data=res.df_baseline, time_step=9000, variable="T_amb",
    variable_type="parameter",
    ).ffill().plot(
        ax=ax2,
        drawstyle="steps-post",
        label="base",
        linestyle="--",
        color=mpcplot.EBCColors.dark_grey,
    )
    x_ticks = np.arange(0, 3600 * 6 + 1, 3600)
    x_tick_labels = [int(tick / 3600) for tick in x_ticks]
    ax2.set_xticks(x_ticks)
    ax2.set_xticklabels(x_tick_labels)
    ax2.set_xlabel("Time in hours")
    for ax in axs:
        mpcplot.make_grid(ax)
        ax.set_xlim(0, 3600 * 24)

    # room temp
    fig, axs = mpcplot.make_fig(style=mpcplot.Style(use_tex=False), rows=1)
    ax1 = axs[0]
    # T out
    ax1.set_ylabel("$T_{room}$ in K")
    # res.df_simulation["T_upper"].dropna().plot(ax=ax1, color="0.5")
    # res.df_simulation["T_lower"].dropna().plot(ax=ax1, color="0.5")
    res.df_simulation["T_Air"].dropna().plot(ax=ax1, color=mpcplot.EBCColors.dark_grey)
    mpc_at_time_step(
        data=res.df_neg_flex, time_step=9000, variable="T_Air"
    ).plot(ax=ax1, label="neg", linestyle="--", color=mpcplot.EBCColors.red)
    mpc_at_time_step(
        data=res.df_pos_flex, time_step=9000, variable="T_Air"
    ).plot(ax=ax1, label="pos", linestyle="--", color=mpcplot.EBCColors.blue)
    mpc_at_time_step(
        data=res.df_baseline, time_step=9900, variable="T_Air"
    ).plot(ax=ax1, label="base", linestyle="--", color=mpcplot.EBCColors.dark_grey)

    ax1.legend()
    ax1.vlines(9000, ymin=0, ymax=500, colors="black")
    ax1.vlines(9900, ymin=0, ymax=500, colors="black")
    ax1.vlines(10800, ymin=0, ymax=500, colors="black")
    ax1.vlines(18000, ymin=0, ymax=500, colors="black")

    ax1.set_ylim(292, 294)
    x_ticks = np.arange(0, 3600 * 6 + 1, 3600)
    x_tick_labels = [int(tick / 3600) for tick in x_ticks]
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(x_tick_labels)
    ax1.set_xlabel("Time in hours")
    for ax in axs:
        mpcplot.make_grid(ax)
        ax.set_xlim(0, 3600 * 24)

    # predictions
    fig, axs = mpcplot.make_fig(style=mpcplot.Style(use_tex=False), rows=2)
    (ax1, ax2) = axs
    # P_el
    ax1.set_ylabel("$P_{el}$ in kW")
    res.df_simulation["P_el"].dropna().plot(ax=ax1, color=mpcplot.EBCColors.dark_grey)
    mpc_at_time_step(
        data=res.df_neg_flex, time_step=9000, variable="P_el"
    ).ffill().plot(
        ax=ax1,
        drawstyle="steps-post",
        label="neg",
        linestyle="--",
        color=mpcplot.EBCColors.red,
    )
    mpc_at_time_step(
        data=res.df_pos_flex, time_step=9000, variable="P_el"
    ).ffill().plot(
        ax=ax1,
        drawstyle="steps-post",
        label="pos",
        linestyle="--",
        color=mpcplot.EBCColors.blue,
    )
    mpc_at_time_step(
        data=res.df_baseline, time_step=9000, variable="P_el"
    ).ffill().plot(
        ax=ax1,
        drawstyle="steps-post",
        label="base",
        linestyle="--",
        color=mpcplot.EBCColors.dark_grey,
    )
    ax1.legend()
    ax1.vlines(9000, ymin=-1000, ymax=5000, colors="black")
    ax1.vlines(9900, ymin=-1000, ymax=5000, colors="black")
    ax1.vlines(10800, ymin=-1000, ymax=5000, colors="black")
    ax1.vlines(18000, ymin=-1000, ymax=5000, colors="black")
    ax1.set_ylim(-0.1, 5)

    # mdot
    ax2.set_ylabel(r"$\dot{m}$ in kg/s")
    res.df_simulation["valve_opening"].dropna().plot(ax=ax2, color=mpcplot.EBCColors.dark_grey)
    # res.df_simulation["nSet_HP"].dropna().plot(ax=ax2, color=mpcplot.EBCColors.grey)
    mpc_at_time_step(
        data=res.df_neg_flex, time_step=9000, variable="valve_opening"
    ).ffill().plot(
        ax=ax2,
        drawstyle="steps-post",
        label="neg",
        linestyle="--",
        color=mpcplot.EBCColors.red,
    )
    mpc_at_time_step(
        data=res.df_pos_flex, time_step=9000, variable="valve_opening"
    ).ffill().plot(
        ax=ax2,
        drawstyle="steps-post",
        label="pos",
        linestyle="--",
        color=mpcplot.EBCColors.blue,
    )
    mpc_at_time_step(
        data=res.df_baseline, time_step=9900, variable="valve_opening"
    ).ffill().plot(
        ax=ax2,
        drawstyle="steps-post",
        label="base",
        linestyle="--",
        color=mpcplot.EBCColors.dark_grey,
    )
    ax2.legend()
    ax2.vlines(9000, ymin=0, ymax=500, colors="black")
    ax2.vlines(9900, ymin=0, ymax=500, colors="black")
    ax2.vlines(13500, ymin=0, ymax=500, colors="black")
    ax2.vlines(27900, ymin=0, ymax=500, colors="black")

    ax2.set_ylim(-0.1, 1.1)

    x_ticks = np.arange(0, 3600 * 6 + 1, 3600)
    x_tick_labels = [int(tick / 3600) for tick in x_ticks]
    ax2.set_xticks(x_ticks)
    ax2.set_xticklabels(x_tick_labels)
    ax2.set_xlabel("Time in hours")
    for ax in axs:
        mpcplot.make_grid(ax)
        ax.set_xlim(0, 3600 * 24)

    # flexibility
    # get only the first prediction time of each time step
    energy_flex_neg = res.df_indicator.xs("negative_energy_flex", axis=1).droplevel(1).dropna()
    energy_flex_pos = res.df_indicator.xs("positive_energy_flex", axis=1).droplevel(1).dropna()
    fig, axs = mpcplot.make_fig(style=mpcplot.Style(use_tex=False), rows=1)
    ax1 = axs[0]
    ax1.set_ylabel(r"$\epsilon$ in kWh")
    energy_flex_neg.plot(ax=ax1, label="neg")
    energy_flex_pos.plot(ax=ax1, label="pos")
    energy_flex_neg.plot(ax=ax1, label="neg", color=mpcplot.EBCColors.red)
    energy_flex_pos.plot(ax=ax1, label="pos", color=mpcplot.EBCColors.blue)

    ax1.legend()

    x_ticks = np.arange(0, 3600 * 6 + 1, 3600)
    x_tick_labels = [int(tick / 3600) for tick in x_ticks]
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(x_tick_labels)
    ax1.set_xlabel("Time in hours")
    for ax in axs:
        mpcplot.make_grid(ax)
        ax.set_xlim(0, 3600 * 24)

    plt.show()


if __name__ == "__main__":
    plot_results()
