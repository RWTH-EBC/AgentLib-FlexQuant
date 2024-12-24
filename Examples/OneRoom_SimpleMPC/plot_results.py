import numpy as np
import matplotlib.pyplot as plt
import agentlib_mpc.utils.plotting.basic as mpcplot
from agentlib_mpc.utils.analysis import mpc_at_time_step
from flexibility_quantification.data_structures.flex_results import Results
from pathlib import Path
from matplotlib.ticker import FormatStrFormatter
import pickle


def plot_results(results_data: dict = None):
    """
    Example how plotting with matplotlib and mpcplot from agentlib_mpc works
    """
    if results_data is None:
        res = Results(
            flex_config="flex_configs/flexibility_agent_config.json",
            simulator_agent_config="mpc_and_sim/simple_sim.json",
            results="results"
        )
    else:
        res = Results(
            flex_config="flex_configs/flexibility_agent_config.json",
            simulator_agent_config="mpc_and_sim/simple_sim.json",
            results=results_data
        )

    fig, axs = mpcplot.make_fig(style=mpcplot.Style(use_tex=False), rows=2)
    (ax1, ax2) = axs
    # load
    ax1.set_ylabel(r"$\dot{Q}_{Room}$ in W")
    res.df_simulation["load"].plot(ax=ax1)
    # T_in
    ax2.set_ylabel("$T_{in}$ in K")
    res.df_simulation["T_in"].plot(ax=ax2)
    x_ticks = np.arange(0, 3600 * 6 + 1, 3600)
    x_tick_labels = [int(tick / 3600) for tick in x_ticks]
    ax2.set_xticks(x_ticks)
    ax2.set_xticklabels(x_tick_labels)
    ax2.set_xlabel("Time in hours")
    for ax in axs:
        mpcplot.make_grid(ax)
        ax.set_xlim(0, 3600 * 6)

    # room temp
    fig, axs = mpcplot.make_fig(style=mpcplot.Style(use_tex=False), rows=1)
    ax1 = axs[0]
    # T out
    ax1.set_ylabel("$T_{room}$ in K")
    res.df_simulation["T_upper"].plot(ax=ax1, color="0.5")
    res.df_simulation["T_lower"].plot(ax=ax1, color="0.5")
    res.df_simulation["T_out"].plot(ax=ax1, color=mpcplot.EBCColors.dark_grey)
    mpc_at_time_step(
        data=res.df_neg_flex, time_step=9000, variable="T"
    ).plot(ax=ax1, label="neg", linestyle="--", color=mpcplot.EBCColors.red)
    mpc_at_time_step(
        data=res.df_pos_flex, time_step=9000, variable="T"
    ).plot(ax=ax1, label="pos", linestyle="--", color=mpcplot.EBCColors.blue)
    mpc_at_time_step(
        data=res.df_baseline, time_step=9900, variable="T"
    ).plot(ax=ax1, label="base", linestyle="--", color=mpcplot.EBCColors.dark_grey)

    ax1.legend()
    ax1.vlines(9000, ymin=0, ymax=500, colors="black")
    ax1.vlines(9900, ymin=0, ymax=500, colors="black")
    ax1.vlines(10800, ymin=0, ymax=500, colors="black")
    ax1.vlines(18000, ymin=0, ymax=500, colors="black")

    ax1.set_ylim(289, 299)
    x_ticks = np.arange(0, 3600 * 6 + 1, 3600)
    x_tick_labels = [int(tick / 3600) for tick in x_ticks]
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(x_tick_labels)
    ax1.set_xlabel("Time in hours")
    for ax in axs:
        mpcplot.make_grid(ax)
        ax.set_xlim(0, 3600 * 6)

    # predictions
    fig, axs = mpcplot.make_fig(style=mpcplot.Style(use_tex=False), rows=2)
    (ax1, ax2) = axs
    # P_el
    ax1.set_ylabel("$P_{el}$ in kW")
    res.df_simulation["P_el"].plot(ax=ax1, color=mpcplot.EBCColors.dark_grey)
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
    ax1.set_ylim(-0.1, 1)

    # mdot
    ax2.set_ylabel(r"$\dot{m}$ in kg/s")
    res.df_simulation["mDot"].plot(ax=ax2, color=mpcplot.EBCColors.dark_grey)
    mpc_at_time_step(
        data=res.df_neg_flex, time_step=9000, variable="mDot"
    ).ffill().plot(
        ax=ax2,
        drawstyle="steps-post",
        label="neg",
        linestyle="--",
        color=mpcplot.EBCColors.red,
    )
    mpc_at_time_step(
        data=res.df_pos_flex, time_step=9000, variable="mDot"
    ).ffill().plot(
        ax=ax2,
        drawstyle="steps-post",
        label="pos",
        linestyle="--",
        color=mpcplot.EBCColors.blue,
    )
    mpc_at_time_step(
        data=res.df_baseline, time_step=9900, variable="mDot"
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
    ax2.vlines(10800, ymin=0, ymax=500, colors="black")
    ax2.vlines(18000, ymin=0, ymax=500, colors="black")

    ax2.set_ylim(0, 0.06)

    x_ticks = np.arange(0, 3600 * 6 + 1, 3600)
    x_tick_labels = [int(tick / 3600) for tick in x_ticks]
    ax2.set_xticks(x_ticks)
    ax2.set_xticklabels(x_tick_labels)
    ax2.set_xlabel("Time in hours")
    for ax in axs:
        mpcplot.make_grid(ax)
        ax.set_xlim(0, 3600 * 6)

    # flexibility
    # get only the first prediction time of each time step
    energy_flex_neg = res.df_indicator.xs("energyflex_neg", axis=1).droplevel(1).dropna()
    energy_flex_pos = res.df_indicator.xs("energyflex_pos", axis=1).droplevel(1).dropna()
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
        ax.set_xlim(0, 3600 * 6)

    plt.show()


def plot_results_2(results: dict = None, offer_type: str = None, until: float = 0):
    """
    Example how plotting with matplotlib and mpcplot from agentlib_mpc works
    """
    # create the folder to store the figure
    Path("plots").mkdir(parents=True, exist_ok=True)
    Path(f"plots/plots_{offer_type}").mkdir(parents=True, exist_ok=True)

    # disturbances
    fig, axs = mpcplot.make_fig(style=mpcplot.Style(use_tex=False), rows=2)
    (ax1, ax2) = axs
    # load
    ax1.set_ylabel("$\dot{Q}_{Room}$ in W")
    results["SimAgent"]["room"]["load"].plot(ax=ax1)
    # T_in
    ax2.set_ylabel("$T_{in}$ in K")
    results["SimAgent"]["room"]["T_in"].plot(ax=ax2)
    x_ticks = np.arange(0, until + 1, 3600)
    x_tick_labels = [int(tick / 3600) for tick in x_ticks]
    ax2.set_xticks(x_ticks)
    ax2.set_xticklabels(x_tick_labels)
    ax2.set_xlabel("Time in hours")
    for ax in axs:
        mpcplot.make_grid(ax)
        ax.set_xlim(0, until)

    # save the figure
    plt.savefig(f"plots/plots_{offer_type}/disturbances.svg", format='svg')
    plt.close()

    # room temp
    fig, axs = mpcplot.make_fig(style=mpcplot.Style(use_tex=False), rows=1)
    ax1 = axs[0]
    # T out
    ax1.set_ylabel("$T_{room}$ in K")
    results["SimAgent"]["room"]["T_upper"].plot(ax=ax1, color="0.5")
    results["SimAgent"]["room"]["T_lower"].plot(ax=ax1, color="0.5")
    results["SimAgent"]["room"]["T_out"].plot(ax=ax1, color=mpcplot.EBCColors.dark_grey)
    mpc_at_time_step(
        data=results["NegFlexMPC"]["NegFlexMPC"], time_step=9000, variable="T"
    ).plot(ax=ax1, label="neg", linestyle="--", color=mpcplot.EBCColors.red)
    mpc_at_time_step(
        data=results["PosFlexMPC"]["PosFlexMPC"], time_step=9000, variable="T"
    ).plot(ax=ax1, label="pos", linestyle="--", color=mpcplot.EBCColors.blue)
    mpc_at_time_step(
        data=results["FlexModel"]["Baseline"], time_step=9900, variable="T"
    ).plot(ax=ax1, label="base", linestyle="--", color=mpcplot.EBCColors.dark_grey)

    ax1.legend()

    mpc_at_time_step(
        data=results["NegFlexMPC"]["NegFlexMPC"], time_step=18900, variable="T"
    ).plot(ax=ax1, label="neg", linestyle="--", color=mpcplot.EBCColors.red)
    mpc_at_time_step(
        data=results["PosFlexMPC"]["PosFlexMPC"], time_step=18900, variable="T"
    ).plot(ax=ax1, label="pos", linestyle="--", color=mpcplot.EBCColors.blue)
    mpc_at_time_step(
        data=results["FlexModel"]["Baseline"], time_step=19800, variable="T"
    ).plot(ax=ax1, label="base", linestyle="--", color=mpcplot.EBCColors.dark_grey)

    ax1.vlines(9000, ymin=0, ymax=500, colors="black")
    ax1.vlines(9900, ymin=0, ymax=500, colors="black")
    ax1.vlines(10800, ymin=0, ymax=500, colors="black")
    ax1.vlines(18000, ymin=0, ymax=500, colors="black")

    ax1.vlines(18900, ymin=0, ymax=500, colors="black")
    ax1.vlines(19800, ymin=0, ymax=500, colors="black")
    ax1.vlines(20700, ymin=0, ymax=500, colors="black")
    ax1.vlines(27900, ymin=0, ymax=500, colors="black")

    ax1.set_ylim(284, 301)
    x_ticks = np.arange(0, until + 1, 3600)
    x_tick_labels = [int(tick / 3600) for tick in x_ticks]
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(x_tick_labels)
    ax1.set_xlabel("Time in hours")
    for ax in axs:
        mpcplot.make_grid(ax)
        ax.set_xlim(0, until)

    # save the figure
    plt.savefig(f"plots/plots_{offer_type}/room_temp.svg", format='svg')
    plt.close()

    # predictions
    fig, axs = mpcplot.make_fig(style=mpcplot.Style(use_tex=False), rows=2)
    (ax1, ax2) = axs
    # P_el
    ax1.set_ylabel("$P_{el}$ in kW")
    results["SimAgent"]["room"]["P_el"].plot(ax=ax1, color=mpcplot.EBCColors.green)
    mpc_at_time_step(
        data=results["NegFlexMPC"]["NegFlexMPC"], time_step=9000, variable="P_el"
    ).ffill().plot(
        ax=ax1,
        drawstyle="steps-post",
        label="neg",
        linestyle="--",
        color=mpcplot.EBCColors.red,
    )
    mpc_at_time_step(
        data=results["PosFlexMPC"]["PosFlexMPC"], time_step=9000, variable="P_el"
    ).ffill().plot(
        ax=ax1,
        drawstyle="steps-post",
        label="pos",
        linestyle="--",
        color=mpcplot.EBCColors.blue,
    )
    mpc_at_time_step(
        data=results["FlexModel"]["Baseline"], time_step=9900, variable="P_el"
    ).ffill().plot(
        ax=ax1,
        drawstyle="steps-post",
        label="base",
        linestyle="--",
        color=mpcplot.EBCColors.dark_grey,
    )

    ax1.legend()

    mpc_at_time_step(
        data=results["NegFlexMPC"]["NegFlexMPC"], time_step=18900, variable="P_el"
    ).ffill().plot(
        ax=ax1,
        drawstyle="steps-post",
        label="neg",
        linestyle="--",
        color=mpcplot.EBCColors.red,
    )
    mpc_at_time_step(
        data=results["PosFlexMPC"]["PosFlexMPC"], time_step=18900, variable="P_el"
    ).ffill().plot(
        ax=ax1,
        drawstyle="steps-post",
        label="pos",
        linestyle="--",
        color=mpcplot.EBCColors.blue,
    )
    mpc_at_time_step(
        data=results["FlexModel"]["Baseline"], time_step=19800, variable="P_el"
    ).ffill().plot(
        ax=ax1,
        drawstyle="steps-post",
        label="base",
        linestyle="--",
        color=mpcplot.EBCColors.dark_grey,
    )

    ax1.vlines(9000, ymin=-1000, ymax=5000, colors="black")
    ax1.vlines(9900, ymin=-1000, ymax=5000, colors="black")
    ax1.vlines(10800, ymin=-1000, ymax=5000, colors="black")
    ax1.vlines(18000, ymin=-1000, ymax=5000, colors="black")

    ax1.vlines(18900, ymin=-1000, ymax=5000, colors="black")
    ax1.vlines(19800, ymin=-1000, ymax=5000, colors="black")
    ax1.vlines(20700, ymin=-1000, ymax=5000, colors="black")
    ax1.vlines(27900, ymin=-1000, ymax=5000, colors="black")

    ax1.set_ylim(-0.1, 1)

    # mdot
    ax2.set_ylabel("$\dot{m}$ in kg/s")
    results["SimAgent"]["room"]["mDot"].plot(ax=ax2, color=mpcplot.EBCColors.dark_grey)
    mpc_at_time_step(
        data=results["NegFlexMPC"]["NegFlexMPC"], time_step=9000, variable="mDot"
    ).ffill().plot(
        ax=ax2,
        drawstyle="steps-post",
        label="neg",
        linestyle="--",
        color=mpcplot.EBCColors.red,
    )
    mpc_at_time_step(
        data=results["PosFlexMPC"]["PosFlexMPC"], time_step=9000, variable="mDot"
    ).ffill().plot(
        ax=ax2,
        drawstyle="steps-post",
        label="pos",
        linestyle="--",
        color=mpcplot.EBCColors.blue,
    )
    mpc_at_time_step(
        data=results["FlexModel"]["Baseline"], time_step=9900, variable="mDot"
    ).ffill().plot(
        ax=ax2,
        drawstyle="steps-post",
        label="base",
        linestyle="--",
        color=mpcplot.EBCColors.dark_grey,
    )

    ax2.legend()

    mpc_at_time_step(
        data=results["NegFlexMPC"]["NegFlexMPC"], time_step=18900, variable="mDot"
    ).ffill().plot(
        ax=ax2,
        drawstyle="steps-post",
        label="neg",
        linestyle="--",
        color=mpcplot.EBCColors.red,
    )
    mpc_at_time_step(
        data=results["PosFlexMPC"]["PosFlexMPC"], time_step=18900, variable="mDot"
    ).ffill().plot(
        ax=ax2,
        drawstyle="steps-post",
        label="pos",
        linestyle="--",
        color=mpcplot.EBCColors.blue,
    )
    mpc_at_time_step(
        data=results["FlexModel"]["Baseline"], time_step=19800, variable="mDot"
    ).ffill().plot(
        ax=ax2,
        drawstyle="steps-post",
        label="base",
        linestyle="--",
        color=mpcplot.EBCColors.dark_grey,
    )

    ax2.vlines(9000, ymin=0, ymax=500, colors="black")
    ax2.vlines(9900, ymin=0, ymax=500, colors="black")
    ax2.vlines(10800, ymin=0, ymax=500, colors="black")
    ax2.vlines(18000, ymin=0, ymax=500, colors="black")

    ax2.vlines(18900, ymin=0, ymax=500, colors="black")
    ax2.vlines(19800, ymin=0, ymax=500, colors="black")
    ax2.vlines(20700, ymin=0, ymax=500, colors="black")
    ax2.vlines(27900, ymin=0, ymax=500, colors="black")

    ax2.set_ylim(0, 0.06)

    x_ticks = np.arange(0, until + 1, 3600)
    x_tick_labels = [int(tick / 3600) for tick in x_ticks]
    ax2.set_xticks(x_ticks)
    ax2.set_xticklabels(x_tick_labels)
    ax2.set_xlabel("Time in hours")
    for ax in axs:
        mpcplot.make_grid(ax)
        ax.set_xlim(0, until)

    # save the figure
    plt.savefig(f"plots/plots_{offer_type}/predictions.svg", format='svg')
    plt.close()

    # flexibility
    # get only the first prediction time of each time step
    ind_res = results["FlexibilityIndicator"]["FlexibilityIndicator"]
    energy_flex_neg = ind_res.xs("energyflex_neg", axis=1).droplevel(1).dropna()
    energy_flex_pos = ind_res.xs("energyflex_pos", axis=1).droplevel(1).dropna()
    fig, axs = mpcplot.make_fig(style=mpcplot.Style(use_tex=False), rows=1)
    ax1 = axs[0]
    ax1.set_ylabel("$\epsilon$ in kWh")
    energy_flex_neg.plot(ax=ax1, label="neg", color=mpcplot.EBCColors.red)
    energy_flex_pos.plot(ax=ax1, label="pos", color=mpcplot.EBCColors.blue)
    ax1.yaxis.set_major_formatter(FormatStrFormatter("%.4f"))

    ax1.legend()

    x_ticks = np.arange(0, until + 1, 3600)
    x_tick_labels = [int(tick / 3600) for tick in x_ticks]
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(x_tick_labels)
    ax1.set_xlabel("Time in hours")
    for ax in axs:
        mpcplot.make_grid(ax)
        ax.set_xlim(0, until + 1)

    # save the figure
    plt.savefig(f"plots/plots_{offer_type}/flexibility.svg", format='svg')
    plt.close()


if __name__ == "__main__":
    offer_types: list[str] = ["neg", "pos", "average"]
    for offer_type in offer_types:
        path2file: str = f'results/results_file_{offer_type}.pkl'
        objPath2file = Path(path2file)

        if objPath2file.exists():
            with open(path2file, 'rb') as results_file:
                results = pickle.load(results_file)
                results_file.close()

            plot_results_2(results=results, offer_type=offer_type, until=results.get('until'))
