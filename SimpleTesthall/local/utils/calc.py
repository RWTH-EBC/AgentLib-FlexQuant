import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib as tikz
import pickle


def costs_per_day(costs):
    """
    Function cumulates the costs of the flexibility event over one day

    Args:
        costs: predicted costs per flexibility event
    Returns:
        cumulative_costs_df: costs per day
    """
    day = 86400
    cumulative_costs_df = pd.DataFrame(columns=[f"Cumulative_cost"])
    costs = costs.sort_index()

    cumulative_costs = 0
    current_day = 0

    for time, cost in zip(costs.index, costs.values):
        if time <= (current_day + 1) * day:
            cumulative_costs += cost
        else:
            cumulative_costs_df.loc[current_day] = cumulative_costs
            cumulative_costs = cost
            current_day += 1

    cumulative_costs_df.loc[current_day] = cumulative_costs

    return cumulative_costs_df


def energyflex_per_day(eps):
    """
    Function cumulates the energy Flexibility of the flexibility event over one day

    Args:
        eps: predicted energy flexibility per flexibility event
    Returns:
        cumulative_energy_df: energyflexibility per day
    """
    day = 86400
    cumulative_energy_df = pd.DataFrame(columns=[f"Cumulative_Energyflex"])
    energy = eps.sort_index()

    cumulative_energy = 0
    current_day = 0

    for time, energy in zip(energy.index, energy.values):
        if time <= (current_day + 1) * day:
            cumulative_energy += energy
        else:
            cumulative_energy_df.loc[current_day] = cumulative_energy
            cumulative_energy = energy
            current_day += 1

    cumulative_energy_df.loc[current_day] = cumulative_energy

    return cumulative_energy_df


def give_solving_time(save_path):
    path = "results"
    for filename in os.listdir(path):
        if "stats" in filename:
            df = pd.read_csv(f"{path}/{filename}", delimiter=',')
            cpu_times = df["t_proc_total"]
            real_sol_times = df["t_wall_total"]
            get_stats_eval_plot(cpu_times, filename=os.path.splitext(filename)[0], save_path=save_path, name="cpu_time")
            get_stats_eval_plot(real_sol_times, filename=os.path.splitext(filename)[0], save_path=save_path,
                                name="real_solve_time")


def get_stats_eval_plot(data, filename, save_path, name):
    mean = np.mean(data)
    median = np.median(data)
    quartiles = np.percentile(data, [5, 50, 95])

    print(f"Mean {name}: {mean}")
    print(f"Median {name}: {median}")
    print(f"Lower Quartile (5%) {name}: {quartiles[0]}")
    print(f"Upper Quartile (95%) {name}: {quartiles[2]}")

    fig, ax = plt.subplots()
    ax.boxplot(data, vert=False, patch_artist=True)
    ax.set_title(filename + ' ' + name)

    ax.set_yticklabels([''])
    stats_text = f"Mean {name}: {mean}\n" \
                 f"Median {name}: {median}\n" \
                 f"Lower Quartile (5%): {quartiles[0]}\n" \
                 f"Upper Quartile (95%): {quartiles[2]}"
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    ax.text(xlim[1] * 0.5, ylim[1] * 0.5, stats_text, fontsize=10, verticalalignment='center')

    os.makedirs(save_path, exist_ok=True)

    plt.savefig(f"{save_path}/boxplot{filename}_{name}.svg", format="svg")
    tikz.save(f"{save_path}/boxplot{filename}_{name}.tex")

    plt.show()


def save_results(results, name):
    save_path = "roaming_results"
    os.makedirs(save_path, exist_ok=True)
    with open(f'{save_path}/results_{name}.pkl', 'wb') as f:
        pickle.dump(results, f)
