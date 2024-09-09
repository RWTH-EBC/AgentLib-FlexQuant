import main_flex
import json
from Model.local.predictor.simple_predictor import power_price_func
import settings
import pickle
import plots
import os

predictor_config = "Model//local//predictor//config.json"
mpc_config = f"Model//local//mpc//config.json"
flex_config = f"flexibility_agent_config.json"
plots.plt.ioff()
def set_value(json_path, key, value):
    with open(json_path) as f:
        d = json.load(f)
    d[key] = value
    with open(json_path, "w+") as f:
        json.dump(d, f, indent=4)

fmu = True

gewichte_ = [ 
        [{
            "name": "s_T1",
            "value": 250
        },
        {
            "name": "s_Pel1",
            "value": 50
        },
        {
            "name": "s_T2",
            "value": 250
        },
        {
            "name": "s_Pel2",
            "value": 100
        }],
    ]
for val in gewichte_:
    name_f = "_".join([str(v["value"])
     for v in val])
    folder = f"debug/debug_1_halb_{name_f}_{fmu}"
    if not os.path.exists(folder):
        os.mkdir(folder)
    set_value(flex_config, "weights", val)
    results = []
    day = 2
    typ = "negative"
    duration = 0.5

    for rate in [1]:#,0.8,0.6,0.4,0.2,0]:
        # set_value(flex_config, "forced_offers", {f"{int(372600+86400*(day-2))}.0": [typ,rate]}) 
        set_value(flex_config, "forced_offers", {}) 
        set_value(flex_config, "offer_acceptance_rate", 0)
        set_value(flex_config, "cooldown_timesteps", 0)
        set_value(flex_config, "delete_files", False)
        agent_configs, env_config, initial_time, until, time_step = main_flex.get_configs(predictor_config, mpc_config, flex_config, start_day=day, duration=duration, fmu=fmu)
        results_with_flex = main_flex.run_example(agent_configs, env_config, with_plots=True, until=until,  initial_time=initial_time, time_step=time_step)
        results.append(results_with_flex)

    # set_value(flex_config, "forced_offers", {}) 
    # set_value(flex_config, "offer_acceptance_rate", 0)
    # set_value(flex_config, "cooldown_timesteps", 0)
    # set_value(flex_config, "delete_files", True)

    # agent_configs, env_config, initial_time, until, time_step = main_flex.get_configs(predictor_config, mpc_config, flex_config, start_day=day, duration=duration, fmu=fmu)

    # results_wo_flex = main_flex.run_example(agent_configs, env_config, with_plots=False, initial_time=initial_time, until=until, time_step=time_step)
    # if fmu:
    #     electricity_prices = power_price_func(results_wo_flex["SimAgent"]["SimTestHall"]["TAirRoom"].index, settings.varying_price_signal)
    # else:
    #     electricity_prices = power_price_func(results_wo_flex["SimAgent"]["SimTestHall"]["P_el_c"].index, settings.varying_price_signal)

    with open(f"{folder}/flex_res_day_{day}_{typ}.res", "w+b") as f:
    #     pickle.dump([results_wo_flex, results, electricity_prices], f)
        pickle.dump(results[0], f)

    # fig = plots.plot_flex_price_graph(results_wo_flex, results, 372600+86400*(day-2), electricity_prices, fmu=fmu)
    # fig.savefig(f"{folder}/fig_flex_day{day}_{typ}.png")
    # fig = plots.plot_power_simulation(results_wo_flex, fmu=fmu, detailed=False)
    # fig.savefig(f"{folder}/fig_baseline_p_day{day}.png")
    # fig = plots.plot_temperature_simulation(results_wo_flex, fmu=fmu)
    # fig.savefig(f"{folder}/fig_baseline_t_day{day}.png")
    for i, res in enumerate(results):
        fig = plots.plot_power_simulation(res, fmu=fmu, detailed=False)
        fig.savefig(f"{folder}/fig_p_day{day}_{typ}_{i}.png")
        fig = plots.plot_temperature_simulation(res, fmu=fmu)
        fig.savefig(f"{folder}/fig_t_day{day}_{typ}_{i}.png")
        # fig = plots.plot_prediction(res, [372600+86400*(day-2)])
        # for indo, f in enumerate(fig):
        #     f.savefig(f"{folder}/{indo}_fig_pred_day{day}_{typ}_{i}.png")
        fig = plots.debug_print(res)
        fig.savefig(f"{folder}/Q_tabs.png")

        fig.show()
        plots.plt.close("all")
        
# if not os.path.exists(folder):
#     os.mkdir(folder)
# set_value(flex_config, "forced_offers", {}) 
# set_value(flex_config, "offer_acceptance_rate", 0.7)
# set_value(flex_config, "pos_neg_rate", 1)
# set_value(flex_config, "cooldown_timesteps", 7)
# set_value(flex_config, "delete_files", True)

# agent_configs, env_config, initial_time, until, time_step = main_flex.get_configs(predictor_config, mpc_config, flex_config, start_day=0, duration=5, fmu=fmu)

# resu = main_flex.run_example(agent_configs, env_config, with_plots=False, initial_time=initial_time, until=until, time_step=time_step)
# with open(f"{folder}/5day.res", "w+b") as f:
#     pickle.dump(resu, f)

# fig = plots.plot_power_simulation(resu, fmu=fmu)
# fig.savefig(f"{folder}/fig_p_5.png")
# fig = plots.plot_temperature_simulation(resu,fmu=fmu)
# fig.savefig(f"{folder}/fig_t_5.png")
# fig = plots.plot_flex_amount(resu,fmu=fmu)
# fig.savefig(f"{folder}/fig_flex_5.png")

 