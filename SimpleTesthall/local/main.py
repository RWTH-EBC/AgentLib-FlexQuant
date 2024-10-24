import logging
from local.utils import config_modify,check_generate
from local.utils.select_radiator import create_radiator_record,find_radiator_type
import json
from agentlib.utils.multi_agent_system import LocalMASAgency
from agentlib_mpc.utils.plotting.interactive import show_dashboard
from local.utils import calc
from local.utils.building_gen_teaser import gen_building
from local.utils.check_generate import check_and_generate_file
import plot_mpc


with open(r"config_main.json", 'r') as f:
    setup = json.load(f)
    initial_time = setup['initial_time']

# Environment config
env_config = {"rt": False, "t_sample": 200, "offset": initial_time}

def run_example(setup):

    with_plots = setup['with_plots']
    log_level_str = setup['log_level']
    log_level = getattr(logging, log_level_str, logging.INFO)
    until = setup['sim_days'] * 86400 + setup['initial_time']
    calc_flex = setup['calc_flex']
    use_scalar = setup['use_scalar']
    ph = setup['prediction_horizon']
    ts = setup['time_sample']
    clear_results = setup['clear_results']
    price_mode = setup['price_mode']
    setup_building = setup['building_info']
    standard_outside_temp = setup_building['standard_outside_temp']

    # update FMU
    if check_and_generate_file("gen_building_history","building_model"):
        # generate new FMU
        building_model = gen_building(setup_building)
        building_model.thermal_zone_from_teaser()
        # ### figure out minimal heat demand with standard outside temperature, select radiator type
        building_model.replace_t_set_idealheater(t_set=295.15)
        building_model.create_model_idealheater(baseACH=0.4)
        heat_demand = building_model.find_max_power(n_cpu=1, log_fmu=False, n_sim=1, output_interval=3600,
                                                    standard_outside_temp=standard_outside_temp)
        # ### generate model with radiator and export FMU
        building_model.create_model_w_radiator(baseACH=0.4)
        path_radiator_record = create_radiator_record(find_radiator_type(heat_demand))
        building_model.rad_record_to_model(path_rad_record=path_radiator_record, change_mflow=True)
        building_model.reloc_zone_record()

        # update config of all agents(T_Floor)
        path_record=f"DataResource/data/{setup_building['tz_name']}_with_radiator_{setup_building['year_of_construction']}_{setup_building['location']}.mo"
        path_fmu=f"fmu/{setup_building['tz_name']}_with_radiator_{setup_building['year_of_construction']}_{setup_building['location']}.fmu"
        config_modify.update_fmu_config(path_fmu,path_record)
        config_modify.update_predictor_config(path_fmu,path_record,path_radiator_record)
        config_modify.update_mpc_config(path_record)

    agent_configs = config_modify.choose_agent_configs(cal_flexibility=calc_flex, use_scalar=use_scalar)
    config_modify.config_ts_ph(ts=ts, ph=ph, agent_configs=agent_configs)
    config_modify.choose_mode(price_mode=price_mode)

    logging.basicConfig(level=log_level)
    mas = LocalMASAgency( agent_configs=agent_configs, env=env_config,variable_logging=False,)
    mas.run(until=until)
    results = mas.get_results(cleanup=clear_results)

    save_name = f"{setup_building['tz_name']}_with_radiator_{setup_building['year_of_construction']}_{setup_building['location']}"
    calc.save_results(results=results, name=save_name)
    #save_path = "plots/cpu"
    #calc.give_solving_time(save_path)

    # Plotting
    if with_plots:
        plot_mpc(setup,results)

    show_dashboard(data=results["myMPCAgent"]['myMPC'], scale="hours")

    return results


if __name__ == "__main__":
    run_example(setup=setup)
