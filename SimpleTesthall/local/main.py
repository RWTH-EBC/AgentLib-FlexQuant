import logging
from local.utils import config_modify, create_series, calc
from local.utils.modelicaprocess import ModelicaProcessor
import pickle
import plotscript as plot
import json
from agentlib.utils.multi_agent_system import LocalMASAgency
from agentlib_mpc.utils.plotting.interactive import show_dashboard
from local.utils import calc
from local.utils.building_gen_teaser import gen_building
from flexibility_quantification.generate_flex_agents import FlexAgentGenerator



# Environment config
env_config = {"rt": False, "t_sample": 200, "offset": 30585600}

def run_example(setup):
    with open(setup, 'r') as f:
        setup = json.load(f)

    with_plots = setup['with_plots']
    log_level_str = setup['log_level']
    log_level = getattr(logging, log_level_str, logging.INFO)
    until = setup['sim_days'] * 86400 + setup['initial_time']
    initial_time = setup['initial_time']
    calc_flex = setup['calc_flex']
    use_scalar = setup['use_scalar']
    ph = setup['prediction_horizon']
    ts = setup['time_sample']
    clear_results = setup['clear_results']
    price_mode = setup['price_mode']
    update_simulator = setup['update_simulator']

    setup_building = setup['building_info']


    # update FMU
    if update_simulator:
        # generate new FMU
        building_model = gen_building(setup_building)
        building_model.thermal_zone_from_teaser()
        # find max.heat demand
        #building_model.replace_t_set_idealheater(t_set=295.15)
        #building_model.create_model_idealheat(baseACH=0.4)
        #building_model.find_max_power(n_cpu=1, log_fmu=False, n_sim=1, output_interval=3600,
        #                              standard_outside_temp=standard_outside_temp)
        #TODO: add radiator selection
        building_model.create_model_w_radiator(baseACH=0.4)
        building_model.rad_record_to_model(path_rad_record=setup_building["radiator_record"],
                                           change_mflow=True)
        building_model.reloc_zone_record()

        # update config of all agents(T_Floor)
        path_record=f"DataResource/data/{setup_building['tz_name']}_with_radiator_{setup_building['year_of_construction']}_{setup_building['location']}.mo"
        path_fmu=f"fmu/{setup_building['tz_name']}_with_radiator_{setup_building['year_of_construction']}_{setup_building['location']}.fmu"
        config_modify.update_fmu_config(path_fmu,path_record)
        config_modify.update_predictor_config(path_fmu,path_record,setup_building["radiator_record"])
        config_modify.update_mpc_config(path_record)

    #TODO: utilize simulation time and sim disturbance time
    # Set the log-level



    agent_configs = config_modify.choose_agent_configs(cal_flexibility=calc_flex, use_scalar=use_scalar)
    if calc_flex:
        config_list = FlexAgentGenerator(
            flex_config=agent_configs[0], mpc_agent_config=agent_configs[1]
        ).generate_flex_agents()
        agent_configs.extend(config_list)
        agent_configs = agent_configs[2:8]

    config_modify.config_ts_ph(ts=ts, ph=ph, agent_configs=agent_configs)
    config_modify.choose_mode(price_mode=price_mode)
    #config_modify.config_time_traj(ts=ts,n=ph,config_name="NF_mpc")
    #config_modify.config_time_traj(ts=ts, n=ph, config_name="PF_mpc")


    # Start capturing console output
    #capture = IpoptOutputCapture()
    #capture.start_capture()

    logging.basicConfig(level=log_level)
    mas = LocalMASAgency(
        agent_configs=agent_configs,
        env=env_config,
        variable_logging=False,
    )
    mas.run(until=until)
    # Stop capturing and get the captured output
    #captured_output = capture.stop_capture()
    # Process and print IPOPT times
    #IpoptOutputCapture.process_ipopt_times(captured_output)

    results = mas.get_results(cleanup=clear_results)
    save_name = f"{setup_building['tz_name']}_with_radiator_{setup_building['year_of_construction']}_{setup_building['location']}"
    calc.save_results(results=results, name=save_name)

    # Plotting
    if with_plots:
        if not calc_flex and use_scalar:
            plot.t_zone_casadi(results=results, initial=initial_time)
            #plot.t_parts(results=results, partsFMU="TRad", partsCasadi="T_rad",initial=initial_time)
            plot.t_parts(results, "T_in", "T_flow_in",initial=initial_time)
            plot.t_parts(results, "T_flow_out", f"T_flow_out_baseline",initial=initial_time)
            plot.t_parts(results,"T_amb","T_amb",initial=initial_time)
            plot.t_parts(results, "T_radiator_m_1", "T_radiator_m_1",initial=initial_time)
            plot.t_parts(results, "T_Roof", "T_Roof", initial=initial_time)
            plot.t_parts(results, "T_ExtWall", "T_ExtWall", initial=initial_time)
            plot.Q_parts(results, "Q_flow_sum", "Q_flow_total", initial_time)
            plot.Q_parts(results, "Q_conv", "Q_conv_debug", initial_time)
            plot.Q_parts(results, "Q_rad", "Q_rad_debug", initial_time)
            plot.Q_parts(results, "solar_radiation", "Q_RadSol", initial_time)
        elif calc_flex:
            t_upper_series = create_series.create_mpc_series(results_mpc=results["myMPCAgent"]['myMPC'],
                                                             variable_mpc="T_upper", value_type="parameter")
            t_lower_series = create_series.create_mpc_series(results_mpc=results["myMPCAgent"]['myMPC'],
                                                             variable_mpc="T_lower", value_type="parameter")
            p_el_series = create_series.create_pel_series(results_mpc=results["myFlexibilityAgent"]["myFlexibility"],
                                                          variable_mpc="P_el_alg", initial_time=initial_time)
            p_el_min_series = create_series.create_pel_series(
                results_mpc=results["myFlexibilityAgent"]["myFlexibility"],
                variable_mpc="P_el_min_alg", initial_time=initial_time)
            p_el_max_series = create_series.create_pel_series(
                results_mpc=results["myFlexibilityAgent"]["myFlexibility"],
                variable_mpc="P_el_max_alg", initial_time=initial_time)
            r_pel_series = create_series.create_pel_series(results_mpc=results["myFlexibilityAgent"]["myFlexibility"],
                                                           variable_mpc="r_pel", initial_time=initial_time)
            T_flow_out_baseline_series = create_series.process_collocation_points(
                results_mpc=results["myMPCAgent"]['myMPC'],
                variable_mpc='T_flow_out_baseline',
                value_type='variable', ts=ts)
            T_flow_out_PF_series = create_series.process_collocation_points(
                results_mpc=results["myMPCAgent_positive"]['myPFMPC'],
                variable_mpc='T_flow_out_PF',
                value_type='variable', ts=ts)
            T_flow_out_NF_series = create_series.process_collocation_points(
                results_mpc=results["myMPCAgent_negative"]['myNFMPC'],
                variable_mpc='T_flow_out_NF',
                value_type='variable', ts=ts)

            # T_room with comfort limit
            plot.t_zone_fmu(t_lower_series=t_lower_series, t_upper_series=t_upper_series, results=results,
                            initial=initial_time,
                            until=until)
            plot.input_output_temp(results=results, initial=initial_time, ts=ts, until=until)
            # disturbance: solar&amb.temp
            plot.ambient(results=results, initial=initial_time, until=until, time_step=ts)
            # baseline Pel & price
            plot.p_el_price(p_el_series=p_el_series, r_pel_series=r_pel_series, until=until, initial=initial_time)

            # Debug
            # T_room from baseline and shadow MPCs
            plot.t_zone_all_mpc(t_lower_series, t_upper_series, results, until, initial_time)
            plot.t_zone_one_prediction(results=results, initial=initial_time, time_step=ts)
            # pel all mpcs
            plot.pel_all_mpcs(p_el_series, p_el_max_series, p_el_min_series, r_pel_series, until, initial_time)
            plot.p_el_one_prediction(results, initial_time, time_step=ts)
            # T_flow_in all mpcs
            plot.input_temp_all_mpc(results, initial_time, until)
            # T_flow_out all mpcs
            plot.t_out_debug(t_baseline_series=T_flow_out_baseline_series, t_PF_series=T_flow_out_PF_series,
                             t_NF_series=T_flow_out_NF_series, initial=initial_time, until=until)

            # plot KPI
            energyflex_neg = create_series.create_flex_after_pre(
                results=results["myFlexibilityAgent"]["myFlexibility"], var="energyflex_neg")
            energyflex_pos = create_series.create_flex_after_pre(
                results=results["myFlexibilityAgent"]["myFlexibility"], var="energyflex_pos")
            powerflex_avg_neg = create_series.create_flex_after_pre(
                results=results["myFlexibilityAgent"]["myFlexibility"], var="powerflex_avg_neg")
            powerflex_avg_pos = create_series.create_flex_after_pre(
                results=results["myFlexibilityAgent"]["myFlexibility"], var="powerflex_avg_pos")
            powerflex_neg_min, powerflex_neg_max = create_series.extract_min_max_flex(
                results["myFlexibilityAgent"]["myFlexibility"]["powerflex_flex_neg"])
            powerflex_pos_min, powerflex_pos_max = create_series.extract_min_max_flex(
                results["myFlexibilityAgent"]["myFlexibility"]["powerflex_flex_pos"])
            costs_neg = create_series.create_flex_after_pre(
                results=results["myFlexibilityAgent"]["myFlexibility"], var="costs_neg")
            costs_neg_rel = create_series.create_flex_after_pre(
                results=results["myFlexibilityAgent"]["myFlexibility"], var="costs_neg_rel")
            costs_pos = create_series.create_flex_after_pre(
                results=results["myFlexibilityAgent"]["myFlexibility"], var="costs_pos")
            costs_pos_rel = create_series.create_flex_after_pre(
                results=results["myFlexibilityAgent"]["myFlexibility"], var="costs_pos_rel")
            # plot energy flexibility
            plot.plot_energyflex_neg_pos(energyflex_neg, energyflex_pos, until, initial_time)
            # plot power flex: avg & max,min
            plot.plot_poweravg_min_max(powerflex_avg_neg, powerflex_avg_pos, powerflex_neg_min, powerflex_neg_max,
                                       powerflex_pos_min, powerflex_pos_max, until, initial_time)
            # plot energy flexibility and absolute cost
            plot.plot_energyflexibility_costs_bar(energyflex_neg, energyflex_pos, costs_neg, costs_pos, ts,
                                                  initial_time, until)
            # plot relative cost and electricity price
            plot.plot_costs_price(costs_neg_rel, costs_pos_rel, r_pel_series, until, initial_time)
        else:
            t_upper_series = create_series.create_mpc_series(results_mpc=results["myMPCAgent"]['myMPC'],
                                                             variable_mpc="T_upper", value_type="parameter")
            t_lower_series = create_series.create_mpc_series(results_mpc=results["myMPCAgent"]['myMPC'],
                                                             variable_mpc="T_lower", value_type="parameter")
            pel_series = create_series.process_collocation_points(results_mpc=results["myMPCAgent"]['myMPC'],
                                                                  variable_mpc='P_el_alg',
                                                                  value_type='variable', ts=ts)
            r_pel_series = create_series.create_mpc_series(results_mpc=results["myMPCAgent"]["myMPC"],
                                                           variable_mpc="r_pel", value_type="parameter")
            T_rad_series = create_series.process_collocation_points(results_mpc=results["myMPCAgent"]['myMPC'],
                                                                    variable_mpc='T_rad',
                                                                    value_type='variable', ts=ts)
            plot.p_el_price(pel_series, r_pel_series, until, initial_time)
            plot.t_zone_fmu(t_lower_series=t_lower_series, t_upper_series=t_upper_series, results=results,initial=initial_time,until=until)
            plot.ambient(results=results, initial=initial_time, until=until, time_step=ts)
            plot.input_output_temp(results=results, initial=initial_time, ts=ts, until=until)

        #save_path = "plots/cpu"
        #calc.give_solving_time(save_path)
        show_dashboard(data=results["myMPCAgent"]['myMPC'], scale="hours")

    return results


if __name__ == "__main__":
    run_example(setup=r"config_main.json")
