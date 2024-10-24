from local.utils import create_series, plotscript as plot

def plot_results(setup,results):
    until = setup['sim_days'] * 86400 + setup['initial_time']
    initial_time = setup['initial_time']
    calc_flex = setup['calc_flex']
    use_scalar = setup['use_scalar']
    ts = setup['time_sample']

    if not calc_flex and use_scalar:
        plot.t_zone_casadi(results=results, initial=initial_time)
        # plot.t_parts(results=results, partsFMU="TRad", partsCasadi="T_rad",initial=initial_time)
        plot.t_parts(results, "T_in", "T_flow_in", initial=initial_time)
        plot.t_parts(results, "T_flow_out", f"T_flow_out_baseline", initial=initial_time)
        plot.t_parts(results, "T_amb", "T_amb", initial=initial_time)
        plot.t_parts(results, "T_radiator_m_1", "T_radiator_m_1", initial=initial_time)
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
        plot.t_zone_fmu(t_lower_series=t_lower_series, t_upper_series=t_upper_series, results=results,
                        initial=initial_time, until=until)
        plot.ambient(results=results, initial=initial_time, until=until, time_step=ts)
        plot.input_output_temp(results=results, initial=initial_time, ts=ts, until=until)