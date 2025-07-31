from agentlib_flexquant.utils.interactive import Dashboard, CustomBound

def plot_results(generated_flex_files_base_path):
    
    sim_config = "mpc_and_sim/fmu_config.json" 
    flex_config = "flex_configs/flexibility_agent_config.json"

    Dashboard( 
            flex_config=flex_config,
            simulator_agent_config=sim_config,
            generated_flex_files_base_path=generated_flex_files_base_path
        ).show(
            custom_bounds=CustomBound(
                for_variable="T_zone",
                lb_name="T_lower",
                ub_name="T_upper"
            )
        )

if __name__ == "__main__":
    flex_event_durations = [7200, 8100, 6300]
    for flex_event_duration in flex_event_durations:
        # Plotting with results extracted from custom base paths defined in main_multi_run.py
        # Press ctrl + C in the terminal to close plot for one run and start the plot for the next run
        # Plots for all runs remain in the browser if the corresponding windows are not closed
        plot_results(f"generated_files/run_with_flex_event_duration_{flex_event_duration}")