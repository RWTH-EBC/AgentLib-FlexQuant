from flexibility_quantification.utils.interactive import Dashboard, CustomBound

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
    # Custom base path for generated flex files directory (config + results directory)
    generated_files_path_run_1 = "C:/Users/fse-ksu/flex_test/run_1"
    generated_files_path_run_2 = "C:/Users/fse-ksu/flex_test/run_2"
    generated_files_path_run_3 = "C:/Users/fse-ksu/flex_test/run_3"

    # PLotting results
    # Plotting with results extracted from current working directory as base
    plot_results(None)
    # Plotting with results extracted from custom base paths
    plot_results(generated_files_path_run_1)
    plot_results(generated_files_path_run_2)
    plot_results(generated_files_path_run_3)