import flexibility_quantification.data_structures.globals as glbs
from flexibility_quantification.utils.interactive import Dashboard, CustomBound
from flexibility_quantification.data_structures.flex_results import Results

def plot_results(results):
# loaded results can be passed to the Dashboard for plotting 
# alternatively, flex and sim agent config, and base path can be passed for loading results
# as shown in plot_results_mult.py
    Dashboard( 
            results = results
        ).show(
            custom_bounds=CustomBound(
                for_variable="T_zone",
                lb_name="T_lower",
                ub_name="T_upper"
            )
        )

def load_and_view_results_data():
    # results object can be loaded from the 
    # results files for further processing, e.g. plotting
    results = Results(
        flex_config="flex_configs/flexibility_agent_config.json",
        simulator_agent_config="mpc_and_sim/fmu_config.json",
        generated_flex_files_base_path=None,
    )

    # portion of the indicator result dataframe
    columns = [glbs.POWER_ALIAS_BASE, glbs.POWER_ALIAS_NEG, glbs.POWER_ALIAS_POS]
    df = results.df_indicator[columns].head()
    print(df)
    return results

if __name__ == "__main__":
    # Plotting with results extracted from current working directory as base
    results = load_and_view_results_data()
    plot_results(results)