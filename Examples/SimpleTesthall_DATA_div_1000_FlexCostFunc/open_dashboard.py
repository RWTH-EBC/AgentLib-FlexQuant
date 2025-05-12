from flexibility_quantification.utils.interactive import Dashboard, CustomBound
import pickle


def create_dash_board(result_data=None) -> None:
    Dashboard(
        flex_config="flex_configs/flexibility_agent_config.json",
        simulator_agent_config="model/local/mpc/ca_simu.json",
        results=result_data
    ).show(
        custom_bounds=CustomBound(
            for_variable="T_out",
            lb_name="T_lower",
            ub_name="T_upper"
        )
    )


if __name__ == "__main__":

    with open('results/results_file_neg.pkl', 'rb') as results_file:
        results = pickle.load(results_file)

    create_dash_board(result_data=results)
