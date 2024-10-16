import csv
import sys

# Write Warnings
def write_solver_waring():
    """
    Returns a warning, if solver ist not successful
    Result-file paths are add manually

    Args:
        None
    Returns:
        Warning which indicates which MPC is not successful at which time
    """
    file_paths = {
        'results/stats_mpc_simple_building_local_broadcast.csv': 'MPC',
        'results/stats_mpc_maxPel_local_broadcast.csv': 'Max MPC',
        'results/stats_mpc_minPel_local_broadcast.csv': 'Min MPC'
    }

    for file_path, solver_name in file_paths.items():
        with open(file_path) as file:
            csv_reader = csv.reader(file, delimiter=',')
            next(csv_reader)
            data = list(csv_reader)

            success = []
            for row in data:
                successful = row[12]
                success.append(successful)
                if successful == 'False':
                    print('\033[91mWarning: Solver of', solver_name, 'not successful at time_step:', row[0], file=sys.stderr)

def write_Q_slack_warning(results):
    """
    Returns a warning if a Slack Variable is below zero
    Agents, AgentId and Slack Variables are add manually

    Args:
        results: Complete results of run_example(): mas
    Returns:
        Warning which indicates which MPC contains Slack Variable below zero at which time
    """
    Agent = ['myMPCAgent',
             'myMPCAgent_maxPel',
             'myMPCAgent_minPel']
    MPC = ['myMPC',
           'myMPC_maxPel',
           'myMPC_minPel']
    Slack = ['Q_tabs_slack1',
             'Q_tabs_slack2',
             'Q_ahu_slack1',
             'Q_ahu_slack2']

    for agent in Agent:
        for mpc in MPC:
            if agent in results and mpc in results[agent]:
                variable = results[agent][mpc]['variable']
                for slack in Slack:
                    if slack in variable:
                        values = results[agent][mpc]['variable'][slack].values
                        index = results[agent][mpc]['variable'][slack].index

                        for i, value in enumerate(values):
                            if value < 0:
                                print('\033[91mWarning:', slack,'in', mpc, 'below zero at', index[i], file=sys.stderr)