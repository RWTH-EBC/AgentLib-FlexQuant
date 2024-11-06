from pathlib import Path
from typing import Union

import pandas as pd

from agentlib.core.agent import AgentConfig
from flexibility_quantification.utils.config_management import (
    load_agent_configs_of_flexquant_simulation,
    BASELINE_AGENT_KEY,
    POS_FLEX_AGENT_KEY,
    NEG_FLEX_AGENT_KEY,
    INDICATOR_AGENT_KEY,
    FLEX_MARKET_AGENT_KEY,
    SIMULATOR_AGENT_KEY,
)

from agentlib_mpc.utils.analysis import load_sim, load_mpc
from flexibility_quantification.data_structures.mpcs import BaselineMPCData, PFMPCData, NFMPCData

baselineID = BaselineMPCData().module_id
posFlexID = PFMPCData().module_id
negFlexID = NFMPCData().module_id

TIME_CONV_FACTOR = {
    "s": 1,
    "min": 60,
    "h": 3600,
    "d": 86400,
}


def load_indicator(file_path: Path) -> pd.DataFrame:
    """Load the flexibility indicator results from the given file path
    """
    df = pd.read_csv(file_path, header=0, index_col=[0, 1])
    return df


def load_market(file_path: Path) -> pd.DataFrame:
    """Load the market results from the given file path
    """
    df = pd.read_csv(file_path, header=0, index_col=[0, 1])
    return df


def load_agent_configs_and_results(agent_configs_paths: Union[list[str], list[Path]], results: Union[str, Path, dict[str, dict[str, pd.DataFrame]]]) -> tuple[dict[str, AgentConfig], dict[str, dict[str, pd.DataFrame]]]:
    """
    Load the agent configurations and results ands stats from the given file paths if necessary
    """

    # Load the agent configurations
    agent_configs = load_agent_configs_of_flexquant_simulation(agent_configs_paths)

    # Load the results
    def load_results(res_path: Union[str, Path]) -> dict[str, dict[str, pd.DataFrame]]:
        res = {
            agent_configs[SIMULATOR_AGENT_KEY].id: {
                agent_configs[SIMULATOR_AGENT_KEY].modules[1]["module_id"]: load_sim(
                    Path(res_path, Path(agent_configs[SIMULATOR_AGENT_KEY].modules[1]["result_filename"]).name))
            },
            agent_configs[BASELINE_AGENT_KEY].id: {
                agent_configs[BASELINE_AGENT_KEY].modules[1]["module_id"]: load_mpc(Path(res_path, Path(
                    agent_configs[BASELINE_AGENT_KEY].modules[1]["optimization_backend"]["results_file"]).name))
            },
            agent_configs[POS_FLEX_AGENT_KEY].id: {
                agent_configs[POS_FLEX_AGENT_KEY].modules[1]["module_id"]: load_mpc(Path(res_path, Path(
                    agent_configs[POS_FLEX_AGENT_KEY].modules[1]["optimization_backend"]["results_file"]).name))
            },
            agent_configs[NEG_FLEX_AGENT_KEY].id: {
                agent_configs[NEG_FLEX_AGENT_KEY].modules[1]["module_id"]: load_mpc(Path(res_path, Path(
                    agent_configs[NEG_FLEX_AGENT_KEY].modules[1]["optimization_backend"]["results_file"]).name))
            },
            agent_configs[INDICATOR_AGENT_KEY].id: {
                agent_configs[INDICATOR_AGENT_KEY].modules[1]["module_id"]: load_indicator(
                    Path(res_path, Path(agent_configs[INDICATOR_AGENT_KEY].modules[1]["results_file"]).name))
            },
            agent_configs[FLEX_MARKET_AGENT_KEY].id: {
                agent_configs[FLEX_MARKET_AGENT_KEY].modules[1]["module_id"]: load_market(
                    Path(res_path, Path(agent_configs[FLEX_MARKET_AGENT_KEY].modules[1]["results_file"]).name))
            },
        }
        return res

    results_path = None
    if isinstance(results, (str, Path)):
        results_path = results
        results = load_results(res_path=results_path)

    # Load the stats of the mpc agents
    def load_stats(res_path: Union[str, Path]) -> dict[str, dict[str, pd.DataFrame]]:
        # todo results["stats"] = {}

        return results

    if not results_path:
        results_path = Path()
    results = load_stats(res_path=results_path)

    return agent_configs, results


def convert_timescale_index(results: dict[str, dict[str, pd.DataFrame]], time_unit: str = "h") -> dict[str, dict[str, pd.DataFrame]]:
    """ Convert the timescale of a dataframe index (from seconds) to the given time unit

    Keyword arguments:
    results -- The dictionary of the results with the dataframes
    time_unit -- The time unit to convert the index to (default "h"; options: "s", "min", "h", "d"; assumption: index is in seconds)
    """
    for key, value in results.items():
        for sub_key, sub_value in value.items():
            if isinstance(sub_value.index, pd.MultiIndex):
                sub_value.index = pd.MultiIndex.from_arrays([sub_value.index.get_level_values(level) / TIME_CONV_FACTOR[time_unit] for level in range(sub_value.index.nlevels)])
            else:
                sub_value.index = sub_value.index / TIME_CONV_FACTOR[time_unit]
    return results
