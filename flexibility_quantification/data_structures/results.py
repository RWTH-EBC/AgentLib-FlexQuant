from typing import Union
from pydantic import FilePath
from pathlib import Path

import pandas as pd

from agentlib.core.agent import AgentConfig
from agentlib.utils import load_config
from agentlib_mpc.modules.mpc_full import MPCConfig
from flexibility_quantification.data_structures.flexquant import FlexQuantConfig, FlexibilityIndicatorConfig, FlexibilityMarketConfig
from flexibility_quantification.data_structures.mpcs import BaselineMPCData, NFMPCData, PFMPCData
from flexibility_quantification.utils.data_handling import convert_timescale_of_index
from agentlib_mpc.utils import TimeConversionTypes
from agentlib_mpc.utils.analysis import load_sim, load_mpc, load_mpc_stats

from flexibility_quantification.modules.flexibility_indicator import FlexibilityIndicatorModuleConfig
from flexibility_quantification.modules.flexibility_market import FlexibilityMarketModuleConfig
import flexibility_quantification.utils.config_management as cmng


def load_indicator(file_path: Union[str, FilePath]) -> pd.DataFrame:
    """Load the flexibility indicator results from the given file path
    """
    df = pd.read_csv(file_path, header=0, index_col=[0, 1])
    return df


def load_market(file_path: Union[str, FilePath]) -> pd.DataFrame:
    """Load the market results from the given file path
    """
    df = pd.read_csv(file_path, header=0, index_col=[0, 1])
    return df


class Results:
    # Configs
    # Generator
    flex_config: FlexQuantConfig
    # Agents/modules
    simulator_module_config: AgentConfig
    baseline_mpc_module_config: MPCConfig
    pos_flex_mpc_module_config: MPCConfig
    neg_flex_mpc_module_config: MPCConfig
    indicator_module_config: FlexibilityIndicatorModuleConfig
    market_module_config: FlexibilityMarketModuleConfig

    # Dataframes
    df_simulation: pd.DataFrame()
    df_baseline: pd.DataFrame()
    df_baseline_stats: pd.DataFrame()
    df_pos_flex: pd.DataFrame()
    df_pos_flex_stats: pd.DataFrame()
    df_neg_flex: pd.DataFrame()
    df_neg_flex_stats: pd.DataFrame()
    df_indicator: pd.DataFrame()
    df_flex_market: pd.DataFrame()

    # time conversion
    current_timescale: TimeConversionTypes = "seconds"
    
    def __init__(
            self,
            flex_config: Union[str, FilePath, FlexQuantConfig],
            simulator_agent_config: Union[str, FilePath, AgentConfig],
            results: Union[str, FilePath, dict[str, dict[str, pd.DataFrame]]] = None,
            to_timescale: TimeConversionTypes = "seconds"
    ):
        # load configs of agents and modules
        # flex config
        self.flex_config = load_config.load_config(
            config=flex_config, config_type=FlexQuantConfig
        )

        # simulator
        self.simulator_agent_config = load_config.load_config(
            config=simulator_agent_config, config_type=AgentConfig
        )
        self.simulator_module_config = self.simulator_agent_config.modules[1]
        # didn't work out:
        # self.simulator_module_config = cmng.get_module(
        #     config=self.simulator_agent_config, module_type=cmng.SIMULATOR_CONFIG_TYPE
        # )

        # get names of the config files
        baseline_name_of_created_file = BaselineMPCData().name_of_created_file
        pos_flex_name_of_created_file = PFMPCData().name_of_created_file
        neg_flex_name_of_created_file = NFMPCData().name_of_created_file
        indicator_name_of_created_file = self.flex_config.indicator_config.name_of_created_file
        market_name_of_created_file = FlexibilityMarketConfig(agent_config="").name_of_created_file     # todo: clean solution to prevent pydantic ValidationError https://errors.pydantic.dev/2.9/v/missing

        for file_path in Path(self.flex_config.path_to_flex_files).rglob("*.json"):
            if file_path.name in baseline_name_of_created_file:
                self.baseline_mpc_agent_config = load_config.load_config(
                    config=file_path, config_type=AgentConfig
                )
                self.baseline_mpc_module_config = cmng.get_module(
                    config=self.baseline_mpc_agent_config, module_type=cmng.BASELINEMPC_CONFIG_TYPE
                )

            elif file_path.name in pos_flex_name_of_created_file:
                self.pos_flex_mpc_agent_config = load_config.load_config(
                    config=file_path, config_type=AgentConfig
                )
                self.pos_flex_mpc_module_config = cmng.get_module(
                    config=self.pos_flex_mpc_agent_config, module_type=cmng.SHADOWMPC_CONFIG_TYPE
                )

            elif file_path.name in neg_flex_name_of_created_file:
                self.neg_flex_mpc_agent_config = load_config.load_config(
                    config=file_path, config_type=AgentConfig
                )
                self.neg_flex_mpc_module_config = cmng.get_module(
                    config=self.neg_flex_mpc_agent_config, module_type=cmng.SHADOWMPC_CONFIG_TYPE
                )

            elif file_path.name in indicator_name_of_created_file:
                self.indicator_config = load_config.load_config(
                    config=self.flex_config.indicator_config, config_type=FlexibilityIndicatorConfig
                )
                self.indicator_agent_config = load_config.load_config(
                    config=file_path, config_type=AgentConfig
                )
                self.indicator_module_config = cmng.get_module(
                    config=self.indicator_agent_config, module_type=cmng.INDICATOR_CONFIG_TYPE
                )

            elif file_path.name in market_name_of_created_file:
                self.market_config = load_config.load_config(
                    config=self.flex_config.market_config, config_type=FlexibilityMarketConfig
                )
                self.market_agent_config = load_config.load_config(
                    config=file_path, config_type=AgentConfig
                )
                self.market_module_config = cmng.get_module(
                    config=self.market_agent_config, module_type=cmng.MARKET_CONFIG_TYPE
                )

            else:
                raise ValueError(f"Unexpected json-file found: {file_path.name}")

        # Get agent and module ids  # todo: cleanup
        self.simulator_agent_id = self.simulator_agent_config.id
        self.simulator_module_id = self.simulator_module_config["module_id"]
        self.baseline_agent_id = self.baseline_mpc_agent_config.id
        self.baseline_module_id = self.baseline_mpc_module_config.module_id
        self.pos_flex_agent_id = self.pos_flex_mpc_agent_config.id
        self.pos_flex_module_id = self.pos_flex_mpc_module_config.module_id
        self.neg_flex_agent_id = self.neg_flex_mpc_agent_config.id
        self.neg_flex_module_id = self.neg_flex_mpc_module_config.module_id
        self.indicator_agent_id = self.indicator_agent_config.id
        self.indicator_module_id = self.indicator_module_config.module_id
        self.flex_market_agent_id = self.market_agent_config.id
        self.flex_market_module_id = self.market_module_config.module_id

        # load results
        if results is None:
            results_path = Path(self.indicator_module_config.results_file).parent
            results = self._load_results(res_path=results_path)
        if isinstance(results, (str, Path)):
            results_path = results
            results = self._load_results(res_path=results_path)
        elif isinstance(results, dict):
            results_path = Path(self.indicator_module_config.results_file).parent
        else:
            raise ValueError("results must be a path or dict")

        # Get dataframes
        self.df_simulation = results[self.simulator_agent_config.id][self.simulator_module_config["module_id"]]
        self.df_baseline = results[self.baseline_mpc_agent_config.id][self.baseline_mpc_module_config.module_id]
        self.df_pos_flex = results[self.pos_flex_mpc_agent_config.id][self.pos_flex_mpc_module_config.module_id]
        self.df_neg_flex = results[self.neg_flex_mpc_agent_config.id][self.neg_flex_mpc_module_config.module_id]
        self.df_indicator = results[self.indicator_agent_config.id][self.indicator_module_config.module_id]
        self.df_flex_market = results[self.market_agent_config.id][self.market_module_config.module_id]

        self.df_baseline_stats = load_mpc_stats(
            Path(results_path, Path(self.baseline_mpc_module_config.optimization_backend["results_file"]).name)
        )
        self.df_pos_flex_stats = load_mpc_stats(
            Path(results_path, Path(self.pos_flex_mpc_module_config.optimization_backend["results_file"]).name)
        )
        self.df_neg_flex_stats = load_mpc_stats(
            Path(results_path, Path(self.neg_flex_mpc_module_config.optimization_backend["results_file"]).name)
        )

        # Convert the time in the dataframes to the desired timescale
        self.convert_timescale_of_dataframe_index(to_timescale=to_timescale)

    def _load_results(self, res_path: Union[str, Path]) -> dict[str, dict[str, pd.DataFrame]]:
        res = {
            self.simulator_agent_config.id: {
                self.simulator_module_config["module_id"]:
                    load_sim(Path(res_path, Path(self.simulator_module_config["result_filename"]).name))
            },
            self.baseline_mpc_agent_config.id: {
                self.baseline_mpc_module_config.module_id:
                    load_mpc(Path(res_path, Path(self.baseline_mpc_module_config.optimization_backend["results_file"]).name))
            },
            self.pos_flex_mpc_agent_config.id: {
                self.pos_flex_mpc_module_config.module_id:
                    load_mpc(Path(res_path, Path(self.pos_flex_mpc_module_config.optimization_backend["results_file"]).name))
            },
            self.neg_flex_mpc_agent_config.id: {
                self.neg_flex_mpc_module_config.module_id:
                    load_mpc(Path(res_path, Path(self.neg_flex_mpc_module_config.optimization_backend["results_file"]).name))
            },
            self.indicator_agent_config.id: {
                self.indicator_module_config.module_id:
                    load_indicator(Path(res_path, Path(self.indicator_module_config.results_file).name))
            },
            self.market_agent_config.id: {
                self.market_module_config.module_id:
                    load_market(Path(res_path, Path(self.market_module_config.results_file).name))
            }
        }
        return res

    def convert_timescale_of_dataframe_index(self, to_timescale: TimeConversionTypes):
        """ Convert the time in the dataframes to the desired timescale

        Keyword arguments:
        timescale -- The timescale to convert the data to
        """
        # Convert the time in the dataframes
        for df in [
            self.df_simulation,
            self.df_baseline, self.df_baseline_stats,
            self.df_pos_flex, self.df_pos_flex_stats,
            self.df_neg_flex, self.df_neg_flex_stats,
            self.df_indicator,
            self.df_flex_market
        ]:
            convert_timescale_of_index(df=df, from_unit=self.current_timescale, to_unit=to_timescale)

        self.current_timescale = to_timescale
