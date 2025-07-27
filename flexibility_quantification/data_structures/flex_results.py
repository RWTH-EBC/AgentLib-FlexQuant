import agentlib
import json
import os
import pandas as pd
import flexibility_quantification.utils.config_management as cmng
from copy import deepcopy
from typing import Union, Optional
from pydantic import FilePath
from pathlib import Path
from agentlib.core.agent import AgentConfig
from agentlib.core.module import BaseModuleConfig
from agentlib.utils import load_config
from agentlib.modules.simulation.simulator import SimulatorConfig
from agentlib_mpc.modules.mpc import BaseMPCConfig
from agentlib_mpc.utils import TimeConversionTypes
from agentlib_mpc.utils.analysis import load_sim, load_mpc, load_mpc_stats
from flexibility_quantification.data_structures.flexquant import (
    FlexQuantConfig,
    FlexibilityIndicatorConfig,
    FlexibilityMarketConfig,
)
from flexibility_quantification.data_structures.mpcs import (
    BaselineMPCData,
    NFMPCData,
    PFMPCData,
)
from flexibility_quantification.utils.data_handling import convert_timescale_of_index
from flexibility_quantification.modules.flexibility_indicator import FlexibilityIndicatorModuleConfig
from flexibility_quantification.modules.flexibility_market import FlexibilityMarketModuleConfig


def load_indicator_results(file_path: Union[str, FilePath]) -> pd.DataFrame:
    """
    Load the flexibility indicator results from the given file path
    Args:
        file_path: the file path of the indicator results file
    Returns:
        df: DataFrame containing the indicator results
    """
    df = pd.read_csv(file_path, header=0, index_col=[0, 1])
    return df


def load_market_results(file_path: Union[str, FilePath]) -> pd.DataFrame:
    """
    Load the market results from the given file path
    Args:
        file_path: the file path of the market results file
    Returns:
        df: DataFrame containing the market results
    """
    df = pd.read_csv(file_path, header=0, index_col=[0, 1])
    return df


class Results:
    # Configs:
    # Generator
    generator_config: FlexQuantConfig
    # Agents
    simulator_agent_config: AgentConfig
    baseline_agent_config: AgentConfig
    pos_flex_agent_config: AgentConfig
    neg_flex_agent_config: AgentConfig
    indicator_agent_config: AgentConfig
    market_agent_config: AgentConfig
    # Modules
    simulator_module_config: SimulatorConfig
    baseline_module_config: BaseMPCConfig
    pos_flex_module_config: BaseMPCConfig
    neg_flex_module_config: BaseMPCConfig
    indicator_module_config: FlexibilityIndicatorModuleConfig
    market_module_config: FlexibilityMarketModuleConfig

    # Dataframes
    df_simulation: pd.DataFrame
    df_baseline: pd.DataFrame
    df_pos_flex: pd.DataFrame
    df_neg_flex: pd.DataFrame
    df_indicator: pd.DataFrame
    df_market: pd.DataFrame

    # Stats of the MPCs
    df_baseline_stats: pd.DataFrame
    df_pos_flex_stats: pd.DataFrame
    df_neg_flex_stats: pd.DataFrame

    # time conversion
    current_timescale_of_data: TimeConversionTypes = "seconds"

    def __init__(
        self,
        flex_config: Optional[Union[str, FilePath, dict]],
        simulator_agent_config: Optional[Union[str, FilePath, dict]],
        generated_flex_files_base_path: Optional[Union[str, FilePath]] = None,
        results: Optional[Union[str, FilePath, dict[str, dict[str, pd.DataFrame]], "Results"]] = None,
        to_timescale: TimeConversionTypes = "seconds",
    ):
        if isinstance(results, Results):
            # Already a Results instance — copy over its data
            self.__dict__ = deepcopy(results.__dict__)
            return
        # if generated flex files are saved at a custom base directory and path is provided,
        # update and overwrite the path "flex_base_directory_path" in flex_config
        # By default: current working directory is used as base
        if generated_flex_files_base_path is not None:
            if isinstance(flex_config, (str, Path)):
                with open(flex_config, "r") as f:
                    flex_config = json.load(f)
            flex_config["flex_base_directory_path"] = str(generated_flex_files_base_path)
        # load configs of agents and modules
        # Generator config
        self.generator_config = load_config.load_config(
            config=flex_config, config_type=FlexQuantConfig
        )

        # get names of the config files
        config_filename_baseline = BaselineMPCData.model_validate(
            self.generator_config.baseline_config_generator_data
        ).name_of_created_file
        config_filename_pos_flex = PFMPCData.model_validate(
            self.generator_config.shadow_mpc_config_generator_data.pos_flex
        ).name_of_created_file
        config_filename_neg_flex = NFMPCData.model_validate(
            self.generator_config.shadow_mpc_config_generator_data.neg_flex
        ).name_of_created_file
        config_filename_indicator = self.generator_config.indicator_config.name_of_created_file
        if self.generator_config.market_config:
            if self.generator_config.market_config is str or Path:
                config_filename_market = FlexibilityMarketConfig.parse_file(
                    self.generator_config.market_config
                ).name_of_created_file
            else:
                config_filename_market = FlexibilityMarketConfig.model_validate(
                    self.generator_config.market_config
                ).name_of_created_file

        # load the agent and module configs
        if simulator_agent_config:
            # (don't validate config, as result file is deleted in simulator validator)
            # check config type: with results path adaptation -> dict; without -> str/Path
            if isinstance(simulator_agent_config, (str, Path)):
                with open(simulator_agent_config, "r") as f:
                    sim_config = json.load(f)
            elif isinstance(simulator_agent_config, dict):
                sim_config = simulator_agent_config
            self.simulator_agent_config = AgentConfig.construct(**sim_config)
            for module in self.simulator_agent_config.modules:
                if module["type"] == "simulator":
                    self.simulator_module_config = SimulatorConfig.construct(**module)
            if not self.simulator_module_config:
                raise ValueError("No simulator module in provided simulator config")
        else:
             self.simulator_agent_config = None

        for file_path in Path(self.generator_config.flex_files_directory).rglob("*.json"):
            if file_path.name in config_filename_baseline:
                self.baseline_agent_config = load_config.load_config(
                    config=file_path, config_type=AgentConfig
                )
                self.baseline_module_config = cmng.get_module(
                    config=self.baseline_agent_config,
                    module_type=cmng.BASELINEMPC_CONFIG_TYPE,
                )

            elif file_path.name in config_filename_pos_flex:
                self.pos_flex_agent_config = load_config.load_config(
                    config=file_path, config_type=AgentConfig
                )
                self.pos_flex_module_config = cmng.get_module(
                    config=self.pos_flex_agent_config,
                    module_type=cmng.SHADOWMPC_CONFIG_TYPE,
                )

            elif file_path.name in config_filename_neg_flex:
                self.neg_flex_agent_config = load_config.load_config(
                    config=file_path, config_type=AgentConfig
                )
                self.neg_flex_module_config = cmng.get_module(
                    config=self.neg_flex_agent_config,
                    module_type=cmng.SHADOWMPC_CONFIG_TYPE,
                )

            elif file_path.name in config_filename_indicator:
                self.indicator_agent_config = load_config.load_config(
                    config=file_path, config_type=AgentConfig
                )
                self.indicator_module_config = cmng.get_module(
                    config=self.indicator_agent_config,
                    module_type=cmng.INDICATOR_CONFIG_TYPE,
                )

            elif (
                self.generator_config.market_config
                and file_path.name in config_filename_market
            ):
                self.market_agent_config = load_config.load_config(
                    config=file_path, config_type=AgentConfig
                )
                self.market_module_config = cmng.get_module(
                    config=self.market_agent_config, module_type=cmng.MARKET_CONFIG_TYPE
                )

        # load results
        if results is None:
            results_path = self.generator_config.results_directory
            results = self._load_results(res_path=results_path)
        if isinstance(results, (str, Path)):
            results_path = results
            results = self._load_results(res_path=results_path)
        elif isinstance(results, dict):
            results_path = self.generator_config.results_directory
        else:
            raise ValueError("results must be a path or dict")

        # Get result dataframes
        if simulator_agent_config:
            self.df_simulation = results[self.simulator_agent_config.id][
                self.simulator_module_config.module_id
            ]
        self.df_baseline = results[self.baseline_agent_config.id][
            self.baseline_module_config.module_id
        ]
        self.df_pos_flex = results[self.pos_flex_agent_config.id][
            self.pos_flex_module_config.module_id
        ]
        self.df_neg_flex = results[self.neg_flex_agent_config.id][
            self.neg_flex_module_config.module_id
        ]
        self.df_indicator = results[self.indicator_agent_config.id][
            self.indicator_module_config.module_id
        ]
        if self.generator_config.market_config:
            self.df_market = results[self.market_agent_config.id][
                self.market_module_config.module_id
            ]
        else:
            self.df_market = None

        # Load the statistics
        self.df_baseline_stats = load_mpc_stats(
            Path(
                results_path,
                Path(
                    self.baseline_module_config.optimization_backend["results_file"]
                ).name,
            )
        )
        self.df_pos_flex_stats = load_mpc_stats(
            Path(
                results_path,
                Path(
                    self.pos_flex_module_config.optimization_backend["results_file"]
                ).name,
            )
        )
        self.df_neg_flex_stats = load_mpc_stats(
            Path(
                results_path,
                Path(
                    self.neg_flex_module_config.optimization_backend["results_file"]
                ).name,
            )
        )

        # Convert the time in the dataframes to the desired timescale
        self.convert_timescale_of_dataframe_index(to_timescale=to_timescale)

    def _load_results(self, res_path: Union[str, Path]) -> dict[str, dict[str, pd.DataFrame]]:
        """
        read the results file depending on the given res_path
        Args:
            res_path: path to the results file
        Returns:
            results: a dictionary containing all the results of the experiment
        """
        res = {
            self.baseline_agent_config.id: {
                self.baseline_module_config.module_id: load_mpc(
                    Path(
                        res_path,
                        Path(
                            self.baseline_module_config.optimization_backend[
                                "results_file"
                            ]
                        ).name,
                    )
                )
            },
            self.pos_flex_agent_config.id: {
                self.pos_flex_module_config.module_id: load_mpc(
                    Path(
                        res_path,
                        Path(
                            self.pos_flex_module_config.optimization_backend[
                                "results_file"
                            ]
                        ).name,
                    )
                )
            },
            self.neg_flex_agent_config.id: {
                self.neg_flex_module_config.module_id: load_mpc(
                    Path(
                        res_path,
                        Path(
                            self.neg_flex_module_config.optimization_backend[
                                "results_file"
                            ]
                        ).name,
                    )
                )
            },
            self.indicator_agent_config.id: {
                self.indicator_module_config.module_id: load_indicator_results(
                    Path(
                        res_path, 
                        Path(self.indicator_module_config.results_file).name,
                    )
                )
            }
        }
        if self.simulator_agent_config:
            res[self.simulator_agent_config.id] = {
                self.simulator_module_config.module_id: load_sim(
                    Path(
                        res_path,
                        Path(self.simulator_module_config.result_filename).name,
                    )
                )
            }
        if self.generator_config.market_config:
            res[self.market_agent_config.id] = {
                self.market_module_config.module_id: load_market_results(
                    Path(
                        res_path, 
                        Path(self.market_module_config.results_file).name,
                    )
                )
            }
        return res

    def convert_timescale_of_dataframe_index(self, to_timescale: TimeConversionTypes):
        """
        Convert the time in the dataframes to the desired timescale
        Args:
            timescale: The timescale to convert the data to
        """
        # Convert the time in the dataframes
        for df in ([
            self.df_baseline,
            self.df_baseline_stats,
            self.df_pos_flex,
            self.df_pos_flex_stats,
            self.df_neg_flex,
            self.df_neg_flex_stats,
            self.df_indicator,
        ] + ([self.df_market] if self.generator_config.market_config else []) +
                   ([self.df_simulation] if self.simulator_agent_config else [])):
            convert_timescale_of_index(
                df=df, from_unit=self.current_timescale_of_data, to_unit=to_timescale
            )

        # Update current unit
        self.current_timescale_of_data = to_timescale

    def get_intersection_mpcs_sim(self) -> dict[str, dict[str, str]]:
        """
        Get the intersection of the MPCs and the simulator variables.
        Returns:
             dictionary with the following structure: Key: variable alias (from baseline)
                                                    Value: {module id: variable name}
        """
        id_alias_name_dict = {}

        def get_id_alias_name_dict_element(alias: str):
            # id as key, {id: name} as value
            id_alias_name_dict[alias] = {}
            for config in [
                self.simulator_module_config,
                self.baseline_module_config,
                self.pos_flex_module_config,
                self.neg_flex_module_config,
            ]:
                for var in config.get_variables():
                    if var.alias == alias or var.name == alias:
                        id_alias_name_dict[alias][config.module_id] = var.name

        # States, controls and power variable
        for variables in [
            self.baseline_module_config.states,
            self.baseline_module_config.controls,
        ]:
            for variable in variables:
                get_id_alias_name_dict_element(variable.alias)
        get_id_alias_name_dict_element(
            self.generator_config.baseline_config_generator_data.power_variable
        )

        return id_alias_name_dict
