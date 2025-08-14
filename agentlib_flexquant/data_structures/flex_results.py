import copy
from typing import Union, Optional, Dict, Any, List, Type

import agentlib
from pydantic import FilePath, BaseModel
from pathlib import Path
import json
import os
import pandas as pd

from agentlib.core.agent import AgentConfig
from agentlib.core.module import BaseModuleConfig
from agentlib.utils import load_config
from agentlib_mpc.modules.mpc import BaseMPCConfig
from agentlib.modules.simulation.simulator import SimulatorConfig
from agentlib_flexquant.data_structures.flexquant import (
    FlexQuantConfig,
    FlexibilityIndicatorConfig,
    FlexibilityMarketConfig,
)
from agentlib_flexquant.data_structures.mpcs import (
    BaselineMPCData,
    NFMPCData,
    PFMPCData,
)
from agentlib_flexquant.utils.data_handling import convert_timescale_of_index
from agentlib_mpc.utils import TimeConversionTypes
from agentlib_mpc.utils.analysis import load_sim, load_mpc, load_mpc_stats

from agentlib_flexquant.modules.flexibility_indicator import (
    FlexibilityIndicatorModuleConfig,
)
from agentlib_flexquant.modules.flexibility_market import (
    FlexibilityMarketModuleConfig,
)
import agentlib_flexquant.utils.config_management as cmng


def load_indicator(file_path: Union[str, FilePath]) -> pd.DataFrame:
    """
    Load the flexibility indicator results from the given file path
    """
    df = pd.read_csv(file_path, header=0, index_col=[0, 1])
    return df


def load_market(file_path: Union[str, FilePath]) -> pd.DataFrame:
    """
    Load the market results from the given file path
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
        # Already a Results instance — copy over its data
        if isinstance(results, Results):
            self.__dict__ = copy.deepcopy(results).__dict__
            return
        
        # Load flex config
        self._load_flex_config(flex_config, generated_flex_files_base_path)
        # Get filenames of configs to load agents and modules
        self._get_config_filenames()
        # Load configs for mpc, indicator, market
        self._load_agent_module_configs()
        # Load sim configs if present
        if simulator_agent_config:
            self._load_simulator_config(simulator_agent_config)
        # Load results and get a dict for generating dataframes
        results_dict, results_path = self._load_results(results)
        # Get dataframes for mpc, sim, flex indicator results
        self._load_results_dataframes(results_dict)
        # Get dataframes for mpc stats
        self._load_stats_dataframes(results_path)
        # Convert the time in the dataframes to the desired timescale
        self.convert_timescale_of_dataframe_index(to_timescale=to_timescale)

    def _load_flex_config(self, flex_config, custom_base_path):
        """
        Load the flex config and optionally override the base directory path.
        If a custom base path is provided, it overwrites the "flex_base_directory_path"
        in the given config. This is useful when the generated flex files are saved
        to a custom directory instead of the default (current working directory).

        """
        if custom_base_path is not None:
            if isinstance(flex_config, (str, Path)):
                with open(flex_config, "r") as f:
                    flex_config = json.load(f)
            flex_config["flex_base_directory_path"] = str(custom_base_path)

        self.generator_config = load_config.load_config(
            config=flex_config, config_type=FlexQuantConfig)
    
    def _get_config_filenames(self):
        """
        Get filenames of configs to load agents and modules.
        """
        self.config_filename_baseline = BaselineMPCData.model_validate(
            self.generator_config.baseline_config_generator_data
        ).name_of_created_file
        self.config_filename_pos_flex = PFMPCData.model_validate(
            self.generator_config.shadow_mpc_config_generator_data.pos_flex
        ).name_of_created_file
        self.config_filename_neg_flex = NFMPCData.model_validate(
            self.generator_config.shadow_mpc_config_generator_data.neg_flex
        ).name_of_created_file
        self.config_filename_indicator = self.generator_config.indicator_config.name_of_created_file
        
        if self.generator_config.market_config:
            market_config_raw = self.generator_config.market_config
            if isinstance(market_config_raw, (str, Path)):
                market_config = FlexibilityMarketConfig.model_validate_json(
                    Path(market_config_raw).read_text()
                )
            else:
                market_config = FlexibilityMarketConfig.model_validate(market_config_raw)
            self.config_filename_market = market_config.name_of_created_file
        
    def _load_agent_module_configs(self):
        """
        Load agent and module configs.
        """
        for file_path in Path(self.generator_config.flex_files_directory).rglob("*.json"):
            if file_path.name in self.config_filename_baseline:
                self.baseline_agent_config = load_config.load_config(
                    config=file_path, config_type=AgentConfig
                )
                self.baseline_module_config = cmng.get_module(
                    config=self.baseline_agent_config,
                    module_type=self._get_module_type(self.baseline_agent_config),
                )

            elif file_path.name in self.config_filename_pos_flex:
                self.pos_flex_agent_config = load_config.load_config(
                    config=file_path, config_type=AgentConfig
                )
                self.pos_flex_module_config = cmng.get_module(
                    config=self.pos_flex_agent_config,
                    module_type=self._get_module_type(self.pos_flex_agent_config),
                )

            elif file_path.name in self.config_filename_neg_flex:
                self.neg_flex_agent_config = load_config.load_config(
                    config=file_path, config_type=AgentConfig
                )
                self.neg_flex_module_config = cmng.get_module(
                    config=self.neg_flex_agent_config,
                    module_type=self._get_module_type(self.neg_flex_agent_config),
                )

            elif file_path.name in self.config_filename_indicator:
                self.indicator_agent_config = load_config.load_config(
                    config=file_path, config_type=AgentConfig
                )
                self.indicator_module_config = cmng.get_module(
                    config=self.indicator_agent_config,
                    module_type=cmng.INDICATOR_CONFIG_TYPE,
                )

            elif (
                self.generator_config.market_config
                and file_path.name in self.config_filename_market
            ):
                self.market_agent_config = load_config.load_config(
                    config=file_path, config_type=AgentConfig
                )
                self.market_module_config = cmng.get_module(
                    config=self.market_agent_config, 
                    module_type=cmng.MARKET_CONFIG_TYPE
                )

    def _load_simulator_config(self, simulator_agent_config):
        """
        Load simulator agent and module config separately.
        Separate loading is required to skip pydantic validation for specific field(s).
        """
        # check config type: with results path adaptation -> dict; without -> str/Path
        if isinstance(simulator_agent_config, (str, Path)):
            with open(simulator_agent_config, "r") as f:
                sim_config = json.load(f)
        elif isinstance(simulator_agent_config, dict):
            sim_config = simulator_agent_config
        sim_module_config = next(
            (module for module in sim_config["modules"] if module["type"] == "simulator"),
            None
        )
        # instantiate and validate sim agent config
        self.simulator_agent_config = AgentConfig.model_validate(sim_config)
        # instantiate sim module config by skipping validation for result_filename 
        # to prevent file deletion
        self.simulator_module_config = self.create_instance_with_skipped_validation(
            model_class=SimulatorConfig, 
            config=sim_module_config, 
            skip_fields=["result_filename"]
        )

    def _get_module_type(self, agent_config: AgentConfig):
        for module in agent_config.modules:
            if module['type'] in [cmng.BASELINEMPC_CONFIG_TYPE, cmng.BASELINEMINLPMPC_CONFIG_TYPE,
                                  cmng.SHADOWMPC_CONFIG_TYPE, cmng.SHADOWMINLPMPC_CONFIG_TYPE]:
                return module['type']

    def _load_results(
        self, results: Union[str, Path, dict]
    ) -> dict[str, dict[str, pd.DataFrame]]:
        """
        Load dict with results for mpc, indicator, market and sim from specified results path.
        """
        # load results
        if results is None:
            res_path = self.generator_config.results_directory
        elif isinstance(results, (str, Path)):
            res_path = results
        elif isinstance(results, dict):
            res_path = self.generator_config.results_directory
        else:
            raise ValueError("results must be a path or dict")
        
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
                self.indicator_module_config.module_id: load_indicator(
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
                        Path(self.simulator_module_config.result_filename)
                )
            }
        if self.generator_config.market_config:
            res[self.market_agent_config.id] = {
                self.market_module_config.module_id: load_market(
                    Path(
                        res_path, 
                        Path(self.market_module_config.results_file).name,
                    )
                )
            }
        return res, res_path

    def _load_results_dataframes(self, results_dict):
        """
        Load results dataframes for mpc, indicator, market and sim.
        """        
        if self.simulator_agent_config:
            self.df_simulation = results_dict[self.simulator_agent_config.id][
                self.simulator_module_config.module_id
            ]
        self.df_baseline = results_dict[self.baseline_agent_config.id][
            self.baseline_module_config.module_id
        ]
        self.df_pos_flex = results_dict[self.pos_flex_agent_config.id][
            self.pos_flex_module_config.module_id
        ]
        self.df_neg_flex = results_dict[self.neg_flex_agent_config.id][
            self.neg_flex_module_config.module_id
        ]
        self.df_indicator = results_dict[self.indicator_agent_config.id][
            self.indicator_module_config.module_id
        ]
        if self.generator_config.market_config:
            self.df_market = results_dict[self.market_agent_config.id][
                self.market_module_config.module_id
            ]
        else:
            self.df_market = None

    def _load_stats_dataframes(self, results_path):
        """
        Load dataframes for mpc stats.
        """
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

    def convert_timescale_of_dataframe_index(self, to_timescale: TimeConversionTypes):
        """Convert the time in the dataframes to the desired timescale

        Keyword arguments:
        timescale -- The timescale to convert the data to
        """
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
        returns a dictionary with the following structure:
        Key: variable alias (from baseline)
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

    def create_instance_with_skipped_validation(
            self, 
            model_class: Type[BaseModel], 
            config: Dict[str, Any], 
            skip_fields: Optional[List[str]] = None
        ) -> BaseModel:
        """
        Create a Pydantic model instance while skipping validation for specified fields.

        This function allows partial validation of a model's config dictionary by validating 
        all fields except those listed in `skip_fields`. Skipped fields are set on the instance 
        after construction without triggering their validators.

        Args:
            model_class (Type[BaseModel]): The Pydantic model class to instantiate.
            config (Dict[str, Any]): The input configuration dictionary.
            skip_fields (Optional[List[str]]): A list of field names to exclude from validation. 
                                                These fields will be manually set after instantiation.

        Returns:
            BaseModel: An instance of the model_class with validated and skipped fields assigned.
        """
        if skip_fields is None:
            skip_fields = []
        # Separate data into validated and skipped fields
        validated_fields = {field: value for field, value in config.items() if field not in skip_fields}
        skipped_fields = {field: value for field, value in config.items() if field in skip_fields}
        # Create instance with validation for non-skipped fields
        if validated_fields:
            instance = model_class(
                **validated_fields, 
                _agent_id=self.simulator_agent_config.id
            )
        else:
            instance = model_class.model_construct()
        # Add skipped fields without validation
        for field, value in skipped_fields.items():
            # bypass pydantic immutability to directly set attribute value
            object.__setattr__(instance, field, value)
        # Store metadata about bypassed fields for deepcopy compatibility
        object.__setattr__(instance, '_bypassed_fields', skip_fields)
        object.__setattr__(instance, '_original_config', config)
        return instance
    
    def __deepcopy__(self, memo: Dict[int, Any]) -> "Results":
        """
        Custom deepcopy implementation that handles Pydantic models with bypassed validation.
        """
        # Create a new instance of the same class
        new_instance = self.__class__.__new__(self.__class__)
        # Add to memo immediately to prevent circular reference issues
        memo[id(self)] = new_instance
        for key, value in self.__dict__.items():
            if key in ['simulator_module_config'] and hasattr(value, '_original_config'):
                # Reconstruct the specific problematic object instead of deepcopying
                new_value = self.create_instance_with_skipped_validation(
                    model_class=value.__class__,
                    config=copy.deepcopy(value._original_config, memo),
                    skip_fields=getattr(value, '_bypassed_fields', [])
                )
                setattr(new_instance, key, new_value)
            else:
                # Everything else should deepcopy normally
                setattr(new_instance, key, copy.deepcopy(value, memo))
        return new_instance