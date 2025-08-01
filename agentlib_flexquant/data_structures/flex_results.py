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
        # Obtain mapping to automate loading later
        self._get_config_specs()
        # Load configs for mpc, indicator, market
        self._load_agent_module_configs()
        # Load sim configs if present
        if simulator_agent_config:
            self._load_simulator_config(simulator_agent_config)
        # Get dataframes for mpc, sim, flex indicator results
        self._load_results_dataframes(results)
        # Get dataframes for mpc stats
        self._load_stats_dataframes()
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
    
    def _get_config_specs(self):
        """
        This method creates a mapping between flexquant components and related
        config metadata for automated downstream processing. This method handles
        core configs like baseline, pos_flex, neg_flex, and indicator. Market config
        is conditionally added if present.
        """        
        self.config_specs = {
            "baseline": {
                "filename": BaselineMPCData.model_validate(
                    self.generator_config.baseline_config_generator_data
                    ).name_of_created_file,
                "agent_attr": "baseline_agent_config",
                "module_attr": "baseline_module_config",
                "module_type": cmng.BASELINEMPC_CONFIG_TYPE,
                "results_loader": load_mpc,
            },
            "pos_flex": {
                "filename": PFMPCData.model_validate(
                    self.generator_config.shadow_mpc_config_generator_data.pos_flex
                    ).name_of_created_file,
                "agent_attr": "pos_flex_agent_config",
                "module_attr": "pos_flex_module_config",
                "module_type": cmng.SHADOWMPC_CONFIG_TYPE,
                "results_loader": load_mpc,
            },
            "neg_flex": {
                "filename": NFMPCData.model_validate(
                    self.generator_config.shadow_mpc_config_generator_data.neg_flex
                    ).name_of_created_file,
                "agent_attr": "neg_flex_agent_config",
                "module_attr": "neg_flex_module_config",
                "module_type": cmng.SHADOWMPC_CONFIG_TYPE,
                "results_loader": load_mpc,
            },
            "indicator": {
                "filename": self.generator_config.indicator_config.name_of_created_file,
                "agent_attr": "indicator_agent_config",
                "module_attr": "indicator_module_config",
                "module_type": cmng.INDICATOR_CONFIG_TYPE,
                "results_loader": load_indicator,
            },
        }

        # Conditionally add market config if it exists
        if self.generator_config.market_config:
            market_config_raw = self.generator_config.market_config
            market_config_model = (
                FlexibilityMarketConfig.model_validate_json(Path(market_config_raw).read_text())
                if isinstance(market_config_raw, (str, Path))
                else FlexibilityMarketConfig.model_validate(market_config_raw)
            )
            self.config_specs["market"] = {
                "filename": market_config_model.name_of_created_file,
                "agent_attr": "market_agent_config",
                "module_attr": "market_module_config",
                "module_type": cmng.MARKET_CONFIG_TYPE,
                "results_loader": load_market,
            }

    def _load_agent_module_configs(self):
        """
        Load agent and module configs for components present in self.config_specs.
        """
        for file_path in Path(self.generator_config.flex_files_directory).rglob("*.json"):
            for key, spec in self.config_specs.items():
                if file_path.name == spec["filename"]:
                    agent_config = load_config.load_config(
                        config=file_path, config_type=AgentConfig)
                    module_config = cmng.get_module(
                        config=agent_config, module_type=spec["module_type"])
                    setattr(self, spec["agent_attr"], agent_config)
                    setattr(self, spec["module_attr"], module_config)
                    break

    def _load_simulator_config(self, simulator_agent_config):
        """
        Load simulator agent and module config separately and add metadata to self.config_specs.
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
        # add metadata to self.config_specs
        self.config_specs["simulation"] = {
                "agent_attr": "simulator_agent_config",
                "module_attr": "simulator_module_config",
                "results_loader": load_sim,
            }

    def _load_results_dataframes(self, results):
        """
        Load results dataframes for mpc, indicator, market and sim.
        """
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
        for key, spec in self.config_specs.items():
            agent = getattr(self, spec["agent_attr"], None)
            module = getattr(self, spec["module_attr"], None)
            if agent is not None and module is not None:
                setattr(self, f"df_{key}", results[agent.id][module.module_id])

    def _load_stats_dataframes(self):
        """
        Load dataframes for mpc stats.
        """
        results_path = self.generator_config.results_directory
        for key in ["baseline", "pos_flex", "neg_flex"]:
            spec = self.config_specs.get(key)
            module = getattr(self, spec["module_attr"], None)
            stats_file = Path(module.optimization_backend["results_file"]).name
            df_stats = load_mpc_stats(results_path / stats_file)
            setattr(self, f"df_{key}_stats", df_stats)

    def _load_results(
        self, res_path: Union[str, Path]
    ) -> dict[str, dict[str, pd.DataFrame]]:
        """
        Load dict with results for mpc, indicator, market and sim from specified results path.
        """
        res: dict[str, dict[str, pd.DataFrame]] = {}

        for key, spec in self.config_specs.items():
            agent = getattr(self, spec["agent_attr"], None)
            module = getattr(self, spec["module_attr"], None)
            loader = spec.get("results_loader")
            if hasattr(module, "optimization_backend"):
                filename = Path(module.optimization_backend["results_file"]).name
            elif hasattr(module, "results_file"):
                filename = Path(module.results_file).name
            elif hasattr(module, "result_filename"):
                filename = Path(module.result_filename).name
            file_path = res_path / filename
            res.setdefault(agent.id, {})[module.module_id] = loader(file_path) 

        return res

    def convert_timescale_of_dataframe_index(self, to_timescale: TimeConversionTypes):
        """Convert the time in the dataframes to the desired timescale

        Keyword arguments:
        timescale -- The timescale to convert the data to
        """
        # Convert the time in the dataframes
        for key in self.config_specs:
            for suffix in ["", "_stats"]:
                attr_name = f"df_{key}{suffix}"
                df = getattr(self, attr_name, None)
                if df is not None:
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