from agentlib.core.agent import AgentConfig
from agentlib.core.module import BaseModuleConfig
from agentlib.utils.load_config import load_config

from pathlib import Path
from copy import deepcopy
from typing import TypeVar, Union
import math
from agentlib.modules import get_all_module_types
import inspect


T = TypeVar('T', bound=BaseModuleConfig)


all_module_types = get_all_module_types(["agentlib_mpc", "flexibility_quantification"])
# remove ML models, since import takes ages
all_module_types.pop("agentlib_mpc.ann_trainer")
all_module_types.pop("agentlib_mpc.gpr_trainer")
all_module_types.pop("agentlib_mpc.linreg_trainer")
all_module_types.pop("agentlib_mpc.ann_simulator")
all_module_types.pop("agentlib_mpc.set_point_generator")
# remove clone since not used
all_module_types.pop("clonemap")

MODULE_TYPE_DICT = {name: inspect.get_annotations(class_type.import_class())["config"] for name, class_type in all_module_types.items()}

MPC_CONFIG_TYPE: str = "agentlib_mpc.mpc"
BASELINEMPC_CONFIG_TYPE: str = "flexibility_quantification.baseline_mpc"
SHADOWMPC_CONFIG_TYPE: str = "flexibility_quantification.shadow_mpc"
INDICATOR_CONFIG_TYPE: str = "flexibility_quantification.flexibility_indicator"
MARKET_CONFIG_TYPE: str = "flexibility_quantification.flexibility_market"
SIMULATOR_CONFIG_TYPE: str = "simulator"

# keys for accessing the agent configs and results dictionary
SIMULATOR_AGENT_KEY: str = "simulator"
BASELINE_AGENT_KEY: str = "baseline"
NEG_FLEX_AGENT_KEY: str = "neg_flex"
POS_FLEX_AGENT_KEY: str = "pos_flex"
INDICATOR_AGENT_KEY: str = "indicator"
FLEX_MARKET_AGENT_KEY: str = "flexibility_market"


def load_agent_configs_of_flexquant_simulation(file_paths: Union[list[str], list[Path]]) -> dict[str, AgentConfig]:
    """
    Load the agent configurations from the given file paths
    return a dictionary with the agent type as key and the agent config as value
    """
    agent_configs = {}
    for file_path in file_paths:
        agent = load_config(file_path, AgentConfig)
        for module in agent.modules:
            if module["type"] == BASELINEMPC_CONFIG_TYPE:
                agent_configs[BASELINE_AGENT_KEY] = agent
            elif module["type"] == SHADOWMPC_CONFIG_TYPE:
                if "neg".casefold() in module["module_id"].casefold():
                    agent_configs[NEG_FLEX_AGENT_KEY] = agent
                elif "pos".casefold() in module["module_id"].casefold():
                    agent_configs[POS_FLEX_AGENT_KEY] = agent
            elif module["type"] == INDICATOR_CONFIG_TYPE:
                agent_configs[INDICATOR_AGENT_KEY] = agent
            elif module["type"] == MARKET_CONFIG_TYPE:
                agent_configs[FLEX_MARKET_AGENT_KEY] = agent
            elif module["type"] == SIMULATOR_CONFIG_TYPE:
                agent_configs[SIMULATOR_AGENT_KEY] = agent
    return agent_configs


def get_module(config: AgentConfig, module_type: str) -> T:
    """Extracts a module from a config based on its name

    """
    for module in config.modules:
        if module["type"] == module_type:
            return deepcopy(MODULE_TYPE_DICT[module["type"]](**module, _agent_id=config.id))


def to_dict_and_remove_unnecessary_fields(module: BaseModuleConfig):
    """Removes unnecessary fields from the module to keep the created json simple

    """
    excluded_fields = ["rdf_class", "source", "type", "timestamp", "description", "unit", "clip",
                       "shared", "interpolation_method", "allowed_values"]

    def check_bounds(parameter):
        delete_list = excluded_fields.copy()
        if parameter.lb == -math.inf:
            delete_list.append("lb")
        if parameter.ub == math.inf:
            delete_list.append("ub")
        return delete_list

    parent_dict = module.dict(exclude_defaults=True)
    # update every variable with a dict excluding the defined fields
    if "parameters" in parent_dict:
        parent_dict["parameters"] = [parameter.dict(exclude=check_bounds(parameter)) for parameter in module.parameters]
    if "inputs" in parent_dict:
        parent_dict["inputs"] = [input.dict(exclude=check_bounds(input)) for input in module.inputs]
    if "outputs" in parent_dict:
        parent_dict["outputs"] = [output.dict(exclude=check_bounds(output)) for output in module.outputs]
    if "controls" in parent_dict:
        parent_dict["controls"] = [control.dict(exclude=check_bounds(control)) for control in module.controls]
    if "states" in parent_dict:
        parent_dict["states"] = [state.dict(exclude=check_bounds(state)) for state in module.states]

    return parent_dict
