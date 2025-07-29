from agentlib.core.agent import AgentConfig
from agentlib.core.module import BaseModuleConfig
import agentlib_flexquant.data_structures.globals as glbs
from copy import deepcopy
from typing import TypeVar
import math
from agentlib.modules import get_all_module_types
import inspect
import os
import importlib.util


T = TypeVar('T', bound=BaseModuleConfig)


all_module_types = get_all_module_types(["agentlib_mpc", "agentlib_flexquant"])
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
BASELINEMPC_CONFIG_TYPE: str = "agentlib_flexquant.baseline_mpc"
SHADOWMPC_CONFIG_TYPE: str = "agentlib_flexquant.shadow_mpc"
INDICATOR_CONFIG_TYPE: str = "agentlib_flexquant.flexibility_indicator"
MARKET_CONFIG_TYPE: str = "agentlib_flexquant.flexibility_market"
SIMULATOR_CONFIG_TYPE: str = "simulator"


def get_module_type_matching_dict(dictionary: dict):
    """Creates two dictionaries, which map the modules types of the agentlib_mpc modules
        to those of the flexquant modules. This is done by using the MODULE_TYPE_DICT

    """
    # Create dictionaries to store keys grouped by values
    value_to_keys = {}
    for k, v in dictionary.items():
        if k.startswith(('agentlib_mpc.', 'agentlib_flexquant.')):
            if v not in value_to_keys:
                value_to_keys[v] = {'agentlib': [], 'flex': []}
            if k.startswith('agentlib_mpc.'):
                value_to_keys[v]['agentlib'].append(k)
            else:
                value_to_keys[v]['flex'].append(k)

    # Create result dictionaries
    baseline_matches = {}
    shadow_matches = {}

    for v, keys in value_to_keys.items():
        # Check if we have both agentlib and flexibility keys for this value
        if keys['agentlib'] and keys['flex']:
            # Map each agentlib key to corresponding flexibility key
            for agent_key in keys['agentlib']:
                for flex_key in keys['flex']:
                    if 'baseline' in flex_key:
                        baseline_matches[agent_key] = flex_key
                    elif 'shadow' in flex_key:
                        shadow_matches[agent_key] = flex_key

    return baseline_matches, shadow_matches


BASELINE_MODULE_TYPE_DICT, SHADOW_MODULE_TYPE_DICT = (
    get_module_type_matching_dict(MODULE_TYPE_DICT))


def get_orig_module_type(config: AgentConfig):
    """Returns the config type of the original MPC

    """
    for module in config.modules:
        if module["type"].startswith("agentlib_mpc"):
            return module["type"]


def get_module(config: AgentConfig, module_type: str) -> T:
    """Extracts a module from a config based on its name

    """
    for module in config.modules:
        if module["type"] == module_type:
            # deepcopy -> avoid changing the original config, when editing the module
            # deepcopy the args of the constructor instead of the module object,
            # because the simulator module exceeds the recursion limit
            config_id = deepcopy(config.id)
            mod = deepcopy(module)
            return MODULE_TYPE_DICT[mod["type"]](**mod, _agent_id=config_id)
    else:
        raise ModuleNotFoundError(f"Module type {module['type']} not found in "
                                  f"agentlib and its plug ins.")



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


def subtract_relative_path(absolute_path, relative_path):
    # Normalize paths (convert slashes to the correct system format)
    absolute_path = os.path.normpath(absolute_path)
    relative_path = os.path.normpath(relative_path)

    # Split relative path to get the first component
    rel_parts = relative_path.split(os.sep)
    first_rel_component = rel_parts[0]

    # Find where the relative path starts in the absolute path
    if first_rel_component in absolute_path:
        # Find the last occurrence of the first component of the relative path
        index = absolute_path.rfind(first_rel_component)

        if index != -1:
            # Return the part of absolute_path before the relative path component
            return absolute_path[:index].rstrip(os.sep)

    # If the relative path component wasn't found, return the original path
    return absolute_path


def get_class_from_file(file_path, class_name):
    # Get the absolute path if needed
    abs_path = os.path.abspath(file_path)

    # Get the module name from the file path
    module_name = os.path.splitext(os.path.basename(file_path))[0]

    # Load the module specification
    spec = importlib.util.spec_from_file_location(module_name, abs_path)

    # Create the module
    module = importlib.util.module_from_spec(spec)

    # Execute the module
    spec.loader.exec_module(module)

    # Get the class from the module
    target_class = getattr(module, class_name)

    return target_class
