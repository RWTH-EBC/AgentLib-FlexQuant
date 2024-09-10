from agentlib.core.agent import AgentConfig
from agentlib.core.module import BaseModuleConfig
import flexibility_quantification.data_structures.module_mapping as mmap
from typing import TypeVar
import math

T = TypeVar('T', bound=BaseModuleConfig)


def get_module(config: AgentConfig, module_type: str) -> T:
    """Extracts a module from a config based on its name

    """
    for module in config.modules:
        if module["type"] == module_type:
            return mmap.MODULE_TYPES[module["type"]](**module, _agent_id=config.id)


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
