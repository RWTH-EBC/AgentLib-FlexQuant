"""
Data models for MPC configurations in flexibility quantification.

This module defines Pydantic data models that encapsulate configuration parameters
for baseline, positive flexibility, and negative flexibility MPC controllers used
in flexquant. The models handle file paths, module configurations, variable
mappings, and optimization weights for MPC implementations.
"""
import pydantic
from agentlib_mpc.data_structures.mpc_datamodels import MPCVariable
from pydantic import ConfigDict, model_validator, field_serializer, Field

import agentlib_flexquant.data_structures.globals as glbs
import agentlib_flexquant.utils.config_management as cmng

excluded_fields = [
        "rdf_class",
        "source",
        "type",
        "timestamp",
        "description",
        "unit",
        "clip",
        "interpolation_method",
        "allowed_values",
    ]


class BaseMPCData(pydantic.BaseModel):
    """Base class containing necessary data for the code creation of the different mpcs"""

    # files and paths
    created_flex_mpcs_file: str = "flex_agents.py"
    name_of_created_file: str
    results_suffix: str
    # modules
    module_types: dict
    class_name: str
    module_id: str
    agent_id: str
    # variables
    power_alias: str
    stored_energy_alias: str
    config_inputs_appendix: list[MPCVariable] = Field(default=[], description="Inputs, which are appended to the MPCs' config (.json file and ConfigClass).")
    config_parameters_appendix: list[MPCVariable] = Field(default=[], description="Parameters, which are appended to the MPCs' config (.json file and ConfigClass).")
    
    @field_serializer('config_inputs_appendix', 'config_parameters_appendix')
    def serialize_mpc_variables(self, variables: list[MPCVariable], _info):
        return [v.dict(exclude=excluded_fields) for v in variables]


class BaselineMPCData(BaseMPCData):
    """Data class for Baseline MPC"""

    # files and paths
    results_suffix: str = "_base.csv"
    name_of_created_file: str = "baseline.json"
    # modules
    module_types: dict = cmng.BASELINE_MODULE_TYPE_DICT
    class_name: str = "BaselineMPCModel"
    module_id: str = "Baseline"
    agent_id: str = "Baseline"
    # variables
    power_alias: str = glbs.POWER_ALIAS_BASE
    stored_energy_alias: str = glbs.STORED_ENERGY_ALIAS_BASE
    power_variable: str = pydantic.Field(
        default="P_el",
        description=(
            "Name of the variable representing the electrical power in the baseline config"
        ),
    )
    profile_deviation_weight: float = pydantic.Field(
        default=0,
        description="Weight of soft constraint for deviation from accepted flexible profile",
    )
    power_unit: str = pydantic.Field(
        default="kW", description="Unit of the power variable"
    )
    comfort_variable: str = pydantic.Field(
        default=None,
        description=(
            "Name of the slack variable representing the thermal comfort in the baseline config"
        ),
    )
    profile_comfort_weight: float = pydantic.Field(
        default=1, description="Weight of soft constraint for discomfort",
    )
    config_inputs_appendix: list[MPCVariable] = [
        MPCVariable(name=glbs.ACCEPTED_POWER_VAR_NAME, value=0, unit="W"),
        MPCVariable(name=glbs.PROVISION_VAR_NAME, value=False),
        MPCVariable(name=glbs.RELATIVE_EVENT_START_TIME_VAR_NAME, value=0, unit="s"),
        MPCVariable(name=glbs.RELATIVE_EVENT_END_TIME_VAR_NAME, value=0, unit="s"),
    ]

    config_parameters_appendix: list[MPCVariable] = []


    @field_serializer('config_inputs_appendix', 'config_parameters_appendix')
    def serialize_mpc_variables(self, variables: list[MPCVariable], _info):
        return [v.dict(exclude=excluded_fields) for v in variables]

    @model_validator(mode="after")
    def update_config_parameters_appendix(self) -> "BaselineMPCData":
        """Update the config parameters appendix with profile deviation and comfort weights.

        Adds the profile deviation weight parameter and optionally the profile comfort
        weight parameter (if comfort_variable is enabled) to the config_parameters_appendix
        list as MPCVariable instances.
        """
        self.config_parameters_appendix = [
            MPCVariable(
                name=glbs.PROFILE_DEVIATION_WEIGHT,
                value=self.profile_deviation_weight,
                unit="-",
                description=(
                    "Weight of soft constraint for deviation from accepted flexible profile"
                ),
            )
        ]
        if self.comfort_variable:
            self.config_parameters_appendix.append(
                MPCVariable(
                    name=glbs.PROFILE_COMFORT_WEIGHT,
                    value=self.profile_comfort_weight,
                    unit="-",
                    description="Weight of soft constraint for discomfort",
                )
            )
        return self


class PFMPCData(BaseMPCData):
    """Data class for PF-MPC"""

    # files and paths
    results_suffix: str = "_pos_flex.csv"
    name_of_created_file: str = "pos_flex.json"
    # modules
    module_types: dict = cmng.SHADOW_MODULE_TYPE_DICT
    class_name: str = "PosFlexModel"
    module_id: str = "PosFlexMPC"
    agent_id: str = "PosFlexMPC"
    # variables
    power_alias: str = glbs.POWER_ALIAS_POS
    stored_energy_alias: str = glbs.STORED_ENERGY_ALIAS_POS
    flex_cost_function: str = pydantic.Field(
        default=None, description="Cost function of the PF-MPC",
    )
    # initialize market parameters with dummy values (0)
    config_parameters_appendix: list[MPCVariable] = pydantic.Field(
        default=[], description = "Parameters, which need to be appended to the shadow MPCs"
    )
    config_inputs_appendix: list[MPCVariable] = pydantic.Field(
        default=[], description = "Inputs, which need to be appended to the shadow MPCs"
    )
        
    
    @field_serializer('config_inputs_appendix', 'config_parameters_appendix')
    def serialize_mpc_variables(self, variables: list[MPCVariable], _info):
        return [v.dict(exclude=excluded_fields) for v in variables]

    @model_validator(mode="before")
    @classmethod 
    def add_default_inputs_to_appendix(cls, data): 
        """
        Ensures that all required framework parameters are included in 
        `config_inputs_appendix`. If any default framework parameter 
        (e.g., PROVISION_VAR_NAME) is missing, 
        it appends them to the list. 
        """
        default_inputs = [
        MPCVariable(name=glbs.PROVISION_VAR_NAME, value=False),
        ]

        # Get the provided config_inputs_appendix or use an empty list
        provided_inputs = data.get("config_inputs_appendix", [])
        
        # Ensure all default parameters are included
        provided_names = {param.name for param in provided_inputs}
        for default_input in default_inputs:
            if default_input.name not in provided_names:
                provided_inputs.append(default_input)
        
        # Update the data with the complete list of parameters
        data["config_inputs_appendix"] = provided_inputs
        return data 

    @model_validator(mode="before")
    @classmethod 
    def add_default_parameters_to_appendix(cls, data): 
        """
        Ensures that all required framework parameters are included in 
        `config_parameters_appendix`. If any default framework parameter 
        (e.g., PREP_TIME, MARKET_TIME, FLEX_EVENT_DURATION) is missing, 
        it appends them to the list. 
        """
        default_parameters = [
        MPCVariable(name=glbs.PREP_TIME, value=0, unit="s"),
        MPCVariable(name=glbs.MARKET_TIME, value=0, unit="s"),
        MPCVariable(name=glbs.FLEX_EVENT_DURATION, value=0, unit="s")
        ]

        # Get the provided config_parameters_appendix or use an empty list
        provided_parameters = data.get("config_parameters_appendix", [])
        
        # Ensure all default parameters are included
        provided_names = {param.name for param in provided_parameters}
        for default_param in default_parameters:
            if default_param.name not in provided_names:
                provided_parameters.append(default_param)
        
        # Update the data with the complete list of parameters
        data["config_parameters_appendix"] = provided_parameters
        return data 
    
class NFMPCData(BaseMPCData):
    """Data class for PF-MPC"""

    # files and paths
    results_suffix: str = "_neg_flex.csv"
    name_of_created_file: str = "neg_flex.json"
    # modules
    module_types: dict = cmng.SHADOW_MODULE_TYPE_DICT
    class_name: str = "NegFlexModel"
    module_id: str = "NegFlexMPC"
    agent_id: str = "NegFlexMPC"
    # variables
    power_alias: str = glbs.POWER_ALIAS_NEG
    stored_energy_alias: str = glbs.STORED_ENERGY_ALIAS_NEG
    flex_cost_function: str = pydantic.Field(
        default=None, description="Cost function of the NF-MPC",
    )
    # initialize market parameters with dummy values (0)
    config_parameters_appendix: list[MPCVariable] = pydantic.Field(
        default=[], description="Parameters, which need to be appended to the shadow MPCs"
    )
    config_inputs_appendix: list[MPCVariable] = pydantic.Field(
        default=[], description = "Inputs, which need to be appended to the shadow MPCs" 
    )


    @field_serializer('config_inputs_appendix', 'config_parameters_appendix')
    def serialize_mpc_variables(self, variables: list[MPCVariable], _info):
        return [v.dict(exclude=excluded_fields) for v in variables]
    
    @model_validator(mode="before")
    @classmethod 
    def add_default_inputs_to_appendix(cls, data): 
        """
        Ensures that all required framework parameters are included in 
        `config_inputs_appendix`. If any default framework parameter 
        (e.g., PROVISION_VAR_NAME) is missing, 
        it appends them to the list. 
        """
        default_inputs = [
        MPCVariable(name=glbs.PROVISION_VAR_NAME, value=False),
        ]

        # Get the provided config_inputs_appendix or use an empty list
        provided_inputs = data.get("config_inputs_appendix", [])
        
        # Ensure all default parameters are included
        provided_names = {param.name for param in provided_inputs}
        for default_input in default_inputs:
            if default_input.name not in provided_names:
                provided_inputs.append(default_input)
        
        # Update the data with the complete list of parameters
        data["config_inputs_appendix"] = provided_inputs
        return data 

    @model_validator(mode="before")
    @classmethod 
    def add_default_parameters_to_appendix(cls, data): 
        """
        Ensures that all required framework parameters are included in 
        `config_parameters_appendix`. If any default framework parameter 
        (e.g., PREP_TIME, MARKET_TIME, FLEX_EVENT_DURATION) is missing, 
        it appends them to the list. 
        """
        default_parameters = [
        MPCVariable(name=glbs.PREP_TIME, value=0, unit="s"),
        MPCVariable(name=glbs.MARKET_TIME, value=0, unit="s"),
        MPCVariable(name=glbs.FLEX_EVENT_DURATION, value=0, unit="s")
        ]

        # Get the provided config_parameters_appendix or use an empty list
        provided_parameters = data.get("config_parameters_appendix", [])
        
        # Ensure all default parameters are included
        provided_names = {param.name for param in provided_parameters}
        for default_param in default_parameters:
            if default_param.name not in provided_names:
                provided_parameters.append(default_param)
        
        # Update the data with the complete list of parameters
        data["config_parameters_appendix"] = provided_parameters
        return data 