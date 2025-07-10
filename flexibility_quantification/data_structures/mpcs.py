import pydantic
from pydantic import ConfigDict, model_validator
from typing import List, Optional
from agentlib_mpc.data_structures.mpc_datamodels import MPCVariable
import flexibility_quantification.data_structures.globals as glbs
import flexibility_quantification.utils.config_management as cmng


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
    # variables
    power_alias: str
    stored_energy_alias: str
    config_inputs_appendix: List[MPCVariable] = []
    config_parameters_appendix: List[MPCVariable] = []
    weights: List[MPCVariable] = pydantic.Field(
        default=[],
        description="Name and value of weights",
    )
    model_config = ConfigDict(json_encoders={MPCVariable: lambda v: v.dict()})


class BaselineMPCData(BaseMPCData):
    """Data class for Baseline MPC"""

    # files and paths
    results_suffix: str = "_base.csv"
    name_of_created_file: str = "baseline.json"
    # modules
    module_types: dict = cmng.BASELINE_MODULE_TYPE_DICT
    class_name: str = "BaselineMPCModel"
    module_id: str = "Baseline"
    # variables
    power_alias: str = glbs.POWER_ALIAS_BASE
    stored_energy_alias: str = glbs.STORED_ENERGY_ALIAS_BASE
    power_variable: str = pydantic.Field(
        default="P_el",
        description="Name of the variable representing the electrical power in the baseline config",
    )
    profile_deviation_weight: float = pydantic.Field(
        default=0,
        description="Weight of soft constraint for deviation from accepted flexible profile",
    )
    power_unit: str = pydantic.Field(
        default="kW", 
        description="Unit of the power variable"
    )
    comfort_variable: str = pydantic.Field(
        default=None,
        description="Name of the slack variable representing the thermal comfort in the baseline config",
    )
    profile_comfort_weight: float = pydantic.Field(
        default=1,
        description="Weight of soft constraint for discomfort",
    )
    config_inputs_appendix: List[MPCVariable] = [
        MPCVariable(name="_P_external", value=0, unit="W"),
        MPCVariable(name="in_provision", value=False),
        MPCVariable(name="rel_start", value=0, unit="s"),
        MPCVariable(name="rel_end", value=0, unit="s"),
    ]

    config_parameters_appendix: List[MPCVariable] = []

    weights: List[MPCVariable] = pydantic.Field(
        default=[],
        description="Name and value of weights",
    )
    model_config = ConfigDict(json_encoders={MPCVariable: lambda v: v.dict()})

    @model_validator(mode='after')
    def update_config_parameters_appendix(self) -> 'BaselineMPCData':
        self.config_parameters_appendix = [
            MPCVariable(
                name=glbs.PROFILE_DEVIATION_WEIGHT,
                value=self.profile_deviation_weight,
                unit="-",
                description="Weight of soft constraint for deviation from accepted flexible profile"
            )
        ]
        if self.comfort_variable:
            self.config_parameters_appendix.append(MPCVariable(
                    name=glbs.PROFILE_COMFORT_WEIGHT,
                    value=self.profile_comfort_weight,
                    unit="-",
                    description="Weight of soft constraint for discomfort"
                ))
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
    # variables
    power_alias: str = glbs.POWER_ALIAS_POS
    stored_energy_alias: str = glbs.STORED_ENERGY_ALIAS_POS
    flex_cost_function: str = pydantic.Field(
        default=None,
        description="Cost function of the PF-MPC",
    )
    # initialize market parameters with dummy values (0)
    config_parameters_appendix: List[MPCVariable] = [
        MPCVariable(name=glbs.PREP_TIME, value=0, unit="s"),
        MPCVariable(name=glbs.MARKET_TIME, value=0, unit="s"),
        MPCVariable(name=glbs.FLEX_EVENT_DURATION, value=0, unit="s")
    ]
    config_inputs_appendix: List[MPCVariable] = [
        MPCVariable(name="in_provision", value=False),
    ]
    weights: List[MPCVariable] = pydantic.Field(
        default=[],
        description="Name and value of weights",
    )
    model_config = ConfigDict(json_encoders={MPCVariable: lambda v: v.dict()})


class NFMPCData(BaseMPCData):
    """Data class for PF-MPC"""

    # files and paths
    results_suffix: str = "_neg_flex.csv"
    name_of_created_file: str = "neg_flex.json"
    # modules
    module_types: dict = cmng.SHADOW_MODULE_TYPE_DICT
    class_name: str = "NegFlexModel"
    module_id: str = "NegFlexMPC"
    # variables
    power_alias: str = glbs.POWER_ALIAS_NEG
    stored_energy_alias: str = glbs.STORED_ENERGY_ALIAS_NEG
    flex_cost_function: str = pydantic.Field(
        default=None,
        description="Cost function of the NF-MPC",
    )
    # initialize market parameters with dummy values (0)
    config_parameters_appendix: List[MPCVariable] = [
        MPCVariable(name=glbs.PREP_TIME, value=0, unit="s"),
        MPCVariable(name=glbs.MARKET_TIME, value=0, unit="s"),
        MPCVariable(name=glbs.FLEX_EVENT_DURATION, value=0, unit="s")
    ]
    config_inputs_appendix: List[MPCVariable] = [
        MPCVariable(name="in_provision", value=False),
    ]
    weights: List[MPCVariable] = pydantic.Field(
        default=[],
        description="Name and value of weights",
    )
    model_config = ConfigDict(json_encoders={MPCVariable: lambda v: v.dict()})
