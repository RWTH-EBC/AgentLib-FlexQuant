import pydantic
from typing import List
from agentlib_mpc.data_structures.mpc_datamodels import MPCVariables, MPCVariable
import flexibility_quantification.data_structures.globals as glbs
import flexibility_quantification.utils.config_management as cmng


#TODO: add validators for these
class BaseMPCData(pydantic.BaseModel):
    """Base class containing necessary data for the code creation of the different mpcs

    """
    # TODO: add Fields
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
    config_inputs_appendix: MPCVariables = []
    config_parameters_appendix: MPCVariables = []
    weights: List[MPCVariable] = pydantic.Field(
        default=[],
        description="Name and value of weights",
    )


class BaselineMPCData(BaseMPCData):
    """Data class for Baseline MPC

    """
    # files and paths
    results_suffix: str = "_base.csv"
    name_of_created_file: str = "baseline.json"
    # modules
    module_types: dict = cmng.BASELINE_MODULE_TYPE_DICT
    class_name: str = "BaselineMPCModel"
    module_id: str = "Baseline"
    # variables
    power_alias: str = glbs.POWER_ALIAS_BASE
    power_variable: str = pydantic.Field(
        default="P_el",
        description="Name of the variable representing the electrical power in the baseline config",
    )
    # TODO: add this as parameter to the mpc config rather than just writing the value in the cost function
    profile_deviation_weight: float = pydantic.Field(
        default=0,
        description="Weight of soft constraint for deviation from accepted flexible profile",
    )
    power_unit: str = pydantic.Field(
        default="kW",
        description="Unit of the power variable"
    )
    # TODO: wie mit diesen Daten umgehen? Vor Aufruf von adapt_mpc_module_config einmal diese Datenklasse initialisieren und Werte entsprechend setzen?
    config_inputs_appendix: MPCVariables = [
        MPCVariable(name="_P_external", value=0, unit="W"),
        MPCVariable(name="in_provision", value=False),
        MPCVariable(name="rel_start", value=0, unit="s"),
        MPCVariable(name="rel_end", value=0, unit="s")
    ]
    weights: List[MPCVariable] = pydantic.Field(
        default=[],
        description="Name and value of weights",
    )


class PFMPCData(BaseMPCData):
    """Data class for PF-MPC

    """
    # files and paths
    results_suffix: str = "_pos_flex.csv"
    name_of_created_file: str = "pos_flex.json"
    # modules
    module_types: dict = cmng.SHADOW_MODULE_TYPE_DICT
    class_name: str = "PosFlexModel"
    module_id: str = "PosFlexMPC"
    # variables
    power_alias: str = glbs.POWER_ALIAS_POS
    flex_cost_function: str = pydantic.Field(
        default=None,
        description="Cost function of the PF-MPC",
    )
    # initialize market parameters with dummy values (0)
    config_parameters_appendix: MPCVariables = [
        MPCVariable(name=glbs.PREP_TIME, value=0, unit="s"),
        MPCVariable(name=glbs.MARKET_TIME, value=0, unit="s"),
        MPCVariable(name=glbs.FLEX_EVENT_DURATION, value=0, unit="s"),
    ]
    config_inputs_appendix: MPCVariables = [
        MPCVariable(name="in_provision", value=False),
    ]
    weights: List[MPCVariable] = pydantic.Field(
        default=[],
        description="Name and value of weights",
    )


class NFMPCData(BaseMPCData):
    """Data class for PF-MPC

    """
    # files and paths
    results_suffix: str = "_neg_flex.csv"
    name_of_created_file: str = "neg_flex.json"
    # modules
    module_types: dict = cmng.SHADOW_MODULE_TYPE_DICT
    class_name: str = "NegFlexModel"
    module_id: str = "NegFlexMPC"
    # variables
    power_alias: str = glbs.POWER_ALIAS_NEG
    flex_cost_function: str = pydantic.Field(
        default=None,
        description="Cost function of the NF-MPC",
    )
    # initialize market parameters with dummy values (0)
    config_parameters_appendix: MPCVariables = [
        MPCVariable(name=glbs.PREP_TIME, value=0, unit="s"),
        MPCVariable(name=glbs.MARKET_TIME, value=0, unit="s"),
        MPCVariable(name=glbs.FLEX_EVENT_DURATION, value=0, unit="s"),
    ]
    config_inputs_appendix: MPCVariables = [
        MPCVariable(name="in_provision", value=False),
    ]
    weights: List[MPCVariable] = pydantic.Field(
        default=[],
        description="Name and value of weights",
    )

