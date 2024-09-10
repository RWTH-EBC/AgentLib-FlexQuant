import pydantic
from typing import List
from agentlib_mpc.data_structures.mpc_datamodels import MPCVariables, MPCVariable


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
    module_type: str
    class_name: str
    module_id: str
    # variables
    full_trajectory_suffix: str = "_full"
    power_alias: str
    config_inputs_appendix: MPCVariables = []
    config_parameters_appendix: MPCVariables = []


class BaselineMPCData(BaseMPCData):
    """Data class for Baseline MPC

    """
    # files and paths
    results_suffix: str = "_base.csv"
    name_of_created_file: str = "baseline.json"
    # modules
    module_type: str = "flexibility_quantification.baseline_mpc"
    class_name: str = "BaselineMPC"
    module_id: str = "Baseline"
    # variables
    power_alias: str = "__P_el_base"
    power_variable: str = pydantic.Field(
        default="P_el",
        description="Name of the variable representing the electrical power in the baseline config",
    )
    # TODO: wie mit diesen Daten umgehen? Vor Aufruf von adapt_mpc_module_config einmal diese Datenklasse initialisieren und Werte entsprechend setzen?
    config_inputs_appendix: MPCVariables = [
        MPCVariable(name="_P_external", value=0),
        MPCVariable(name="in_provision", value=False),
        MPCVariable(name="rel_start", value=0),
        MPCVariable(name="rel_end", value=0)
    ]


class PFMPCData(BaseMPCData):
    """Data class for PF-MPC

    """
    # files and paths
    results_suffix: str = "_pos_flex.csv"
    name_of_created_file: str = "pos_flex.json"
    # modules
    module_type: str = "flexibility_quantification.shadow_mpc"
    class_name: str = "PosFlexModel"
    module_id: str = "PosFlexMPC"
    # variables
    power_alias: str = "__P_el_pos"
    flex_cost_function: str = pydantic.Field(
        default=None,
        description="Cost function of the PF-MPC",
    )
    # initialize market parameters with dummy values (0)
    config_parameters_appendix: MPCVariables = [
        MPCVariable(name="prep_time", value=0),
        MPCVariable(name="market_time", value=0),
        MPCVariable(name="flex_event_duration", value=0),
    ]
    config_inputs_appendix: MPCVariables = [
        MPCVariable(name="in_provision", value=False),
    ]


class NFMPCData(BaseMPCData):
    """Data class for PF-MPC

    """
    # files and paths
    results_suffix: str = "_neg_flex.csv"
    name_of_created_file: str = "neg_flex.json"
    # modules
    module_type: str = "flexibility_quantification.shadow_mpc"
    class_name: str = "NegFlexModel"
    module_id: str = "NegFlexMPC"
    # variables
    power_alias: str = "__P_el_neg"
    flex_cost_function: str = pydantic.Field(
        default=None,
        description="Cost function of the NF-MPC",
    )
    # initialize market parameters with dummy values (0)
    config_parameters_appendix: MPCVariables = [
        MPCVariable(name="prep_time", value=0),
        MPCVariable(name="market_time", value=0),
        MPCVariable(name="flex_event_duration", value=0),
    ]
    config_inputs_appendix: MPCVariables = [
        MPCVariable(name="in_provision", value=False),
    ]

