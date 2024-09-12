
from agentlib.modules import get_all_module_types
import inspect
from agentlib.utils import custom_injection

all_module_types = get_all_module_types(["agentlib_mpc", "flexibility_quantification"])
# remove ML models, since import takes ages
all_module_types.pop("agentlib_mpc.ann_trainer")
all_module_types.pop("agentlib_mpc.gpr_trainer")
all_module_types.pop("agentlib_mpc.linreg_trainer")
all_module_types.pop("agentlib_mpc.ann_simulator")
all_module_types.pop("agentlib_mpc.set_point_generator")
# remove clone since not used
all_module_types.pop("clonemap")

MODULE_TYPES = {name: inspect.get_annotations(class_type.import_class())["config"] for name, class_type in all_module_types.items()}

MPC_CONFIG_TYPE: str = "agentlib_mpc.mpc"
INDICATOR_CONFIG_TYPE: str = "flexibility_quantification.flexibility_indicator"
MARKET_CONFIG_TYPE: str = "flexibility_quantification.flexibility_market"
