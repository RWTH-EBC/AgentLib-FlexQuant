"""agentlib plugin"""

from agentlib.utils.plugin_import import ModuleImport
from agentlib_mpc.optimization_backends import BackendImport, backend_types

from .modules import shadow_mpc
from .modules import baseline_mpc
from .modules import flexibility_indicator
from .modules import flexibility_market

MODULE_TYPES = {
    "shadow_mpc": ModuleImport(
        import_path="flexibility_quantification.modules.shadow_mpc",
        class_name=shadow_mpc.FlexibilityShadowMPC.__name__,
    ),
    "shadow_minlp_mpc": ModuleImport(
        import_path="flexibility_quantification.modules.shadow_mpc",
        class_name=shadow_mpc.FlexibilityShadowMINLPMPC.__name__,
    ),
    "baseline_mpc": ModuleImport(
        import_path="flexibility_quantification.modules.baseline_mpc",
        class_name=baseline_mpc.FlexibilityBaselineMPC.__name__,
    ),
    "baseline_minlp_mpc": ModuleImport(
        import_path="flexibility_quantification.modules.baseline_mpc",
        class_name=baseline_mpc.FlexibilityBaselineMINLPMPC.__name__,
    ),
    "flexibility_indicator": ModuleImport(
        import_path="flexibility_quantification.modules.flexibility_indicator",
        class_name=flexibility_indicator.FlexibilityIndicatorModule.__name__,
    ),
    "flexibility_market": ModuleImport(
        import_path="flexibility_quantification.modules.flexibility_market",
        class_name=flexibility_market.FlexibilityMarketModule.__name__,
    ),
}


backend_types.update(
    {
        "casadi_cia_cons": BackendImport(
            import_path="flexibility_quantification.optimization_backends.constrained_cia",
            class_name="ConstrainedCasADiCIABackend",
        )
    }
)
