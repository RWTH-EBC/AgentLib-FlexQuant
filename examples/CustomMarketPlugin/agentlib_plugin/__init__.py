from agentlib.utils.plugin_import import ModuleImport
from . import custom_market

MODULE_TYPES = {
    "custommarket": ModuleImport(
        import_path="agentlib_plugin.custom_market", class_name=custom_market.CustomMarketModule.__name__
    ),
}
