# Example for implementing a custom addon

This example is based on the SimpleBuilding example and implements a very simple custom market agent.

To implement a custom plugin following steps are needed:

### 1) Create a new plugin package for the AgentLib plugin

In this case the folder `agentlib_plugin` and `__init__.py` inside was created. Important: The plugin package directory name needs to be lowercase, else this will cause issues while importing the plugin.

### 2) Create custom logic (in this case market logic)
The custom market logic is implemented in `agentlib_plugin/custom_market.py` by creating a new class that is inheriting from the default FlexibilityMarket. In here we add a custom config option and overwrite the custom_flexibility_callback function to add our own logic. 

### 3) Add the custom module to the plugin package
Edit `agentlib_plugin/__init__.py` and add:

```python 
from agentlib.utils.plugin_import import ModuleImport
from . import custom_market

MODULE_TYPES = {
    "custommarket": ModuleImport(
        import_path="agentlib_plugin.custom_market", class_name=custom_market.CustomMarketModule.__name__
    ),
}
```

### 4) Tell FlexQuant to use our custom plugin
First we need to let the AgentGenerator know to load our custom plugin by adding the `"custom_plugins": ["agentlib_plugin"]` key to the `flexibility_agent_config.json`. 
This will ensure that the package is available to the ConfigHandler to get the module we define in the `market.json`.

Next `market.json` needs to be adapted to use the new custom market agent. For that the key `"module_type": "agentlib_plugin.custommarket"` has to be added at the top level of the dict. Then the module id and type need to be set to the new values:

```
          "module_id": "CustomMarketModule",  -> name of the custom class
          "type": "agentlib_plugin.custommarket", -> name of the package and module defined in agentlib_plugin/__init__.py
```


### 5) Test custom plugin
To test if all works correctly, run `main_single_run.py`. You should now see the custom string we set and the flex offers printed.