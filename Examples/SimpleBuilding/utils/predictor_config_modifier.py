import json
import agentlib as al
import os
from agentlib.utils import custom_injection, load_config 
import os
from typing import Union, List
from pydantic import FilePath
from pathlib import Path
from agentlib.core.agent import AgentConfig
from Examples.SimpleBuilding.predictor.simple_predictor import PredictorModuleConfig, FlexEvent, PredictorModule
from flexibility_quantification.utils import config_management as cmng
import shutil




class PredictorConfigModifier(PredictorModule):
    
     
    def __init__(self, predictor_config: Union[FilePath, str]): 
    
        self.predictor_config = Path(predictor_config)
        self.temp_path = self.predictor_config.parent / "predictor_config_temp.json"
        self.predictor_config_temp = self.load_predictor_config()
        # self.schedule_type = self.config.schedule.type
      

        
    def load_predictor_config(self) -> PredictorModuleConfig:
        
    
        """
        If predictor_config_temp.json exists, return its path.
        If not, make a deep copy of self.predictor_config to predictor_config_temp.json and return the temp path.
        """
        if self.temp_path.exists():
            with open(self.temp_path, 'r') as file:
                config_data = json.load(file)
            return config_data
        else:
            shutil.copy(self.predictor_config, self.temp_path)
            with open(self.temp_path, 'r') as file:
                config_data = json.load(file)
            return config_data


    def modify_predictor_config(self, flex_event_params: Union[FlexEvent]):
        """
        Modify the predictor configuration with flex event parameters and save it.
        """
        # Load the config
        # with open(self.temp_path, 'r') as file:
        #     config_data = json.load(file)

        config_data = self.predictor_config_temp

        # Find the MyPredictor module
        for module in config_data.get("modules", []):
            if module.get("module_id") == "MyPredictor":
                parameters = module.setdefault("parameters", [])
                # Try to find the flex_event parameter
                for param in parameters:
                    if param.get("name") == "flex_event":
                        # Update existing flex_event
                        param["value"] = flex_event_params.model_dump() if isinstance(flex_event_params, FlexEvent) else flex_event_params
                        break
                else:
                    # If not found, add a new flex_event parameter
                    parameters.append({
                        "name": "flex_event",
                        "value": flex_event_params.model_dump() if isinstance(flex_event_params, FlexEvent) else flex_event_params
                    })
                break

        # Save the modified config
        with open(self.temp_path, 'w') as file:
            json.dump(config_data, file, indent=4)
            
        return self.temp_path

    # def get_basecomfort_type(self):
    #     config_default= PredictorModuleConfig() 
    #     return config_default.schedule



 