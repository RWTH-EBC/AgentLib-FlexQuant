import logging
from flexibility_quantification.generate_flex_agents import FlexAgentGenerator
from agentlib.utils.multi_agent_system import LocalMASAgency
from flexibility_quantification.utils.interactive import Dashboard, CustomBound
from agentlib_mpc.utils.plotting.interactive import show_dashboard
import os
from pathlib import Path
import json
from Examples.SimpleBuilding.utils.predictor_config_modifier import PredictorConfigModifier
from Examples.SimpleBuilding.predictor.simple_predictor import FlexEvent, PredictorModuleConfig


# from config import PathVariables
logging.basicConfig(level=logging.WARN)
until = 3600 * 24 


basepath = Path(__file__).parent
os.chdir(basepath)


ENV_CONFIG = {"rt": False, "factor": 10, "t_sample": 60} 

def run_example(until=until):

   
    mpc_config = basepath / "mpc_and_sim/simple_model.json"
    sim_config = basepath / "mpc_and_sim/fmu_config.json"
    predictor_config = basepath / "predictor/predictor_config.json"
    flex_config = basepath / "flex_configs/flexibility_agent_config.json"

  
    flex_event_params = FlexEvent(
        upper_boundary_shift=1,
        lower_boundary_shift=1.5,
        gradient=0.1,
        start=1800,
        end=5000)
  
    


    
    predictor_modifier = PredictorConfigModifier(predictor_config)
    predictor_config_temp_modified = predictor_modifier.modify_predictor_config(flex_event_params)

    

    # predictor_config_temp_modified = basepath / "predictor/predictor_config_temp.json"
    
                                                                      
                                        
    # neuer pfad z.B: predictor/predictor_config_temp.json" + flex event times + ?ph? | ?"hier auch mpc_config abrufen" Wofür??
    
    # basecomfort_type = "TestSchedule"

    # metadata = f"{flex_event_params.upper_boundary_shift}_{flex_event_params.lower_boundary_shift}_{flex_event_params.gradient}_{flex_event_params.start}_{flex_event_params.end}"
    
    # new_result_path = basepath / "results_{basecomfort_type}_{metadata}"


    config_list = FlexAgentGenerator(
        flex_config=flex_config, mpc_agent_config=mpc_config
    ).generate_flex_agents()

    # predictor_config_temp_path = basepath / "predictor/predictor_config_temp.json"

    # agent_configs = [sim_config, predictor_config]
    agent_configs = [sim_config, predictor_config_temp_modified] #predictor_config_temp_path
    agent_configs.extend(config_list)

    mas = LocalMASAgency(
        agent_configs=agent_configs, env=ENV_CONFIG, variable_logging=False
    )
    mas.run(until=until)

    results = []
    results = mas.get_results(cleanup=False)
    
    Dashboard( 
        flex_config=flex_config,
        simulator_agent_config=sim_config,
        results=results
    ).show(
        custom_bounds=CustomBound(
            for_variable="T_zone",
            lb_name="T_lower",
            ub_name="T_upper"
        )
    )    

if __name__ == "__main__":
    run_example(until)