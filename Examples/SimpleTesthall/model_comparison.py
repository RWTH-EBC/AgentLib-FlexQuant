import sys, os
os.chdir(os.path.dirname(__file__))
sys.path.append(os.path.join("..",".."))
import pickle
import plots
from typing import List

import json
import agentlib as al
from agentlib.utils.multi_agent_system import LocalMASAgency

predictor_config = "Model//local//predictor//config.json"
mpc_config = f"Model//local//mpc//config.json"
flex_config = f"flexibility_agent_config.json"
ca_sim_config = "Model//local//mpc//ca_simu.json" 
fmu_config = "Model//local//fmu//config.json"
folder = f"debug/model_correction"
if not os.path.exists(folder):
    os.mkdir(folder)

def set_delay_const(val):
    with open(ca_sim_config) as f:
        d = json.load(f)
    d["modules"][1]["model"]["parameters"][0]["value"] = val
    with open(ca_sim_config, "w+") as f:
        json.dump(d, f, indent=4)
    
class DummyControllerConfig(al.BaseModuleConfig):
    outputs: List[al.AgentVariable] = [
        al.AgentVariable(
            name="Q_Tabs_set", alias="Q_Tabs_set",
            value=0, description="", shared=True
        )]
class DummyController(al.BaseModule):
    sprung_list = [(0,0), (355000, 4000), (365000, 0), (375000, 400), (376000, 800), (377000, 1200), (378000, 1600), (379000, 2000),
(380000, 2400), (381000, 2800), (382000, 3200), (383000, 3600), (384000, 4000), (385000, 3600),
(386000, 3200), (387000, 2800), (388000, 2400), (389000, 2000), (390000, 1600), (391000, 1200),
(392000, 800), (393000, 400), (394000, 0), (397000, 1200), (400000, 1200),
(403000, 3600),
(408000, 2400), (411000, 1200),  (414000, 0)]
    index = 0
    config: DummyControllerConfig
    def process(self):
        while True:
            self.set("Q_Tabs_set", self.sprung_list[self.index][1])
            if self.index < len(self.sprung_list) - 1:
                if self.env.time > self.sprung_list[self.index+1][0]:
                    self.index += 1 
            yield self.env.timeout(self.env.config.t_sample)

    def register_callbacks(self):
        pass
    
if __name__ == "__main__":
    agent_configs = [
        {"id": "Controller",
            "modules": [
                {
                    "module_id": "Ag100Com",
                    "type": "local_broadcast"
                },
                {
                    "module_id": "Controller",
                    "type": {
                    "file":__file__,
                    "class_name": "DummyController"
                    },
                },
            ]
        },
        predictor_config,
        None
    ]
    start_day = 2 
    duration = 1

    initial_time = 172800 + 86400 * start_day

    until = initial_time + 86400 * duration

    env_config = {"rt": False, "t_sample": 60, "offset": initial_time}

    agent_configs[-1] = fmu_config

    mas = LocalMASAgency(
        agent_configs=agent_configs,
        env=env_config,
        variable_logging=True,
    )
    mas.run(until=until)
    fmu_res =  mas.get_results(cleanup=True)
    results = []

    const_list = [200, 300, 400, 500]
    agent_configs[-1] = ca_sim_config

    for delay_const in const_list:
        set_delay_const(delay_const)
        mas = LocalMASAgency(
            agent_configs=agent_configs,
            env=env_config,
            variable_logging=True,
        )
        mas.run(until=until)
        results.append(mas.get_results(cleanup=True))
    with open(f"{folder}/sim_comp.res", "w+b") as f:
        pickle.dump((fmu_res, results), f)

    fig = plots.debug_print_mult_sim(fmu_res, results, const_list)
    fig.savefig(f"{folder}/sim_comp")