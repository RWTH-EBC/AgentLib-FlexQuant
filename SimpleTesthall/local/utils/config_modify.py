import json
from utilities.modelica_parser import parse_modelica_record


def choose_agent_configs(cal_flexibility: bool, use_scalar: bool):
    if cal_flexibility:
        agent_configs = [
            "flex_configs/flexibility_agent_config.json",
            "mpc//config.json",
            "fmu//config.json",
            "predictor//config.json",
            #"shadow_mpc//config_NF_mpc.json",
            #"shadow_mpc//config_PF_mpc.json",
            #"flexibility//config.json"
        ]
    else:
        if use_scalar:#casadi-simulator
            agent_configs = [
                "mpc//ca_simu.json",
                "fmu//config.json",
                "predictor//config_scalar.json"
            ]
        else:
            agent_configs = [
                "mpc//config.json",
                "fmu//config.json",
                "predictor//config.json"
            ]
    return agent_configs



def config_time_traj(ts, n, config_name):
    """
    Add time trajectory to config of shadow mpc depending on the characteristic times

    ts: time step of mpc
    n: number of time steps during prediction horizon
    config_name: name of config of mpc
    """
    file_name = f'shadow_mpc/config_{config_name}.json'

    with open(file_name, 'r') as config_file:
        config_data = json.load(config_file)

    Time = list(range(0, ts * n, ts))
    config_data["modules"][1]["inputs"][0]["value"] = Time

    with open(file_name, 'w') as config_file:
        json.dump(config_data, config_file, indent=4)


def config_ts_ph(ts, ph, agent_configs):
    for config in agent_configs:
        if config == "mpc//ca_simu.json" or config == "fmu//config.json":
            pass
        elif config == "predictor//config.json" or config == "predictor//config_scalar.json":
            with open(config, 'r') as config_file:
                config_data = json.load(config_file)
            config_data["modules"][1]["parameters"][1]["value"] = ts
            config_data["modules"][1]["parameters"][2]["value"] = ph + 1

            with open(config, 'w') as config_file:
                json.dump(config_data, config_file, indent=4)
        elif config == "flexibility//config.json":
            with open(config, 'r') as config_file:
                config_data = json.load(config_file)
            config_data["modules"][1]["parameters"][1]["value"] = ts
            config_data["modules"][1]["parameters"][2]["value"] = ph

            with open(config, 'w') as config_file:
                json.dump(config_data, config_file, indent=4)
        else:
            with open(config, 'r') as config_file:
                config_data = json.load(config_file)
            config_data["modules"][1]["time_step"] = ts
            config_data["modules"][1]["prediction_horizon"] = ph

            with open(config, 'w') as config_file:
                json.dump(config_data, config_file, indent=4)


def choose_mode(price_mode):
    """
    price_mode = 0: constant power price
    price_mode = 1: base and peak power price
    price_mode = 2: dynamic power price
    """
    file_name = f'predictor/config.json'

    with open(file_name, 'r') as config_file:
        config_data = json.load(config_file)

    config_data["modules"][1]["parameters"][3]["value"] = price_mode

    with open(file_name, 'w') as config_file:
        json.dump(config_data, config_file, indent=4)


def update_fmu_config(path_fmu, path_record):
    config_fmu = f'fmu/config.json'
    with open(config_fmu, 'r') as config_file:
        config_data = json.load(config_file)
    # update FMU path
    config_data["modules"][1]["model"]["path"] = path_fmu

    # Add T_Floor as output in config
    new_output = {"name": "T_Floor"}
    has_floor = parse_modelica_record(path_record)['AFloor'] > 0
    outputs = config_data["modules"][1]["outputs"]
    if has_floor:
        if new_output not in outputs:
            outputs.append(new_output)
    else:
        outputs = [item for item in outputs if item != new_output]
    config_data["modules"][1]["outputs"] = outputs

    with open(config_fmu, 'w') as config_file:
        json.dump(config_data, config_file, indent=3)


def update_predictor_config(path_fmu, path_zone_record,path_rad_record):
    config_disturbances = f'predictor/setup_disturbances.json'
    with open(config_disturbances, 'r') as config_file:
        config_data = json.load(config_file)
    config_data["path_fmu"] = path_fmu
    config_data["path_zone_record"] = path_zone_record
    config_data["path_radiator_record"] = path_rad_record
    with open(config_disturbances, 'w') as config_file:
        json.dump(config_data, config_file, indent=4)


def update_mpc_config(path_record):
    config_ca_simu = f'mpc/ca_simu.json'
    config_mpc = f'mpc/config.json'
    with open(config_ca_simu, 'r') as config_file:
        config_data = json.load(config_file)

    new_model_states = {"name": "T_Floor", "value": 290.15}
    new_state = {"name": "T_Floor", "value": 290.15, "shared": True, "alias": "T_Floor"}
    has_floor = parse_modelica_record(path_record)['AFloor'] > 0

    module = config_data["modules"][1]
    model_states = module["model"]["states"]
    states = module["states"]

    if has_floor:
        if not any(item["name"] == "T_Floor" for item in model_states):
            model_states.append(new_model_states)
        if not any(item["name"] == "T_Floor" for item in states):
            states.append(new_state)
    else:
        model_states = [item for item in model_states if item["name"] != "T_Floor"]
        states = [item for item in states if item["name"] != "T_Floor"]

    module["model"]["states"] = model_states
    module["states"] = states
    config_data["modules"][1] = module

    with open(config_ca_simu, 'w') as config_file:
        json.dump(config_data, config_file, indent=4)
