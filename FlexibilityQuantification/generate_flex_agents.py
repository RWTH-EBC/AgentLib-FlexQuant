from copy import deepcopy
import json
from agentlib.utils import custom_injection
import ast
import atexit
import os
from itertools import zip_longest

def _generate_flex_model_definition(output_fname, model_specs, controls, power_variable,
                                    weights, pos_flex_cost_func, neg_flex_cost_func, std_cost_func,
                                    profile_deviation_weight):
    """
    Generates a python module for negative and positive flexbility agents from the Baseline MPC model 
        (specified with the dict @model_specs)

    The @power_variable must be defined on the Baseline model. 
    The @weights are added to the config classes as parameters, 
        they must have the pydantic form
    The @std_cost_func can be None, if so the cost function of the baseline MPC is used.
    @profile_deviation_weight specifies the weight of the soft constraint
    
    In the module @output_fname the following classes will be written:

    Configs
        These two configs are copied from the config of base MPC. The following changes are made:
        * BaseConfig (named as in model_specs):
            The weights and the input for the flexibility provision are added to the respective fields.
            The full trajectories for the controls are also added to the outputs
            Input for the flexibility provision: _P_external
            Inputs for the Relative start and end of the flexibility provision.
            Input for the Time.
        * FlexShadowMPCConfig:
            The weights are added to the parameters, the control trajectories to the inputs.
            Flexibility variables are added to the parameters
            Input for the Time.

    Models:
        The Models are copies of the base MPC with the following changes
        BaselineMPC:
            Hard constraint of the power variable during the flex. event.
            The control trajectory is sent with the respective variables
            If @std_cost_func is not None, then it is set to be the cost function
            The power variable is soft constrained during the flex. event.
        Pos and NegFlexModel s:
            During the market time, the controls are constrained with the
                control trajectory.
            The objective functions of the respective case are added. Casadi's 
                if_else function is used to switch between objectives.
                    t=0->t=market time + prep_time : obj_std
                    t=market time + prep_time->t=t+flex_event_duration : obj_flex
                    t=market time + prep_time + flex_event_duration -> : obj_std
    """
    fname = model_specs["file"]
    
    model_name = model_specs["class_name"]
    # Get the class name for the config
    config_class = custom_injection(model_specs).__annotations__["config"]
    config_instance = config_class()
    # extract the variables of the model
    model_vars = config_instance.dict(include={"inputs", "parameters", "outputs", "states"})
    if all(power_variable not in var["name"] for var in model_vars["outputs"]):
        raise ValueError(f"The power variable {power_variable} is not found on the model outputs!")
    model_vars_list = []
    for var_tup in zip_longest(
        model_vars["inputs"], model_vars["parameters"], 
        model_vars["outputs"], model_vars["states"], weights,
        fillvalue=None):
        for var in var_tup:
            if var is None:
                continue
            model_vars_list.append(var["name"])

    config_name = config_class.__name__
    def get_element_with_name(parsed_ast, name):
        """
        helper function to get an element from the ast object. 
        Note: only works for ast objects that have the attribute name!
        """
        for i, body in enumerate(parsed_ast.body):
            try:
                if body.name == name:
                    return i, body
            except AttributeError:
                pass
        return None, None

    # generate the ast object by reading the file of the baseline MPC
    with open(fname) as f:
        string_ = f.read()
    parsed = ast.parse(string_)

    class_ind, class_body = get_element_with_name(parsed, config_name)
    # copy the config to be used for the flex case
    parsed.body.append(deepcopy(parsed.body[class_ind]))
    # Add the new weights and control trajectories to the baseline config class
    for i, body in enumerate(class_body.body):
        if body.target.id == "parameters":
            for weight in weights:
                body.value.elts.append(ast.parse(f"CasadiParameter(name='{weight['name']}', value=0, unit='-', description='Weight for P in objective function')").body[0].value)
        if body.target.id == "outputs":
            for control in controls:
                body.value.elts.append(ast.parse(f"CasadiOutput(name='{control}_full', unit='W', type='pd.Series', value=pd.Series([0]), description='full control output')").body[0].value)
            

    # add the flexibility inputs
    for i, body in enumerate(parsed.body[class_ind].body):
        if body.target.id == "inputs":
            body.value.elts.append(ast.parse("CasadiInput(name='Time', value=0, unit='s', description='time trajectory')").body[0].value)
            body.value.elts.append(ast.parse(f"CasadiInput(name='_P_external', value=0, unit='W', description='External power profile to be provised')").body[0].value)
            body.value.elts.append(ast.parse(f"CasadiInput(name='in_provision', value=False, unit='-', description='Flag signaling if the flexibility is in provision')").body[0].value)
            body.value.elts.append(ast.parse(f"CasadiInput(name='rel_start', value=0, unit='s', description='relative start time of the flexibility event')").body[0].value)
            body.value.elts.append(ast.parse(f"CasadiInput(name='rel_end', value=0, unit='s', description='relative end time of the flexibility event')").body[0].value)

    # parse the flex config class
    parsed.body[-1].name = "FlexShadowMPCConfig"
    class_ind, class_body = get_element_with_name(parsed, "FlexShadowMPCConfig")
    # Add the new variables to the class
    for i, body in enumerate(class_body.body):
        if body.target.id == "inputs":
            body.value.elts.append(ast.parse("CasadiInput(name='Time', value=0, unit='s', description='time trajectory')").body[0].value)
            for control in controls:
                # add the control trajectorsy inputs
                body.value.elts.append(ast.parse(f"CasadiInput(name='_{control}', unit='W', type='pd.Series', value=pd.Series([0]))").body[0].value)
            body.value.elts.append(ast.parse(f"CasadiInput(name='in_provision', unit='-', value=False)").body[0].value)
        # add the flex variables and the weights
        if body.target.id == "parameters":
            body.value.elts.append(ast.parse("CasadiParameter(name='prep_time', value=0, unit='s', description='time to switch objective')").body[0].value)
            body.value.elts.append(ast.parse("CasadiParameter(name='flex_event_duration', value=0, unit='s', description='time to switch objective')").body[0].value)
            body.value.elts.append(ast.parse("CasadiParameter(name='market_time', value=0, unit='s', description='time to switch objective')").body[0].value)
            for weight in weights:
                body.value.elts.append(ast.parse(f"CasadiParameter(name='{weight['name']}', value=0, unit='-', description='Weight for P in objective function')").body[0].value)           

    class_ind, class_body = get_element_with_name(parsed, model_name)
    # generate the baseline class, dont edit the objective yet, as it will be copied!
    parsed.body[class_ind].name = "BaselineMPC"

    func_ind, func_body = get_element_with_name(class_body, "setup_system")
    # remove the return object of the function
    for ind in reversed(range(len(func_body.body))):
        if isinstance(func_body.body[ind], ast.Return):
            func_body.body.pop(ind)

    parsed.body.append(deepcopy(parsed.body[class_ind]))

    def check_args(assign_obj, args):
        """function to check if all variables used in @assign_obj are in the list @args"""
        args_in_assign = []
        for a in ast.walk(assign_obj):
            if isinstance(a, ast.Attribute):
                args_in_assign.append(a.attr)
        
        if not all(arg in args for arg in args_in_assign):
            missing_args = [arg for arg in args_in_assign if arg not in args]
            raise ValueError(f"Variables in the cost function {missing_args} are not assigned as module variables!")
        
    
    obj_flex = ast.parse(f"obj_flex = {pos_flex_cost_func}").body[0]
    check_args(obj_flex, model_vars_list)
    # if there is no standard cost function in config use the one from the baseline system
    if std_cost_func in (None, ""):
        func_ind, func_body = get_element_with_name(class_body, "setup_system")

        for ind in reversed(range(len(func_body.body))):
            if isinstance(func_body.body[ind], ast.Return):
                std_cost_func = ast.unparse(func_body.body[ind].value)

    
    obj_std = ast.parse(f"obj_std = {std_cost_func}").body[0]
    check_args(obj_std, model_vars_list)

    # Generate the class for positive flexibility
    parsed.body[-1].name = "PosFlexModel"
    class_ind, class_body = get_element_with_name(parsed, "PosFlexModel")
    # loop through the body until the config_type definition is found
    for b in parsed.body[class_ind].body:
        try:
            if b.annotation.id == config_name:
                b.annotation.id = "FlexShadowMPCConfig"
        except AttributeError:
            pass

    # Before changing the objective function, duplicate the class for pos. flexibility as a template for neg. flexibility
    parsed.body.append(deepcopy(parsed.body[class_ind]))
    parsed.body[-1].name = "NegFlexModel"

    # find the function setup_system, which defines the objective
    func_ind, func_body = get_element_with_name(class_body, "setup_system")

    # constraint the control trajectories for t < market_time 
    for ind in reversed(range(len(func_body.body))):
        if isinstance(func_body.body[ind], ast.Assign):
            if isinstance(func_body.body[ind].targets[0], ast.Attribute) and func_body.body[ind].targets[0].attr == "constraints":
                for i, control in enumerate(controls):
                    func_body.body[ind+2*i].value.elts.append(ast.parse(f"({control}_neg, self.{control}, {control}_pos)").body[0])
                    func_body.body.insert(
                        ind, ast.parse(
                            f"{control}_pos = ca.if_else(self.Time.sym < self.market_time.sym, self._{control}.sym, self.{control}.ub)").body[0])
                    func_body.body.insert(
                        ind, ast.parse(
                            f"{control}_neg = ca.if_else(self.Time.sym < self.market_time.sym, self._{control}.sym, self.{control}.lb)").body[0])
                break
    func_body.body.append(obj_flex)
    func_body.body.append(obj_std)
    # append the cost function with the if_else switches
    func_body.body.append(ast.parse(
        "return ca.if_else(self.Time.sym < self.prep_time.sym + self.market_time.sym, obj_std,"
        "ca.if_else(self.Time.sym < (self.prep_time.sym + self.flex_event_duration.sym + self.market_time.sym), obj_flex, obj_std))").body[0])

    # the same steps for the negative flexibility
    func_ind, func_body = get_element_with_name(parsed.body[-1], "setup_system")
        
    for ind in reversed(range(len(func_body.body))):
        if isinstance(func_body.body[ind], ast.Assign):
            if isinstance(func_body.body[ind].targets[0], ast.Attribute) and func_body.body[ind].targets[0].attr == "constraints":
                for i, control in enumerate(controls):
                    func_body.body[ind+2*i].value.elts.append(ast.parse(f"({control}_neg, self.{control}, {control}_pos)").body[0])
                    func_body.body.insert(
                        ind, ast.parse(
                            f"{control}_pos = ca.if_else(self.Time.sym < self.market_time.sym, self._{control}.sym, self.{control}.ub)").body[0])
                    func_body.body.insert(
                        ind, ast.parse(
                            f"{control}_neg = ca.if_else(self.Time.sym < self.market_time.sym, self._{control}.sym, self.{control}.lb)").body[0])
                break
    obj_flex = ast.parse(f"obj_flex = {neg_flex_cost_func}").body[0]
    check_args(obj_flex, model_vars_list)
    func_body.body.append(obj_flex)
    func_body.body.append(obj_std)
   
    func_body.body.append(ast.parse(
        "return ca.if_else(self.Time.sym < self.prep_time.sym + self.market_time.sym, obj_std,"
        "ca.if_else(self.Time.sym < (self.prep_time.sym + self.flex_event_duration.sym + self.market_time.sym), obj_flex, obj_std))").body[0])

    class_ind, class_body = get_element_with_name(parsed, "BaselineMPC")
    func_ind, func_body = get_element_with_name(class_body, "setup_system")
        
    # set the control trajectories with the respective variables
    for control in controls:
        func_body.body.append(ast.parse(f"self.{control}_full.alg = self.{control}"))
    # add the soft constraint for the provision
    func_body.body.append(obj_std)
    func_body.body.append(ast.parse("return obj_std + ca.if_else(self.in_provision.sym, "
                                    "ca.if_else(self.Time.sym < self.rel_start.sym, 0, "
                                    "ca.if_else(self.Time.sym > self.rel_end.sym, 0, "
                                    f"sum([{profile_deviation_weight}*(self.{power_variable} - self._P_external)**2]))),0)").body[0])

    with open(output_fname, "w+") as f:
        f.write(ast.unparse(parsed))

def _delete_changes(to_be_deleted):
    """
    function to run at exit if the files are to be deleted
    """
    for file in to_be_deleted:
        os.remove(file)

def generate_flex_agents(baseline_mpc_config, flex_config, flex_file="flex_agents.py"):
    """
    Generates the configs and the python module for the flexibility agents.
    
    @baseline_mpc_config: path to the json for the baseline model
    @indicator_config: path to json holding the config for the flexibility indicator
    @flex_config: path to json holding the variables for the flexibility
            keys:
                "prep_time": preperation time
                "market_time": time in which the offer will be accepted/rejected
                "flex_event_duration": duration of the flex event
                "weights": new weights
                "pos_flex_cost_function": cost function to calculate pos flex
                "neg_flex_cost_function":  cost function to calculate neg flex
                "standard_cost_function": cost function for the base case, can be None
                "profile_deviation_weight": weight of the soft constraint of the provision
                "random_seed": random seed for reproducability
                "pos_neg_rate": the rate of the pos flex acceptance after a offer is accepted
                "offer_acceptance_rate": the rate of offer acceptance
                "minimum_average_flex": the minimum average of an offer to be accepted
                "power_variable": power variable in the model
                "delete_files": if set the generated files are deleted at exit 
                "indicator_config": name of the json holding the config of the flex indicators. should be in the flex files directory
                "market_config": name of the json holding the config of the flex market. should be in the flex files directory
                "path_to_flex_files": the path to the flex files, relative to the cwd of the running program
                "cooldown_timesteps": the cooldown steps after a flex event during which no offer is accepted.
                "forced_offers": the dictionary to hold the predefined offers:
                               {timestep as string: ["positive" or "negative", power_multiplier]} 

    """
    to_be_deleted = []

    def append_file_suffix(path):
        return os.path.join(flex_config["path_to_flex_files"], path)
    
    with open(flex_config) as f:
        flex_config = json.load(f)
    
    with open(baseline_mpc_config) as f:
        baseline_mpc_config_dict = json.load(f)

    # file name for the copy of the baseline
    baseline_mpc_config = "baseline.json"
    
    # copy the mpc config as a template for the flexibility agents
    pos_flex_mpc_config = deepcopy(baseline_mpc_config_dict)
    neg_flex_mpc_config = deepcopy(baseline_mpc_config_dict)
    indicator_config = flex_config["indicator_config"]
    market_config = flex_config["market_config"]
    

    with open(append_file_suffix(indicator_config)) as f:
        indicator_config_dict = json.load(f)
    indicator_config_dict["modules"][1]["type"]["file"]  = append_file_suffix(indicator_config_dict["modules"][1]["type"]["file"])
    
    with open(append_file_suffix(market_config)) as f:
        market_config_dict = json.load(f)
    market_config_dict["modules"][1]["type"]["file"]  = append_file_suffix(market_config_dict["modules"][1]["type"]["file"])
    # file name for the copy of the indicator config
    indicator_config = "indicator.json"

    to_be_deleted.append(flex_file)
    to_be_deleted.append(baseline_mpc_config)
    to_be_deleted.append(indicator_config)
    to_be_deleted.append(market_config)


    pos_flex_mpc_config["id"] = "PosFlexMPC"
    for i, module in enumerate(pos_flex_mpc_config["modules"]):
        if module["type"] == "agentlib_mpc.mpc":
            # append the new weights as parameters to the baseline MPC
            for weight in flex_config["weights"]:
                in_params = False

                for param_ind, param in enumerate(baseline_mpc_config_dict["modules"][i]["parameters"]):
                    if param == weight:
                        baseline_mpc_config_dict["modules"][i]["parameters"][param_ind]["value"] = weight["value"]
                        in_params = True
                if not in_params:
                    baseline_mpc_config_dict["modules"][i]["parameters"].append(
                        weight
                    )
            baseline_mpc_config_dict["modules"][i]["type"] = {
                "file": append_file_suffix("shadow_mpc.py"),
                "class_name": "FlexibilityProvisorMPC"
            }
            baseline_mpc_config_dict["modules"][i]["optimization_backend"]["model"]["type"] = { 
                "file": flex_file,
                "class_name": "BaselineMPC"
            }
            baseline_mpc_config_dict["modules"][i]["optimization_backend"]["results_file"] = baseline_mpc_config_dict["modules"][i]["optimization_backend"]["results_file"].replace(".csv", "_base.csv")
            baseline_mpc_config_dict["modules"][i]["module_id"] = "FlexMPC"
    
            controls = [x["name"] for x in module["controls"]]

            # Set the variable alias for the power variable as P_el_base.
            in_outputs = False
            for output_ind, output in enumerate(baseline_mpc_config_dict["modules"][i]["outputs"]):
                if output["name"] == flex_config["power_variable"]:
                    baseline_mpc_config_dict["modules"][i]["outputs"][output_ind]["alias"] = "__P_el_base"#_raw"
                    in_outputs = True
            if not in_outputs:
                baseline_mpc_config_dict["modules"][i]["outputs"].append(
                    {"name": flex_config["power_variable"], "alias":"__P_el_base"}#_raw"},
                )
            

            # Set the variable alias for the power variable as P_el_base.
            in_outputs = False
            for control in controls:
                baseline_mpc_config_dict["modules"][i]["outputs"].append(
                        {"name": f"{control}_full", "alias":f"{control}_full"}#_raw"},
                    )
            
            baseline_mpc_config_dict["modules"][i]["inputs"].extend([
                {
                    "name": "_P_external",
                    "value": 0
                },
                {
                    "name": "in_provision",
                    "value": False
                },
                {
                    "name": "rel_start",
                    "value": 0
                },
                {
                    "name": "rel_end",
                    "value": 0
                }
            ])
            
            model_specs = module["optimization_backend"]["model"]["type"]

            timestep = module["time_step"]
            horizon = module["prediction_horizon"]
            # add the time variable to config
            baseline_mpc_config_dict["modules"][i]["inputs"].append(
                {"name": "Time", "value":[i * timestep for i in range(horizon)]}
            )

            pos_flex_mpc_config["modules"][i]["module_id"] = "PosFlexMPC" 
            # set the class of the ShadowMPC to be the custom mpc, as implemented in shadow_mpc.py
            pos_flex_mpc_config["modules"][i]["type"] = {
                "file": append_file_suffix("shadow_mpc.py"),
                "class_name": "FlexibilityShadowMPC"
            } 
            pos_flex_mpc_config["modules"][i]["optimization_backend"]["model"]["type"] = { 
                "file": flex_file,
                "class_name": "PosFlexModel"
            }
            # modify the results file for the positive flexibility agent
            res_file = module["optimization_backend"]["results_file"]
            res_file = res_file.split(".")
            res_file[-2] += "_pos_flex"
            res_file = ".".join(res_file)
            pos_flex_mpc_config["modules"][i]["optimization_backend"]["results_file"] = res_file
            # generate time trajectory
            pos_flex_mpc_config["modules"][i]["inputs"].extend([
                {"name": "Time", "value":[i * timestep for i in range(horizon)]},
                {"name": "in_provision", "value": False},
            ])
            # Set the variable alias for the power variable as P_el_min for the positive flexibility case
            in_outputs = False
            for output_ind, output in enumerate(pos_flex_mpc_config["modules"][i]["outputs"]):
                if output["name"] == flex_config["power_variable"]:
                    pos_flex_mpc_config["modules"][i]["outputs"][output_ind]["alias"] = "__P_el_min"
                    in_outputs = True
            if not in_outputs:
                pos_flex_mpc_config["modules"][i]["outputs"].append(
                    {"name": "P_el", "alias":"__P_el_min"},
                )

            # append the flexibility variables to the parameters
            pos_flex_mpc_config["modules"][i]["parameters"].extend(
                [
                    {"name": "prep_time", "value": flex_config["prep_time"]},
                    {"name": "market_time", "value": flex_config["market_time"]},
                    {"name": "flex_event_duration", "value": flex_config["flex_event_duration"]},
                ]
            )
            for control in controls:
                pos_flex_mpc_config["modules"][i]["inputs"].append({"name": f"_{control}", "value": 0})
            pos_flex_mpc_config["modules"][i]["parameters"].extend(flex_config["weights"])
            # to make sure the flexibility agents dont control the simulation, set the shared variable fields
            pos_flex_mpc_config["modules"][i]["shared_variable_fields"] = ["outputs"]

            # and to the indicator dict
            indicator_config_dict["modules"][1]["parameters"] = [
                    {"name": "prep_time", "value": flex_config["prep_time"] + flex_config["market_time"]},
                    {"name": "flex_event_duration", "value": flex_config["flex_event_duration"]},
                    {"name": "time_step", "value": timestep},
                    {"name": "prediction_horizon", "value": horizon},
                
            ]

            market_config_dict["modules"][1]["parameters"] = [ 
                {"name": "random_seed", "value": flex_config.get("random_seed")},
                {"name": "pos_neg_rate", "value": flex_config.get("pos_neg_rate", 0)},
                {"name": "offer_acceptance_rate", "value": flex_config.get("offer_acceptance_rate", 1)},
                {"name": "minimum_average_flex", "value": flex_config.get("minimum_average_flex", 0)},
                {"name": "maximum_time_flex", "value": flex_config.get("maximum_time_flex", 0)},
                {"name": "time_step", "value": timestep},
                {"name": "cooldown", "value": flex_config.get("cooldown_timesteps", 0)},
                {"name": "forced_offers", "value":flex_config.get("forced_offers", {})}
                                                                  
            ] 
           
            break
        
    neg_flex_mpc_config["id"] = "NegFlexMPC"
    for i, module in enumerate(neg_flex_mpc_config["modules"]):
        if module["type"] == "agentlib_mpc.mpc":
            neg_flex_mpc_config["modules"][i]["module_id"] = "NegFlexMPC" 
            neg_flex_mpc_config["modules"][i]["optimization_backend"]["model"]["type"] = { 
                "file": flex_file,
                "class_name": "NegFlexModel"
            }
            neg_flex_mpc_config["modules"][i]["type"] = {
                "file": append_file_suffix("shadow_mpc.py"),
                "class_name": "FlexibilityShadowMPC"
            }
            res_file = module["optimization_backend"]["results_file"]
            # modify the results file for the negative flexibility agent
            res_file = res_file.split(".")
            res_file[-2] += "_neg_flex"
            res_file = ".".join(res_file)
            neg_flex_mpc_config["modules"][i]["optimization_backend"]["results_file"] = res_file
            # generate time trajectory
            neg_flex_mpc_config["modules"][i]["inputs"].extend([
                {"name": "Time", "value":[i * timestep for i in range(horizon)]},
                {"name": "in_provision", "value": False},
            ])
            # append the flexibility variables to the parameters
            neg_flex_mpc_config["modules"][i]["parameters"].extend(
                [
                    {"name": "prep_time", "value": flex_config["prep_time"]},
                    {"name": "market_time", "value": flex_config["market_time"]},
                    {"name": "flex_event_duration", "value": flex_config["flex_event_duration"]},
                ]
            )
            neg_flex_mpc_config["modules"][i]["parameters"].extend(flex_config["weights"])
            for control in controls:
                neg_flex_mpc_config["modules"][i]["inputs"].append({"name": f"_{control}", "value": 0})
            
            # Set the variable alias for the power variable as P_el_min for the positive flexibility case
            in_outputs = False
            for output_ind, output in enumerate(neg_flex_mpc_config["modules"][i]["outputs"]):
                if output["name"] == flex_config["power_variable"]:
                    neg_flex_mpc_config["modules"][i]["outputs"][output_ind]["alias"] = "__P_el_max"
                    in_outputs = True
            if not in_outputs:
                neg_flex_mpc_config["modules"][i]["outputs"].append(
                    {"name": flex_config["power_variable"], "alias":"__P_el_max"},
                )
            # to make sure the flexibility agents dont control the simulation, set the shared variable fields
            neg_flex_mpc_config["modules"][i]["shared_variable_fields"] = ["outputs"]
            break
    
    # # write out jsons
    with open(baseline_mpc_config, "w+") as f:
        json.dump(baseline_mpc_config_dict, f, indent=4)

    pos_flex_mpc_config_fname = baseline_mpc_config.replace("baseline", "pos_flex")
    to_be_deleted.append(pos_flex_mpc_config_fname)
    with open(pos_flex_mpc_config_fname, "w+") as f:
        json.dump(pos_flex_mpc_config, f, indent=4)

    neg_flex_mpc_config_fname = baseline_mpc_config.replace("baseline", "neg_flex")
    to_be_deleted.append(neg_flex_mpc_config_fname)

    with open(neg_flex_mpc_config_fname, "w+") as f:
        json.dump(neg_flex_mpc_config, f, indent=4)
    
    with open(indicator_config, "w+") as f:
        json.dump(indicator_config_dict, f, indent=4) 
    

    with open(market_config, "w+") as f:
        json.dump(market_config_dict, f, indent=4) 

    # generate the module
    _generate_flex_model_definition(flex_file, model_specs, controls, flex_config["power_variable"], flex_config["weights"],
                                    flex_config["pos_flex_cost_function"], flex_config["neg_flex_cost_function"],
                                    flex_config["standard_cost_function"], flex_config["profile_deviation_weight"])
    
    # register the exit function if the corresponding flag is set
    if flex_config["delete_files"]:
        atexit.register(lambda: _delete_changes(to_be_deleted))
    return pos_flex_mpc_config_fname, neg_flex_mpc_config_fname, baseline_mpc_config, indicator_config, market_config