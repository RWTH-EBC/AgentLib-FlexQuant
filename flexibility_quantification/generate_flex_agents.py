import inspect
import logging
from copy import deepcopy
from agentlib.utils import custom_injection, load_config
from agentlib.core.errors import ConfigurationError
from flexibility_quantification.data_structures.flexquant import FlexQuantConfig, \
    FlexibilityIndicatorConfig, FlexibilityMarketConfig
import flexibility_quantification.data_structures.globals as glbs
from flexibility_quantification.data_structures.mpcs import BaseMPCData, BaselineMPCData
import flexibility_quantification.utils.config_management as cmng
from flexibility_quantification.modules.flexibility_indicator import \
    FlexibilityIndicatorModuleConfig
from flexibility_quantification.modules.flexibility_market import FlexibilityMarketModuleConfig
from agentlib_mpc.modules.mpc_full import MPCConfig
from agentlib_mpc.data_structures.mpc_datamodels import MPCVariable
from agentlib.core.agent import AgentConfig
from agentlib.core.module import BaseModuleConfig
import ast
import atexit
import os
from itertools import zip_longest
from typing import Union, List
from pydantic import FilePath
from pathlib import Path


class FlexAgentGenerator:
    baseline_mpc_module_config: MPCConfig
    pos_flex_mpc_module_config: MPCConfig
    neg_flex_mpc_module_config: MPCConfig
    indicator_module_config: FlexibilityIndicatorModuleConfig
    market_module_config: FlexibilityMarketModuleConfig

    def __init__(self, flex_config: Union[str, FilePath, FlexQuantConfig],
                 mpc_agent_config: Union[str, FilePath, AgentConfig]):
        # load configs
        self.flex_config = load_config.load_config(flex_config, config_type=FlexQuantConfig)
        # baseline agent
        self.baseline_mpc_agent_config = load_config.load_config(mpc_agent_config,
                                                                 config_type=AgentConfig)
        # pos agent
        self.pos_flex_mpc_agent_config = load_config.load_config(mpc_agent_config,
                                                                 config_type=AgentConfig)
        # neg agent
        self.neg_flex_mpc_agent_config = load_config.load_config(mpc_agent_config,
                                                                 config_type=AgentConfig)
        # baseline module
        self.baseline_mpc_module_config = cmng.get_module(config=self.baseline_mpc_agent_config,
                                                          module_type=cmng.MPC_CONFIG_TYPE)
        # pos module
        self.pos_flex_mpc_module_config = cmng.get_module(config=self.pos_flex_mpc_agent_config,
                                                          module_type=cmng.MPC_CONFIG_TYPE)
        # neg module
        self.neg_flex_mpc_module_config = cmng.get_module(config=self.neg_flex_mpc_agent_config,
                                                          module_type=cmng.MPC_CONFIG_TYPE)
        # load indicator config
        self.indicator_config = load_config.load_config(self.flex_config.indicator_config,
                                                        config_type=FlexibilityIndicatorConfig)
        # load indicator module config
        self.indicator_agent_config = load_config.load_config(
            self.flex_config.indicator_config.agent_config,
            config_type=AgentConfig)
        self.indicator_module_config = cmng.get_module(config=self.indicator_agent_config,
                                                       module_type=cmng.INDICATOR_CONFIG_TYPE)
        # load market config
        if self.flex_config.market_config:
            self.market_config = load_config.load_config(self.flex_config.market_config,
                                                         config_type=FlexibilityMarketConfig)
            # load market module config
            self.market_agent_config = load_config.load_config(self.market_config.agent_config,
                                                               config_type=AgentConfig)
            self.market_module_config = cmng.get_module(config=self.market_agent_config,
                                                        module_type=cmng.MARKET_CONFIG_TYPE)
        else:
            self.flex_config.market_time = 0

    def generate_flex_agents(self) -> [MPCConfig, MPCConfig, MPCConfig,
                                       FlexibilityIndicatorModuleConfig,
                                       FlexibilityMarketModuleConfig]:
        """Generates the configs and the python module for the flexibility agents.
        Power variable must be defined in the mpc config.

        """
        # TODO: Add validation (e.g. price is the same for indicator and mpc_config).
        #  Otherwise throw warning or make assumptions
        if self.flex_config.baseline_config_generator_data.power_variable not in [output.name for
                                                                                  output in
                                                                                  self.baseline_mpc_module_config.outputs]:
            raise ConfigurationError("Given power variable is not defined in baseline mpc config.")
        # extract the original optiization backend
        baseline_opt_backend = self.baseline_mpc_module_config.optimization_backend["model"]["type"]

        # adapt modules to include necessary communication variables
        baseline_mpc_config = self.adapt_mpc_module_config(
            module_config=self.baseline_mpc_module_config,
            mpc_dataclass=self.flex_config.baseline_config_generator_data)
        pf_mpc_config = self.adapt_mpc_module_config(module_config=self.pos_flex_mpc_module_config,
                                                     mpc_dataclass=self.flex_config.shadow_mpc_config_generator_data.pos_flex)
        nf_mpc_config = self.adapt_mpc_module_config(module_config=self.neg_flex_mpc_module_config,
                                                     mpc_dataclass=self.flex_config.shadow_mpc_config_generator_data.neg_flex)

        # dump jsons of the agents including the adapted module configs
        self.append_module_and_dump_agent(module=baseline_mpc_config,
                                          agent=self.baseline_mpc_agent_config,
                                          module_type=cmng.MPC_CONFIG_TYPE,
                                          config_name=self.flex_config.baseline_config_generator_data.name_of_created_file)
        self.append_module_and_dump_agent(module=pf_mpc_config,
                                          agent=self.pos_flex_mpc_agent_config,
                                          module_type=cmng.MPC_CONFIG_TYPE,
                                          config_name=self.flex_config.shadow_mpc_config_generator_data.pos_flex.name_of_created_file)
        self.append_module_and_dump_agent(module=nf_mpc_config,
                                          agent=self.neg_flex_mpc_agent_config,
                                          module_type=cmng.MPC_CONFIG_TYPE,
                                          config_name=self.flex_config.shadow_mpc_config_generator_data.neg_flex.name_of_created_file)

        # same for indicator and market
        indicator_module_config = self.adapt_indicator_config(
            module_config=self.indicator_module_config)
        self.append_module_and_dump_agent(module=indicator_module_config,
                                          agent=self.indicator_agent_config,
                                          module_type=cmng.INDICATOR_CONFIG_TYPE,
                                          config_name=self.flex_config.indicator_config.name_of_created_file)
        if self.flex_config.market_config:
            market_module_config = self.adapt_market_config(module_config=self.market_module_config)
            self.append_module_and_dump_agent(module=market_module_config,
                                              agent=self.market_agent_config,
                                              module_type=cmng.MARKET_CONFIG_TYPE,
                                              config_name=self.market_config.name_of_created_file)

        # generate python files for the shadow mpcs
        self._generate_flex_model_definition(
            created_file_name=os.path.join(self.flex_config.path_to_flex_files,
                                           self.flex_config.baseline_config_generator_data.created_flex_mpcs_file),
            casadi_model_data=baseline_opt_backend,
            controls=[control.name for control in self.baseline_mpc_module_config.controls],
            power_variable=self.flex_config.baseline_config_generator_data.power_variable,
            weights=[weight.dict() for weight in
                     self.flex_config.shadow_mpc_config_generator_data.weights],
            pos_flex_cost_func=self.flex_config.shadow_mpc_config_generator_data.pos_flex.flex_cost_function,
            neg_flex_cost_func=self.flex_config.shadow_mpc_config_generator_data.neg_flex.flex_cost_function,
            profile_deviation_weight=self.flex_config.shadow_mpc_config_generator_data.profile_deviation_weight)

        # register the exit function if the corresponding flag is set
        if self.flex_config.delete_files:
            atexit.register(lambda: self._delete_created_files())
        return self.get_config_file_paths()

    def append_module_and_dump_agent(self, module: BaseModuleConfig, agent: AgentConfig,
                                     module_type: str, config_name: str):
        """Appends the given module config to the given agent config and dumps th agent config to a
        json file. The json file is named based on the config_name."""

        # if module is not from the baseline, set a new agent id, based on module id
        if module.type is not self.baseline_mpc_module_config.type:
            agent.id = module.module_id
        # get the module as a dict without default values
        module_dict = cmng.to_dict_and_remove_unnecessary_fields(module=module)
        # write given module to agent config
        for i, agent_module in enumerate(agent.modules):
            if cmng.MODULE_TYPE_DICT[module_type] is cmng.MODULE_TYPE_DICT[agent_module["type"]]:
                agent.modules[i] = module_dict

        # create folder
        Path(self.flex_config.path_to_flex_files).mkdir(parents=True, exist_ok=True)
        # dump agent config
        if agent.modules:
            if self.flex_config.overwrite_files:
                try:
                    Path(os.path.join(self.flex_config.path_to_flex_files, config_name)).unlink()
                except OSError:
                    pass
            with open(os.path.join(self.flex_config.path_to_flex_files, config_name), "w+") as f:
                module_json = agent.model_dump_json(exclude_defaults=True)
                f.write(module_json)
        else:
            logging.error("Provided agent config does not contain any modules.")

    def get_config_file_paths(self) -> List[str]:
        """Returns a list of paths with the created config files"""
        paths = [os.path.join(self.flex_config.path_to_flex_files,
                              self.flex_config.baseline_config_generator_data.name_of_created_file),
                 os.path.join(self.flex_config.path_to_flex_files,
                              self.flex_config.shadow_mpc_config_generator_data.pos_flex.name_of_created_file),
                 os.path.join(self.flex_config.path_to_flex_files,
                              self.flex_config.shadow_mpc_config_generator_data.neg_flex.name_of_created_file),
                 os.path.join(self.flex_config.path_to_flex_files,
                              self.flex_config.indicator_config.name_of_created_file)]
        if self.flex_config.market_config:
            paths.append(os.path.join(self.flex_config.path_to_flex_files,
                                      self.market_config.name_of_created_file))
        return paths

    def _delete_created_files(self):
        """Function to run at exit if the files are to be deleted

        """
        to_be_deleted = self.get_config_file_paths()
        # delete files
        for file in to_be_deleted:
            Path(file).unlink()
        # also delete folder
        Path(self.flex_config.path_to_flex_files).rmdir()

    def adapt_mpc_module_config(self, module_config: MPCConfig,
                                mpc_dataclass: BaseMPCData) -> MPCConfig:
        """Adapts the mpc module config for automated flexibility quantification.
        Things adapted among others are:
        - the file name/path of the mpc config file
        - names of the control variables for the shadow mpcs
        - reduce communicated variables of shadow mpcs to outputs
        - add the power variable to the outputs
        - add the Time variable to the inputs
        - add parameters for the activation and quantification of flexibility

        """
        # allow the module config to be changed
        module_config.model_config["frozen"] = False

        module_config.module_id = mpc_dataclass.module_id

        # append the new weights as parameters to the baseline MPC or update its value
        parameter_dict = {parameter.name: parameter for parameter in module_config.parameters}
        for weight in self.flex_config.shadow_mpc_config_generator_data.weights:
            if weight.name in parameter_dict:
                parameter_dict[weight.name].value = weight.value
            else:
                module_config.parameters.append(weight)

        # set new MPC type
        module_config.type = mpc_dataclass.module_type
        # set new id (needed for plotting)
        module_config.module_id = mpc_dataclass.module_id
        # update optimization backend to use the created mpc files and classes
        module_config.optimization_backend["model"]["type"] = {
            "file": os.path.join(self.flex_config.path_to_flex_files,
                                 mpc_dataclass.created_flex_mpcs_file),
            "class_name": mpc_dataclass.class_name}
        # update results file with suffix
        module_config.optimization_backend["results_file"] = module_config.optimization_backend[
            "results_file"].replace(".csv", mpc_dataclass.results_suffix)
        # add the control signal of the baseline to outputs (used during market time)
        # and as inputs for the shadow mpcs
        if type(mpc_dataclass) is not BaselineMPCData:
            for control in module_config.controls:
                module_config.inputs.append(
                    MPCVariable(name=f"_{control.name}", value=control.value))

            # only communicate outputs for the shadow mpcs
            module_config.shared_variable_fields = ["outputs"]
        else:
            for control in module_config.controls:
                module_config.outputs.append(
                    MPCVariable(name=control.name + mpc_dataclass.full_trajectory_suffix,
                                value=control.value))
        # add outputs for the power variables, for easier handling create a lookup dict
        output_dict = {output.name: output for output in module_config.outputs}
        if self.flex_config.baseline_config_generator_data.power_variable in output_dict:
            output_dict[
                self.flex_config.baseline_config_generator_data.power_variable].alias = mpc_dataclass.power_alias
        else:
            module_config.outputs.append(
                MPCVariable(name=self.flex_config.baseline_config_generator_data.power_variable,
                            alias=mpc_dataclass.power_alias))
        # add inputs for the Time variable as well as extra inputs needed for activation of flex
        module_config.inputs.append(MPCVariable(name="Time",
                                                value=[i * module_config.time_step for i in
                                                       range(module_config.prediction_horizon)]))
        module_config.inputs.extend(mpc_dataclass.config_inputs_appendix)
        # CONFIG_PARAMETERS_APPENDIX only includes dummy values
        # overwrite dummy values with values from flex config and append it to module config
        for var in mpc_dataclass.config_parameters_appendix:
            if var.name in self.flex_config.model_fields:
                var.value = getattr(self.flex_config, var.name)
        module_config.parameters.extend(mpc_dataclass.config_parameters_appendix)

        # freeze the config again
        module_config.model_config["frozen"] = True

        return module_config

    def adapt_indicator_config(self, module_config: FlexibilityIndicatorModuleConfig) \
            -> FlexibilityIndicatorModuleConfig:
        """Adapts the indicator module config for automated flexibility quantification.

        """
        # allow the module config to be changed
        module_config.model_config["frozen"] = False
        for parameter in module_config.parameters:
            if parameter.name == glbs.PREP_TIME:
                parameter.value = self.flex_config.prep_time
            if parameter.name == glbs.MARKET_TIME:
                parameter.value = self.flex_config.market_time
            if parameter.name == glbs.FLEX_EVENT_DURATION:
                parameter.value = self.flex_config.flex_event_duration
            if parameter.name == "time_step":
                parameter.value = self.baseline_mpc_module_config.time_step
            if parameter.name == "prediction_horizon":
                parameter.value = self.baseline_mpc_module_config.prediction_horizon
        if "method" in self.baseline_mpc_module_config.optimization_backend["discretization_options"]:
            module_config.discretization = self.baseline_mpc_module_config.optimization_backend["discretization_options"]["method"]
        module_config.power_unit = self.flex_config.baseline_config_generator_data.power_unit
        module_config.model_config["frozen"] = True
        return module_config

    def adapt_market_config(self, module_config: FlexibilityMarketModuleConfig) \
            -> FlexibilityMarketModuleConfig:
        """Adapts the market module config for automated flexibility quantification.

        """
        # allow the module config to be changed
        module_config.model_config["frozen"] = False
        for field in module_config.__fields__:
            if field in self.market_module_config.__fields__.keys():
                module_config.__setattr__(field, getattr(self.market_module_config, field))
            if field == "time_step":
                module_config.__setattr__(field, self.baseline_mpc_module_config.time_step)
        module_config.model_config["frozen"] = True
        return module_config

    def _generate_flex_model_definition(self, created_file_name, casadi_model_data, controls,
                                        power_variable,
                                        weights, pos_flex_cost_func, neg_flex_cost_func,
                                        profile_deviation_weight, std_cost_func=None):
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
        fname = casadi_model_data["file"]

        model_name = casadi_model_data["class_name"]
        # Extract the config class of the casadi model
        config_class = inspect.get_annotations(custom_injection(casadi_model_data))["config"]
        config_instance = config_class()
        # extract the variables of the model
        model_vars = config_instance.dict(include={"inputs", "parameters", "outputs", "states"})
        # check if power variable is in model outputs  #TODO: move this to a validator
        if all(power_variable not in var["name"] for var in model_vars["outputs"]):
            raise ValueError(
                f"The power variable {power_variable} is not found on the model outputs!")

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
        # add pandas and casadi import to file, if not already there
        import_statement_pd = ast.Import(names=[ast.alias(name='pandas', asname='pd')])
        import_statement_ca = ast.Import(names=[ast.alias(name='casadi', asname='ca')])
        for node in parsed.body:
            if isinstance(node, ast.Import):
                alias_names = [alias.name for alias in node.names]
                alias_asnames = [alias.asname for alias in node.names]
                if "pandas" not in alias_names and "pd" not in alias_asnames:
                    parsed.body.insert(0, import_statement_pd)
                if "casadi" not in alias_names and "ca" not in alias_asnames:
                    parsed.body.insert(0, import_statement_ca)
                break
        else:
            parsed.body.insert(0, import_statement_pd)
            parsed.body.insert(0, import_statement_ca)


        class_ind, class_body = get_element_with_name(parsed, config_name)
        # copy the config to be used for the flex case
        parsed.body.append(deepcopy(parsed.body[class_ind]))
        # Add the new weights and control trajectories to the baseline config class
        for i, body in enumerate(class_body.body):
            if body.target.id == "parameters":
                for weight in weights:
                    body.value.elts.append(ast.parse(
                        f"CasadiParameter(name='{weight['name']}', value=0, unit='-', description='Weight for P in objective function')").body[
                                               0].value)
            if body.target.id == "outputs":
                for control in controls:
                    body.value.elts.append(ast.parse(
                        f"CasadiOutput(name='{control}_full', unit='W', type='pd.Series', value=pd.Series([0]), description='full control output')").body[
                                               0].value)

        # add the flexibility inputs
        for i, body in enumerate(parsed.body[class_ind].body):
            if body.target.id == "inputs":
                body.value.elts.append(ast.parse(
                    "CasadiInput(name='Time', value=0, unit='s', description='time trajectory')").body[
                                           0].value)
                body.value.elts.append(ast.parse(
                    f"CasadiInput(name='_P_external', value=0, unit='W', description='External power profile to be provised')").body[
                                           0].value)
                body.value.elts.append(ast.parse(
                    f"CasadiInput(name='in_provision', value=False, unit='-', description='Flag signaling if the flexibility is in provision')").body[
                                           0].value)
                body.value.elts.append(ast.parse(
                    f"CasadiInput(name='rel_start', value=0, unit='s', description='relative start time of the flexibility event')").body[
                                           0].value)
                body.value.elts.append(ast.parse(
                    f"CasadiInput(name='rel_end', value=0, unit='s', description='relative end time of the flexibility event')").body[
                                           0].value)

        # parse the flex config class
        parsed.body[-1].name = "FlexShadowMPCConfig"
        class_ind, class_body = get_element_with_name(parsed, "FlexShadowMPCConfig")
        # Add the new variables to the class
        for i, body in enumerate(class_body.body):
            if body.target.id == "inputs":
                body.value.elts.append(ast.parse(
                    "CasadiInput(name='Time', value=0, unit='s', description='time trajectory')").body[
                                           0].value)
                for control in controls:
                    # add the control trajectorsy inputs
                    body.value.elts.append(ast.parse(
                        f"CasadiInput(name='_{control}', unit='W', type='pd.Series', value=pd.Series([0]))").body[
                                               0].value)
                body.value.elts.append(
                    ast.parse(f"CasadiInput(name='in_provision', unit='-', value=False)").body[
                        0].value)
            # add the flex variables and the weights
            if body.target.id == "parameters":
                body.value.elts.append(ast.parse(
                    "CasadiParameter(name='prep_time', value=0, unit='s', description='time to switch objective')").body[
                                           0].value)
                body.value.elts.append(ast.parse(
                    "CasadiParameter(name='flex_event_duration', value=0, unit='s', description='time to switch objective')").body[
                                           0].value)
                body.value.elts.append(ast.parse(
                    "CasadiParameter(name='market_time', value=0, unit='s', description='time to switch objective')").body[
                                           0].value)
                for weight in weights:
                    body.value.elts.append(ast.parse(
                        f"CasadiParameter(name='{weight['name']}', value=0, unit='-', description='Weight for P in objective function')").body[
                                               0].value)

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
                raise ValueError(
                    f"Variables in the cost function {missing_args} are not assigned as module variables!")

        obj_flex = ast.parse(f"obj_flex = {pos_flex_cost_func}").body[0]
        check_args(obj_flex, model_vars_list)
        # if there is no standard cost function in config use the one from the baseline system
        if std_cost_func in (None, ""):
            func_ind, func_body = get_element_with_name(class_body, "setup_system")

            for ind in reversed(range(len(func_body.body))):
                # TODO: this implies, that the objective has to be defined at the end of the setup_system
                if isinstance(func_body.body[ind], ast.Assign):
                    std_cost_func = ast.unparse(func_body.body[ind].value)
                    break

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
                if isinstance(func_body.body[ind].targets[0], ast.Attribute) and \
                        func_body.body[ind].targets[0].attr == "constraints":
                    for i, control in enumerate(controls):
                        func_body.body[ind + 2 * i].value.elts.append(
                            ast.parse(f"({control}_neg, self.{control}, {control}_pos)").body[0])
                        func_body.body.insert(
                            ind, ast.parse(
                                f"{control}_pos = ca.if_else(self.Time.sym < self.market_time.sym, self._{control}.sym, self.{control}.ub)").body[
                                0])
                        func_body.body.insert(
                            ind, ast.parse(
                                f"{control}_neg = ca.if_else(self.Time.sym < self.market_time.sym, self._{control}.sym, self.{control}.lb)").body[
                                0])
                    break
        func_body.body.append(obj_flex)
        func_body.body.append(obj_std)
        # append the cost function with the if_else switches
        func_body.body.append(ast.parse(
            "return ca.if_else(self.Time.sym < self.prep_time.sym + self.market_time.sym, obj_std,"
            "ca.if_else(self.Time.sym < (self.prep_time.sym + self.flex_event_duration.sym + self.market_time.sym), obj_flex, obj_std))").body[
                                  0])

        # the same steps for the negative flexibility
        func_ind, func_body = get_element_with_name(parsed.body[-1], "setup_system")

        for ind in reversed(range(len(func_body.body))):
            if isinstance(func_body.body[ind], ast.Assign):
                if isinstance(func_body.body[ind].targets[0], ast.Attribute) and \
                        func_body.body[ind].targets[0].attr == "constraints":
                    for i, control in enumerate(controls):
                        func_body.body[ind + 2 * i].value.elts.append(
                            ast.parse(f"({control}_neg, self.{control}, {control}_pos)").body[0])
                        func_body.body.insert(
                            ind, ast.parse(
                                f"{control}_pos = ca.if_else(self.Time.sym < self.market_time.sym, self._{control}.sym, self.{control}.ub)").body[
                                0])
                        func_body.body.insert(
                            ind, ast.parse(
                                f"{control}_neg = ca.if_else(self.Time.sym < self.market_time.sym, self._{control}.sym, self.{control}.lb)").body[
                                0])
                    break
        obj_flex = ast.parse(f"obj_flex = {neg_flex_cost_func}").body[0]
        check_args(obj_flex, model_vars_list)
        func_body.body.append(obj_flex)
        func_body.body.append(obj_std)

        func_body.body.append(ast.parse(
            "return ca.if_else(self.Time.sym < self.prep_time.sym + self.market_time.sym, obj_std,"
            "ca.if_else(self.Time.sym < (self.prep_time.sym + self.flex_event_duration.sym + self.market_time.sym), obj_flex, obj_std))").body[
                                  0])

        class_ind, class_body = get_element_with_name(parsed, "BaselineMPC")
        func_ind, func_body = get_element_with_name(class_body, "setup_system")

        # set the control trajectories with the respective variables
        for control in controls:
            func_body.body.append(ast.parse(f"self.{control}_full.alg = self.{control}"))
        # add the soft constraint for the provision
        func_body.body.append(obj_std)
        func_body.body.append(ast.parse("return ca.if_else(self.in_provision.sym, "
                                        "ca.if_else(self.Time.sym < self.rel_start.sym, obj_std, "
                                        "ca.if_else(self.Time.sym >= self.rel_end.sym, obj_std, "
                                        f"sum([{profile_deviation_weight}*(self.{power_variable} - self._P_external)**2]))),obj_std)").body[
                                  0])
        # func_body.body.append(ast.parse("return obj_std + ca.if_else(self.in_provision.sym, "
        #                                 "ca.if_else(self.Time.sym < self.rel_start.sym, 0, "
        #                                 "ca.if_else(self.Time.sym >= self.rel_end.sym, 0, "
        #                                 f"sum([{profile_deviation_weight}*(self.{power_variable} - self._P_external)**2]))),0)").body[
        #                           0])

        if self.flex_config.overwrite_files:
            try:
                Path(os.path.join(self.flex_config.path_to_flex_files,
                                  self.flex_config.baseline_config_generator_data.created_flex_mpcs_file)).unlink()
            except OSError:
                pass

        with open(created_file_name, "w+") as f:
            f.write(ast.unparse(parsed))
