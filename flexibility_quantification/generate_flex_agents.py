import inspect
import logging
from copy import deepcopy
import pandas as pd
from agentlib.utils import custom_injection, load_config
from agentlib.core.errors import ConfigurationError
from flexibility_quantification.data_structures.flexquant import (
    FlexQuantConfig,
    FlexibilityIndicatorConfig,
    FlexibilityMarketConfig,
)
import flexibility_quantification.data_structures.globals as glbs
from flexibility_quantification.data_structures.mpcs import BaseMPCData, BaselineMPCData
import flexibility_quantification.utils.config_management as cmng
from flexibility_quantification.modules.flexibility_indicator import (
    FlexibilityIndicatorModuleConfig,
)
from flexibility_quantification.modules.flexibility_market import (
    FlexibilityMarketModuleConfig,
)
from agentlib_mpc.modules.mpc_full import BaseMPCConfig
from agentlib_mpc.data_structures.mpc_datamodels import MPCVariable
from agentlib_mpc.models.casadi_model import CasadiModelConfig
from agentlib.core.agent import AgentConfig
from agentlib.core.module import BaseModuleConfig
from agentlib.core.datamodels import AgentVariable
import ast
import atexit
import os
from typing import Union, List
from pydantic import FilePath
from pathlib import Path
import black


class FlexAgentGenerator:
    orig_mpc_module_config: BaseMPCConfig
    baseline_mpc_module_config: BaseMPCConfig
    pos_flex_mpc_module_config: BaseMPCConfig
    neg_flex_mpc_module_config: BaseMPCConfig
    indicator_module_config: FlexibilityIndicatorModuleConfig
    market_module_config: FlexibilityMarketModuleConfig

    def __init__(
        self,
        flex_config: Union[str, FilePath, FlexQuantConfig],
        mpc_agent_config: Union[str, FilePath, AgentConfig],
    ):
        if isinstance(flex_config, str or FilePath):
            self.flex_config_file_name = os.path.basename(flex_config)
        else:
            # provide default name for json
            self.flex_config_file_name = "flex_config.json"
        # load configs
        self.flex_config = load_config.load_config(
            flex_config, config_type=FlexQuantConfig
        )

        # original mpc agent
        self.orig_mpc_agent_config = load_config.load_config(
            mpc_agent_config, config_type=AgentConfig
        )
        # baseline agent
        self.baseline_mpc_agent_config = self.orig_mpc_agent_config.__deepcopy__()
        # pos agent
        self.pos_flex_mpc_agent_config = self.orig_mpc_agent_config.__deepcopy__()
        # neg agent
        self.neg_flex_mpc_agent_config = self.orig_mpc_agent_config.__deepcopy__()

        # original mpc module
        self.orig_mpc_module_config = cmng.get_module(
            config=self.orig_mpc_agent_config,
            module_type=cmng.get_orig_module_type(self.orig_mpc_agent_config),
        )
        # baseline module
        self.baseline_mpc_module_config = cmng.get_module(
            config=self.baseline_mpc_agent_config,
            module_type=cmng.get_orig_module_type(self.orig_mpc_agent_config),
        )
        # convert to flexquant Module Config
        self.baseline_mpc_module_config = cmng.get_flex_mpc_module_config(agent_config=self.baseline_mpc_agent_config, module_config=self.baseline_mpc_module_config,
                                                                          module_type=self.flex_config.baseline_config_generator_data.module_types[self.baseline_mpc_module_config.type])
        # pos module
        self.pos_flex_mpc_module_config = cmng.get_module(
            config=self.pos_flex_mpc_agent_config,
            module_type=cmng.get_orig_module_type(self.orig_mpc_agent_config),
        )
        # neg module
        self.neg_flex_mpc_module_config = cmng.get_module(
            config=self.neg_flex_mpc_agent_config,
            module_type=cmng.get_orig_module_type(self.orig_mpc_agent_config),
        )
        # load indicator config
        self.indicator_config = load_config.load_config(
            self.flex_config.indicator_config, config_type=FlexibilityIndicatorConfig
        )
        # load indicator module config
        self.indicator_agent_config = load_config.load_config(
            self.flex_config.indicator_config.agent_config, config_type=AgentConfig
        )
        self.indicator_module_config = cmng.get_module(
            config=self.indicator_agent_config, module_type=cmng.INDICATOR_CONFIG_TYPE
        )
        # load market config
        if self.flex_config.market_config:
            self.market_config = load_config.load_config(
                self.flex_config.market_config, config_type=FlexibilityMarketConfig
            )
            # load market module config
            self.market_agent_config = load_config.load_config(
                self.market_config.agent_config, config_type=AgentConfig
            )
            self.market_module_config = cmng.get_module(
                config=self.market_agent_config, module_type=cmng.MARKET_CONFIG_TYPE
            )
        else:
            self.flex_config.market_time = 0

    def generate_flex_agents(
        self,
    ) -> [
        BaseMPCConfig,
        BaseMPCConfig,
        BaseMPCConfig,
        FlexibilityIndicatorModuleConfig,
        FlexibilityMarketModuleConfig,
    ]:
        """Generates the configs and the python module for the flexibility agents.
        Power variable must be defined in the mpc config.

        """

        # adapt modules to include necessary communication variables and dump jsons of the agents including the adapted module configs
        indicator_module_config = self.adapt_indicator_config(
            module_config=self.indicator_module_config
        )
        self.append_module_and_dump_agent(
            module=indicator_module_config,
            agent=self.indicator_agent_config,
            module_type=cmng.INDICATOR_CONFIG_TYPE,
            config_name=self.flex_config.indicator_config.name_of_created_file,
        )
        if self.flex_config.market_config:
            market_module_config = self.adapt_market_config(
                module_config=self.market_module_config
            )
            self.append_module_and_dump_agent(
                module=market_module_config,
                agent=self.market_agent_config,
                module_type=cmng.MARKET_CONFIG_TYPE,
                config_name=self.market_config.name_of_created_file,
            )

        # check if the power variable exists in the mpc config
        if self.flex_config.baseline_config_generator_data.power_variable not in [
            output.name for output in self.baseline_mpc_module_config.outputs
        ]:
            raise ConfigurationError(
                f"Given power variable {self.flex_config.baseline_config_generator_data.power_variable} is not defined as output in baseline mpc config."
            )
        # check if the comfort variable exists in the mpc slack variables
        if self.flex_config.baseline_config_generator_data.comfort_variable:
            file_path = self.baseline_mpc_module_config.optimization_backend["model"]["type"]["file"]
            class_name = self.baseline_mpc_module_config.optimization_backend["model"]["type"]["class_name"]
            # Get the class
            dynamic_class = cmng.get_class_from_file(file_path, class_name)
            if self.flex_config.baseline_config_generator_data.comfort_variable not in [
                state.name for state in dynamic_class().states
            ]:
                raise ConfigurationError(
                    f"Given comfort variable {self.flex_config.baseline_config_generator_data.comfort_variable} is not defined as state in baseline mpc config."
                )
        # check if the energy storage variable exists in the mpc config
        if indicator_module_config.correct_costs.enable_energy_costs_correction:
            if indicator_module_config.correct_costs.stored_energy_variable not in [
                output.name for output in self.baseline_mpc_module_config.outputs
            ]:
                raise ConfigurationError(
                    f"The stored energy variable {indicator_module_config.correct_costs.stored_energy_variable} is not defined in baseline mpc config."
                    f"It must be defined in the base MPC model and config as output if the correction of costs is enabled"
                )

        # adapt modules to include necessary communication variables
        baseline_mpc_config = self.adapt_mpc_module_config(
            module_config=self.baseline_mpc_module_config,
            mpc_dataclass=self.flex_config.baseline_config_generator_data,
        )
        pf_mpc_config = self.adapt_mpc_module_config(
            module_config=self.pos_flex_mpc_module_config,
            mpc_dataclass=self.flex_config.shadow_mpc_config_generator_data.pos_flex,
        )
        nf_mpc_config = self.adapt_mpc_module_config(
            module_config=self.neg_flex_mpc_module_config,
            mpc_dataclass=self.flex_config.shadow_mpc_config_generator_data.neg_flex,
        )

        # dump jsons of the agents including the adapted module configs
        self.append_module_and_dump_agent(
            module=baseline_mpc_config,
            agent=self.baseline_mpc_agent_config,
            module_type=cmng.get_orig_module_type(self.orig_mpc_agent_config),
            config_name=self.flex_config.baseline_config_generator_data.name_of_created_file,
        )
        self.append_module_and_dump_agent(
            module=pf_mpc_config,
            agent=self.pos_flex_mpc_agent_config,
            module_type=cmng.get_orig_module_type(self.orig_mpc_agent_config),
            config_name=self.flex_config.shadow_mpc_config_generator_data.pos_flex.name_of_created_file,
        )
        self.append_module_and_dump_agent(
            module=nf_mpc_config,
            agent=self.neg_flex_mpc_agent_config,
            module_type=cmng.get_orig_module_type(self.orig_mpc_agent_config),
            config_name=self.flex_config.shadow_mpc_config_generator_data.neg_flex.name_of_created_file,
        )

        # generate python files for the shadow mpcs
        self._generate_flex_model_definition()

        # save flex config to created flex files
        with open(os.path.join(self.flex_config.path_to_flex_files, self.flex_config_file_name), "w") as f:
            config_json = self.flex_config.model_dump_json(exclude_defaults=True)
            f.write(config_json)

        # register the exit function if the corresponding flag is set
        if self.flex_config.delete_files:
            atexit.register(lambda: self._delete_created_files())
        return self.get_config_file_paths()

    def append_module_and_dump_agent(
        self,
        module: BaseModuleConfig,
        agent: AgentConfig,
        module_type: str,
        config_name: str,
    ):
        """Appends the given module config to the given agent config and dumps the agent config to a
        json file. The json file is named based on the config_name."""

        # if module is not from the baseline, set a new agent id, based on module id
        if module.type is not self.baseline_mpc_module_config.type:
            agent.id = module.module_id
        # get the module as a dict without default values
        module_dict = cmng.to_dict_and_remove_unnecessary_fields(module=module)
        # write given module to agent config
        for i, agent_module in enumerate(agent.modules):
            if (
                cmng.MODULE_TYPE_DICT[module_type]
                is cmng.MODULE_TYPE_DICT[agent_module["type"]]
            ):
                agent.modules[i] = module_dict

        # create folder
        Path(self.flex_config.path_to_flex_files).mkdir(parents=True, exist_ok=True)
        # dump agent config
        if agent.modules:
            if self.flex_config.overwrite_files:
                try:
                    Path(
                        os.path.join(self.flex_config.path_to_flex_files, config_name)
                    ).unlink()
                except OSError:
                    pass
            with open(
                os.path.join(self.flex_config.path_to_flex_files, config_name), "w+"
            ) as f:
                module_json = agent.model_dump_json(exclude_defaults=True)
                f.write(module_json)
        else:
            logging.error("Provided agent config does not contain any modules.")

    def get_config_file_paths(self) -> List[str]:
        """Returns a list of paths with the created config files"""
        paths = [
            os.path.join(
                self.flex_config.path_to_flex_files,
                self.flex_config.baseline_config_generator_data.name_of_created_file,
            ),
            os.path.join(
                self.flex_config.path_to_flex_files,
                self.flex_config.shadow_mpc_config_generator_data.pos_flex.name_of_created_file,
            ),
            os.path.join(
                self.flex_config.path_to_flex_files,
                self.flex_config.shadow_mpc_config_generator_data.neg_flex.name_of_created_file,
            ),
            os.path.join(
                self.flex_config.path_to_flex_files,
                self.flex_config.indicator_config.name_of_created_file,
            ),
        ]
        if self.flex_config.market_config:
            paths.append(
                os.path.join(
                    self.flex_config.path_to_flex_files,
                    self.market_config.name_of_created_file,
                )
            )
        return paths

    def _delete_created_files(self):
        """Function to run at exit if the files are to be deleted"""
        to_be_deleted = self.get_config_file_paths()
        to_be_deleted.append(
            os.path.join(
                self.flex_config.path_to_flex_files,
                self.flex_config_file_name,
            ))
        # delete files
        for file in to_be_deleted:
            Path(file).unlink()
        # also delete folder
        Path(self.flex_config.path_to_flex_files).rmdir()

    def adapt_mpc_module_config(
        self, module_config: BaseMPCConfig, mpc_dataclass: BaseMPCData
    ) -> BaseMPCConfig:
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

        # append the new weights as parameter to the MPC or update its value
        parameter_dict = {
            parameter.name: parameter for parameter in module_config.parameters
        }
        for weight in mpc_dataclass.weights:
            if weight.name in parameter_dict:
                parameter_dict[weight.name].value = weight.value
            else:
                module_config.parameters.append(weight)

        # set new MPC type
        module_config.type = mpc_dataclass.module_types[
            cmng.get_orig_module_type(self.orig_mpc_agent_config)
        ]
        # set new id (needed for plotting)
        module_config.module_id = mpc_dataclass.module_id
        # update optimization backend to use the created mpc files and classes
        module_config.optimization_backend["model"]["type"] = {
            "file": os.path.join(
                self.flex_config.path_to_flex_files,
                mpc_dataclass.created_flex_mpcs_file,
            ),
            "class_name": mpc_dataclass.class_name,
        }
        # update results file with suffix
        module_config.optimization_backend["results_file"] = (
            module_config.optimization_backend["results_file"].replace(
                ".csv", mpc_dataclass.results_suffix
            )
        )
        # change cia backend to custom backend of flexquant
        if module_config.optimization_backend["type"] == "casadi_cia":
            module_config.optimization_backend["type"] = "casadi_cia_cons"
            module_config.optimization_backend["market_time"] = (
                self.flex_config.market_time
            )

        # add the control signal of the baseline as inputs for the shadow mpcs
        if type(mpc_dataclass) is not BaselineMPCData:
            for control in module_config.controls:
                module_config.inputs.append(
                    MPCVariable(
                        name=control.name + glbs.full_trajectory_suffix,
                        value=None,
                        description='only dummy value, true value is a series'
                    )
                )
                control.alias = control.name + glbs.shadow_suffix
            # also include binary controls
            if hasattr(module_config, "binary_controls"):
                for control in module_config.binary_controls:
                    module_config.inputs.append(
                        MPCVariable(
                            name=control.name + glbs.full_trajectory_suffix,
                            value=None,
                            description='only dummy value, true value is a series'
                        )
                    )

            # only communicate outputs for the shadow mpcs
            module_config.shared_variable_fields = ["outputs"]
        else:
            for control in module_config.controls:
                module_config.full_controls.append(AgentVariable(name=control.name+glbs.full_trajectory_suffix,
                                                                 alias=control.name+glbs.full_trajectory_suffix,
                                                                 shared=True))
            if hasattr(module_config, "binary_controls"):
                for binary_controls in module_config.binary_controls:
                    module_config.full_controls.append(AgentVariable(name=binary_controls.name + glbs.full_trajectory_suffix,
                                                                     alias=binary_controls.name + glbs.full_trajectory_suffix,
                                                                     shared=True))
        module_config.set_outputs = True
        # add outputs for the power variables, for easier handling create a lookup dict
        output_dict = {output.name: output for output in module_config.outputs}
        if (
            self.flex_config.baseline_config_generator_data.power_variable
            in output_dict
        ):
            output_dict[
                self.flex_config.baseline_config_generator_data.power_variable
            ].alias = mpc_dataclass.power_alias
        else:
            module_config.outputs.append(
                MPCVariable(
                    name=self.flex_config.baseline_config_generator_data.power_variable,
                    alias=mpc_dataclass.power_alias,
                )
            )
        # add or change alias for stored energy variable
        if self.indicator_module_config.correct_costs.enable_energy_costs_correction:
            output_dict[
                self.indicator_module_config.correct_costs.stored_energy_variable
            ].alias = mpc_dataclass.stored_energy_alias

        # add inputs for the Time variable as well as extra inputs needed for activation of flex
        module_config.inputs.append(
            MPCVariable(
                name="Time",
                value=[
                    i * module_config.time_step
                    for i in range(module_config.prediction_horizon)
                ],
            )
        )
        module_config.inputs.extend(mpc_dataclass.config_inputs_appendix)
        # CONFIG_PARAMETERS_APPENDIX only includes dummy values
        # overwrite dummy values with values from flex config and append it to module config
        for var in mpc_dataclass.config_parameters_appendix:
            if var.name in self.flex_config.model_fields:
                var.value = getattr(self.flex_config, var.name)
            if var.name in self.flex_config.baseline_config_generator_data.model_fields:
                var.value = getattr(self.flex_config.baseline_config_generator_data, var.name)
        module_config.parameters.extend(mpc_dataclass.config_parameters_appendix)

        # freeze the config again
        module_config.model_config["frozen"] = True

        return module_config

    def adapt_indicator_config(
        self, module_config: FlexibilityIndicatorModuleConfig
    ) -> FlexibilityIndicatorModuleConfig:
        """Adapts the indicator module config for automated flexibility quantification."""
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
        # set power unit
        module_config.power_unit = (
            self.flex_config.baseline_config_generator_data.power_unit
        )
        module_config.results_file = Path(
            Path(self.orig_mpc_module_config.optimization_backend["results_file"]).parent,
            self.indicator_config.name_of_created_file.replace(".json", ".csv"),
        )
        module_config.model_config["frozen"] = True
        return module_config

    def adapt_market_config(
        self, module_config: FlexibilityMarketModuleConfig
    ) -> FlexibilityMarketModuleConfig:
        """Adapts the market module config for automated flexibility quantification."""
        # allow the module config to be changed
        module_config.model_config["frozen"] = False
        for field in module_config.__fields__:
            if field in self.market_module_config.__fields__.keys():
                module_config.__setattr__(
                    field, getattr(self.market_module_config, field)
                )
        module_config.results_file = Path(
            Path(self.orig_mpc_module_config.optimization_backend["results_file"]).parent,
            self.market_config.name_of_created_file.replace(".json", ".csv"),
        )
        module_config.model_config["frozen"] = True
        return module_config

    def _generate_flex_model_definition(self):
        """Generates a python module for negative and positive flexibility agents from
        the Baseline MPC model

        """
        from flexibility_quantification.utils.parsing import (
            SetupSystemModifier,
            add_import_to_tree,
        )
        import astor

        output_file = os.path.join(
            self.flex_config.path_to_flex_files,
            self.flex_config.baseline_config_generator_data.created_flex_mpcs_file,
        )
        opt_backend = self.orig_mpc_module_config.optimization_backend["model"]["type"]

        # Extract the config class of the casadi model to check cost functions
        config_class = inspect.get_annotations(custom_injection(opt_backend))["config"]
        config_instance = config_class()
        self.check_variables_in_casadi_config(
            config_instance,
            self.flex_config.shadow_mpc_config_generator_data.neg_flex.flex_cost_function,
        )
        self.check_variables_in_casadi_config(
            config_instance,
            self.flex_config.shadow_mpc_config_generator_data.pos_flex.flex_cost_function,
        )

        # parse mpc python file
        with open(opt_backend["file"], "r") as f:
            source = f.read()
        tree = ast.parse(source)

        # create modifiers for python file
        modifier_base = SetupSystemModifier(
            mpc_data=self.flex_config.baseline_config_generator_data,
            controls=self.baseline_mpc_module_config.controls,
            binary_controls=self.baseline_mpc_module_config.binary_controls if hasattr(self.baseline_mpc_module_config, "binary_controls") else None,
        )
        modifier_pos = SetupSystemModifier(
            mpc_data=self.flex_config.shadow_mpc_config_generator_data.pos_flex,
            controls=self.pos_flex_mpc_module_config.controls,
            binary_controls=self.pos_flex_mpc_module_config.binary_controls if hasattr(self.pos_flex_mpc_module_config, "binary_controls") else None,
        )
        modifier_neg = SetupSystemModifier(
            mpc_data=self.flex_config.shadow_mpc_config_generator_data.neg_flex,
            controls=self.neg_flex_mpc_module_config.controls,
            binary_controls=self.neg_flex_mpc_module_config.binary_controls if hasattr(self.neg_flex_mpc_module_config, "binary_controls") else None,
        )
        # run the modification
        modified_tree_base = modifier_base.visit(deepcopy(tree))
        modified_tree_pos = modifier_pos.visit(deepcopy(tree))
        modified_tree_neg = modifier_neg.visit(deepcopy(tree))
        # combine modifications to one file
        modified_tree = ast.Module(body=[], type_ignores=[])
        modified_tree.body.extend(
            modified_tree_base.body + modified_tree_pos.body + modified_tree_neg.body
        )
        modified_source = astor.to_source(modified_tree)
        # Use black to format the generated code
        formatted_code = black.format_str(modified_source, mode=black.FileMode())

        if self.flex_config.overwrite_files:
            try:
                Path(
                    os.path.join(
                        self.flex_config.path_to_flex_files,
                        self.flex_config.baseline_config_generator_data.created_flex_mpcs_file,
                    )
                ).unlink()
            except OSError:
                pass

        with open(output_file, "w") as f:
            f.write(formatted_code)

    def check_variables_in_casadi_config(self, config: CasadiModelConfig, expr: str):
        """Check if all variables in the expression are defined in the config.

        Args:
            config (CasadiModelConfig): casadi model config.
            expr (str): The expression to check.

        Raises:
            ValueError: If any variable in the expression is not defined in the config.

        """
        variables_in_config = set(config.get_variable_names())
        variables_in_cost_function = set(ast.walk(ast.parse(expr)))
        variables_in_cost_function = {
            node.attr
            for node in variables_in_cost_function
            if isinstance(node, ast.Attribute)
        }
        variables_newly_created = set(
            weight.name
            for weight in self.flex_config.shadow_mpc_config_generator_data.weights
        )
        unknown_vars = (
            variables_in_cost_function - variables_in_config - variables_newly_created
        )
        if unknown_vars:
            raise ValueError(f"Unknown variables in new cost function: {unknown_vars}")
