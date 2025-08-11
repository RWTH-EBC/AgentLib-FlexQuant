from agentlib_mpc.modules import mpc_full, minlp_mpc
from agentlib_flexquant.utils.data_handling import strip_multi_index, fill_nans, MEAN, INTERPOLATE
from agentlib_flexquant.data_structures.globals import full_trajectory_suffix
from typing import Dict, Union
from agentlib.core.datamodels import AgentVariable


class FlexibilityShadowMPC(mpc_full.MPC):

    config: mpc_full.MPCConfig

    def __init__(self, *args, **kwargs):
        # create instance variable
        self._full_controls: Dict[str, Union[AgentVariable, None]] = {}
        super().__init__(*args, **kwargs)

    def register_callbacks(self):
        for control_var in self.config.controls:
            self.agent.data_broker.register_callback(
                name=f"{control_var.name+full_trajectory_suffix}",
                alias=f"{control_var.name+full_trajectory_suffix}",
                callback=self.calc_flex_callback,
            )
        for input_var in self.config.inputs:
            adapted_name = input_var.name.replace(full_trajectory_suffix, "")
            if adapted_name in [control_var.name for control_var in self.config.controls]:
                self._full_controls[input_var.name] = input_var

        super().register_callbacks()

    def calc_flex_callback(self, inp, name):
        """set the control trajectories before calculating the flexibility offer.
        self.model should account for flexibility in its cost function

        """
        # during provision dont calculate flex
        if self.get("in_provision").value:
            return

        # do not trigger callback on self set variables
        if self.agent.config.id == inp.source.agent_id:
            return

        # get the value of the input and reformat index
        vals = strip_multi_index(inp.value)
        # the MPC Predictions starts at t=env.now not t=0
        vals.index += self.env.time
        # update value in the mapping dictionary
        self._full_controls[name].value = vals
        # update the Agentvariable
        self.set(name, vals)
        # make sure all controls are set
        if all(x.value is not None for x in self._full_controls.values()):
            self.do_step()
            # set the full controls to None
            for key_name in self._full_controls.keys():
                self._full_controls[key_name].value = None

    def process(self):
        # the shadow mpc should only be run after the results of the baseline are sent
        yield self.env.event()


class FlexibilityShadowMINLPMPC(minlp_mpc.MINLPMPC):

    config: minlp_mpc.MINLPMPCConfig

    def __init__(self, *args, **kwargs):
        # create instance variable
        self._full_controls: Dict[str, Union[AgentVariable, None]] = {}
        super().__init__(*args, **kwargs)

    def register_callbacks(self):
        for control_var in self.config.controls + self.config.binary_controls:
            self.agent.data_broker.register_callback(
                name=f"{control_var.name}{full_trajectory_suffix}",
                alias=f"{control_var.name}{full_trajectory_suffix}",
                callback=self.calc_flex_callback,
            )
        for input_var in self.config.inputs:
            adapted_name = input_var.name.replace(full_trajectory_suffix, "")
            if adapted_name in [control_var.name for control_var in self.config.controls + self.config.binary_controls]:
                self._full_controls[input_var.name] = input_var

        super().register_callbacks()

    def calc_flex_callback(self, inp, name):
        """set the control trajectories before calculating the flexibility offer.
        self.model should account for flexibility in its cost function

        """
        # during provision dont calculate flex
        if self.get("in_provision").value:
            return

        # do not trigger callback on self set variables
        if self.agent.config.id == inp.source.agent_id:
            return

        # get the value of the input and reformat index
        vals = strip_multi_index(inp.value)
        # the MPC Predictions starts at t=env.now not t=0
        vals.index += self.env.time
        # update value in the mapping dictionary
        self._full_controls[name].value = vals
        # update the Agentvariable
        self.set(name, vals)
        # update the value of the variable in the model if we want to limit the binary control in the market time during optimization
        # self.model.set(name, vals)
        # make sure all controls are set
        if all(x.value is not None for x in self._full_controls.values()):
            self.do_step()
            # set the full controls to None
            for key_name in self._full_controls.keys():
                self._full_controls[key_name].value = None

    def process(self):
        # the shadow mpc should only be run after the results of the baseline are sent
        yield self.env.event()
