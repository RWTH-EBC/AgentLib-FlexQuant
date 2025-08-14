from typing import Dict
from pydantic import Field
from agentlib import AgentVariable
from agentlib_mpc.modules import mpc_full, minlp_mpc
from agentlib_mpc.data_structures.mpc_datamodels import Results
from agentlib_flexquant.data_structures.globals import full_trajectory_suffix


class FlexibilityBaselineMPCConfig(mpc_full.MPCConfig):

    # define an AgentVariable list for the full control trajectory, since use MPCVariable output affects the optimization result
    full_controls: list[AgentVariable] = Field(default=[])


class FlexibilityBaselineMPC(mpc_full.MPC):
    config: FlexibilityBaselineMPCConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # initialize a control mapping dictionary which maps the full control names to the control names
        self._controls_name_mapping: Dict[str, str] = {}

        for full_control in self.config.full_controls:
            # add full_control to the variables dictionary, so that the set function can be applied to it
            self._variables_dict[full_control.name] = full_control
            # fill the mapping dictionary
            self._controls_name_mapping[full_control.name] = full_control.name.replace(full_trajectory_suffix, "")

    def pre_computation_hook(self):
        if self.get("in_provision").value:
            timestep = (self.get("_P_external").value.index[1] -
                        self.get("_P_external").value.index[0])
            self.set("rel_start", self.get("_P_external").value.index[0] -
                     self.env.time)
            # the provision profile gives a value for the start of a time step.
            # For the end of the flex interval add time step!
            self.set("rel_end", self.get("_P_external").value.index[-1] -
                     self.env.time + timestep)

    def set_actuation(self, solution: Results):
        super().set_actuation(solution)
        for full_control in self.config.full_controls:
            # get the corresponding control name
            control = self._controls_name_mapping[full_control.name]
            # set value to full_control
            self.set(full_control.name, solution.df.variable[control].ffill())


class FlexibilityBaselineMINLPMPCConfig(minlp_mpc.MINLPMPCConfig):

    # define an AgentVariable list for the full control trajectory, since use MPCVariable output affects the optimization result
    full_controls: list[AgentVariable] = Field(default=[])


class FlexibilityBaselineMINLPMPC(minlp_mpc.MINLPMPC):
    config: FlexibilityBaselineMINLPMPCConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # initialize a control mapping dictionary which maps the full control names to the control names
        self._controls_name_mapping: Dict[str, str] = {}

        for full_control in self.config.full_controls:
            # add full_control to the variables dictionary, so that the set function can be applied to it
            self._variables_dict[full_control.name] = full_control
            # fill the mapping dictionary
            self._controls_name_mapping[full_control.name] = full_control.name.replace(full_trajectory_suffix, "")

    def pre_computation_hook(self):
        if self.get("in_provision").value:
            timestep = (self.get("_P_external").value.index[1] -
                        self.get("_P_external").value.index[0])
            self.set("rel_start", self.get("_P_external").value.index[0] -
                     self.env.time)
            # the provision profile gives a value for the start of a time step.
            # For the end of the flex interval add time step!
            self.set("rel_end", self.get("_P_external").value.index[-1] -
                     self.env.time + timestep)

    def set_actuation(self, solution: Results):
        super().set_actuation(solution)
        for full_control in self.config.full_controls:
            # get the corresponding control name
            control = self._controls_name_mapping[full_control.name]
            # set value to full_control
            self.set(full_control.name, solution.df.variable[control].ffill())
