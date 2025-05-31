from typing import List
from pydantic import model_validator, Field
from agentlib import AgentVariable
from agentlib_mpc.modules import mpc_full, minlp_mpc
from agentlib_mpc.data_structures.mpc_datamodels import Results, MPCVariable
from flexibility_quantification.data_structures.globals import full_trajectory_suffix, base_suffix


class FlexibilityBaselineMPCConfig(mpc_full.MPCConfig):

    full_controls: List[AgentVariable] = Field(default=[])

    @model_validator(mode='after')
    def init_full_controls(cls, model):
        '''fill the full_controls list if it's empty
        '''
        if not model.full_controls:
            for control in model.controls:
                model.full_controls.append(AgentVariable(name=control.name+full_trajectory_suffix,
                                                         alias=control.name+full_trajectory_suffix+base_suffix,
                                                         shared=True))
            return model


class FlexibilityBaselineMPC(mpc_full.MPC):
    config: FlexibilityBaselineMPCConfig

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
            # add full_control to the variables dictionary, so that the set function can be applied to it
            self._variables_dict[full_control.name] = full_control
            # get the corresponding control name
            control = full_control.name.replace(full_trajectory_suffix, "")
            # set value to full_control
            self.set(full_control.name, solution.df.variable[control].dropna())


class FlexibilityBaselineMINLPMPC(minlp_mpc.MINLPMPC):
    config: minlp_mpc.MINLPMPCConfig

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
