import pandas as pd
from agentlib_mpc.modules import mpc_full, minlp_mpc
from flexibility_quantification.utils.data_handling import strip_multi_index, fill_nans, MEAN, INTERPOLATE
from flexibility_quantification.data_structures.globals import (
    full_trajectory_prefix,
    full_trajectory_suffix,
)
from typing import Dict, Union
from agentlib.core.datamodels import AgentVariable


class FlexibilityShadowMPC(mpc_full.MPC):

    config: mpc_full.MPCConfig

    def __init__(self, *args, **kwargs):
        # create instance variable
        self._full_controls: Dict[str, Union[AgentVariable, None]] = {}
        super().__init__(*args, **kwargs)

    # def set_output(self, result_df: pd.DataFrame):
    #
    #     # fill output values at the discretization points if collocation method is used
    #     if 'collocation' in next(iter(self.config.optimization_backend['discretization_options'])):
    #
    #         # store the current state and control value of the casadi model in a dictionary
    #         current_states = dict()
    #         for state in self.var_ref.states:
    #             current_states[state] = self.model.get_state(state).value
    #
    #         # store the current state and control value of the casadi model in a dictionary
    #         current_controls = dict()
    #         for control in self.var_ref.controls:
    #             current_controls[control] = self.model.get_input(control).value
    #
    #         # get state and control values from the mpc optimization result
    #         state_values = result_df.variable[self.var_ref.states]
    #         control_values = result_df.variable[self.var_ref.controls]
    #
    #         for i in range(1, len(state_values)):
    #             # get the integration time
    #             t_start = float(state_values.index[i-1].strip("()").split(", ")[1]) # or a dummy value? is t_start itself used at all, or do we just need inputs at t_start?
    #             t_sample = float(state_values.index[i].strip("()").split(", ")[1])
    #             # only integrate to get outputs at the discretization points
    #             if not t_sample % self.config.time_step:
    #                 # set the state to the one at the last collocation/discretization point
    #                 for j in range(len(self.var_ref.states)):
    #                     self.model.set(self.var_ref.states[j], state_values.iloc[i-1, j])
    #                 # set the control to the one at the last discretization point
    #                 for j in range(len(self.var_ref.controls)):
    #                     control_inx_shift = int(self.config.optimization_backend['discretization_options']['collocation_order'])+1
    #                     self.model.set(self.var_ref.controls[j], control_values.iloc[i-control_inx_shift, j])
    #                 # do the integration
    #                 self.model.do_step(t_start=t_start, t_sample=t_sample)
    #                 # set output at the discretization points
    #                 for output in self.var_ref.outputs:
    #                     result_df.loc[result_df.index[i], ('variable', output)] = self.model.get_output(output).value
    #
    #         # reset the state and control value in the casadi model
    #         for state in self.var_ref.states:
    #             self.model.set(state, current_states[state])
    #
    #         for control in self.var_ref.controls:
    #             self.model.set(control, current_controls[control])
    #
    #     # send the modified output to AgentVariables
    #     super().set_output(result_df)


    def register_callbacks(self):
        for control_var in self.config.controls:
            self.agent.data_broker.register_callback(
                name=f"{full_trajectory_prefix}{control_var.name}{full_trajectory_suffix}",
                alias=f"{full_trajectory_prefix}{control_var.name}{full_trajectory_suffix}",
                callback=self.calc_flex_callback,
            )
        for input_var in self.config.inputs:
            if input_var.name.replace(full_trajectory_prefix, "", 1).replace(
                full_trajectory_suffix, ""
            ) in [control_var.name for control_var in self.config.controls]:
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

        vals = strip_multi_index(inp.value)
        if vals.isna().any():
            vals = fill_nans(series=vals, method=MEAN)

        # the MPC Predictions starts at t=env.now not t=0
        vals.index += self.env.time
        self._full_controls[name].value = vals
        self.set(name, vals)
        # make sure all controls are set
        if all(x.value is not None for x in self._full_controls.values()):
            self.do_step()
            for name in self._full_controls.keys():
                self._full_controls[name].value = None

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
                name=f"{full_trajectory_prefix}{control_var.name}{full_trajectory_suffix}",
                alias=f"{full_trajectory_prefix}{control_var.name}{full_trajectory_suffix}",
                callback=self.calc_flex_callback,
            )
        for input_var in self.config.inputs:
            if input_var.name.replace(full_trajectory_prefix, "", 1).replace(
                full_trajectory_suffix, ""
            ) in [control_var.name for control_var in self.config.controls + self.config.binary_controls]:
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

        vals = strip_multi_index(inp.value)
        if vals.isna().any():
            vals = fill_nans(vals, method=MEAN)

        # the MPC Predictions starts at t=env.now not t=0
        vals.index += self.env.time
        self._full_controls[name].value = vals
        self.set(name, vals)
        # make sure all controls are set
        if all(x.value is not None for x in self._full_controls.values()):
            self.do_step()
            for name in self._full_controls.keys():
                self._full_controls[name].value = None

    def process(self):
        # the shadow mpc should only be run after the results of the baseline are sent
        yield self.env.event()
