import os
import math
import numpy as np
import pandas as pd
from pydantic import Field
from typing import Dict, Union
from collections.abc import Iterable
from agentlib.core.datamodels import AgentVariable
from agentlib_mpc.modules import mpc_full, minlp_mpc
from flexibility_quantification.utils.data_handling import strip_multi_index, fill_nans, MEAN, INTERPOLATE
from flexibility_quantification.data_structures.globals import (
    full_trajectory_prefix,
    full_trajectory_suffix
)


class FlexibilityShadowMPCConfig(mpc_full.MPCConfig):

    casadi_sim_time_step: int = Field(default=0, description="Time step for simulation with Casadi simulator. Value is read from FlexQuantConfig")


class FlexibilityShadowMPC(mpc_full.MPC):
    config: FlexibilityShadowMPCConfig

    def __init__(self, *args, **kwargs):
        # create instance variable
        self._full_controls: Dict[str, Union[AgentVariable, None]] = {}
        super().__init__(*args, **kwargs)

    def set_output(self, solution):
        super().set_output(solution)
        self.sim_flex_model(solution)

    def sim_flex_model(self,solution):

        # read the high resolution time step
        dt = self.config.casadi_sim_time_step

        # simulate the flex_model if dt is a positive integer and system is not in provision
        if dt > 0 and not self.get("in_provision").value:

            # initialize flex result
            horizon_length = int(self.config.prediction_horizon*(self.config.time_step))
            time_points = math.floor((horizon_length)/dt) + 1  # if int then plus one
            index_first_level = [self.env.now] * time_points
            index_second_level = range(0, horizon_length + dt, dt)
            multi_index = pd.MultiIndex.from_tuples(zip(index_first_level, index_second_level), names=['time_step', 'time'])
            self.flex_results = pd.DataFrame(np.nan, index=multi_index, columns=self.var_ref.outputs)

            # initialize the flex_model for integration
            self.flex_model = type(self.model)(dt=dt)

            # update the value of module inputs and parameters with value from config, since creating a model just reads the value in the model class but not the config
            for inp in self.config.inputs + self.config.parameters:
               if not isinstance(inp.value, Iterable):
                    self.flex_model.set(inp.name, inp.value)

            # read the current optimization result
            result_df = solution.df

            # get control values from the mpc optimization result
            control_values = result_df.variable[self.var_ref.controls].dropna()

            for i in range(1, time_points, 1):
                # set control
                control_num = int(i*dt // self.config.time_step - (i*dt % self.config.time_step == 0))
                for control, value in zip(self.var_ref.controls, control_values.iloc[control_num]):
                    self.flex_model.set(control, value)
                # set t_sample
                t_sample = self.flex_model.dt
                # do integration
                self.flex_model.do_step(t_start=0, t_sample=t_sample)
                # save output
                for output in self.var_ref.outputs:
                    self.flex_results.loc[(self.env.now, t_sample*i), output] = self.flex_model.get_output(output).value

            # set index to the same as mpc result
            self.flex_results.index = multi_index.tolist()

            # clear the casadi simulator result at the first time step
            res_file = self.config.optimization_backend['results_file'].replace('mpc', 'mpc_sim')
            if self.env.now == 0:
                try:
                    os.remove(res_file)
                except:
                    pass

            # save results
            if not os.path.exists(res_file):
                self.flex_results.to_csv(res_file)
            else:
                self.flex_results.to_csv(res_file, mode='a', header=False)


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
