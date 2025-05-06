import os
import math
import numpy as np
import pandas as pd
from copy import deepcopy
from typing import Optional

# from agentlib_mpc.data_structures.mpc_datamodels import Results
from pydantic import Field
from collections.abc import Iterable
from agentlib_mpc.utils.analysis import mpc_at_time_step
from agentlib_mpc.modules import mpc_full, minlp_mpc

class FlexibilityBaselineMPCConfig(mpc_full.MPCConfig):

    casadi_sim_time_step: int = Field(default=0, description="Time step for simulation with Casadi simulator. Value is read from FlexQuantConfig")


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

    def set_output(self, solution):
        super().set_output(solution)
        self.sim_flex_model(solution)

    def sim_flex_model(self, solution):

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

