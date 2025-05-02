import os
import math
import numpy as np
import pandas as pd
from copy import deepcopy
from typing import Optional
from pydantic import Field
from collections.abc import Iterable
from agentlib_mpc.utils.analysis import mpc_at_time_step
from agentlib_mpc.modules import mpc_full, minlp_mpc

class FlexibilityBaselineMPCConfig(mpc_full.MPCConfig):

    casadi_sim_time_step: int = Field(default=0, description="Time step for simulation with Casadi simulator. Value is read from FlexQuantConfig")
    power_variable_name: str = Field(default=None, description="Name of the power variable in the baseline mpc model.")


class FlexibilityBaselineMPC(mpc_full.MPC):
    config: FlexibilityBaselineMPCConfig

    def __init__(self, config, agent):
        super().__init__(config, agent)
        # clear the casadi simulator result at the first time step
        self.res_file_flex = self.config.optimization_backend['results_file'].replace('mpc', 'mpc_sim')
        # initialize the flex_model for integration
        self.flex_model = type(self.model)(dt=self.config.casadi_sim_time_step)
        try:
            os.remove(self.res_file_flex)
        except:
            pass


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
        """Takes the solution from optimization backend and sends it to AgentVariables."""
        # Output must be defined in the conig as "type"="pd.Series"
        if not self.config.set_outputs:
            return
        self.logger.info("Sending optimal output values to data_broker.")
        df = solution.df
        for output in self.var_ref.outputs:
            if not output == self.config.power_variable_name:
                series = df.variable[output]
                self.set(output, series)
        self.sim_flex_model(solution)
        upsampled_output = self.flex_results[self.config.power_variable_name]
        self.set(self.config.power_variable_name, upsampled_output)

    def sim_flex_model(self, solution):
        # read the high resolution time step
        dt = self.config.casadi_sim_time_step

        # simulate the flex_model if dt is a positive integer and system is not in provision
        if dt > 0 and not self.get("in_provision").value:

            # initialize flex result
            horizon_length = int(self.config.prediction_horizon*(self.config.time_step))
            time_points = math.floor((horizon_length)/dt) + 1  # if int then plus one
            index_first_level = [self.env.now] * time_points
            multi_index = pd.MultiIndex.from_tuples(zip(index_first_level, range(0,horizon_length+dt,dt)), names=['time_step', 'time'])
            self.flex_results = pd.DataFrame(np.nan, index=multi_index, columns=self.var_ref.outputs)

            # update the value of module inputs and parameters with value from config, since creating a model just reads the value in the model class but not the config
            for inp in self.config.inputs + self.config.parameters:
                if not isinstance(inp.value, Iterable):
                    self.flex_model.set(inp.name, inp.value)

            # read the current optimization result
            result_df = solution.df

            # get control values from the mpc optimization result
            control_values = result_df.variable[self.var_ref.controls]

            # get state values from the mpc optimization result
            state_values = result_df.variable[self.var_ref.states]
            # update state values with last measurement
            for state, value in zip(self.var_ref.states, state_values.iloc[0]):
                self.flex_model.set(state, value)

            # read the collocation order
            collocation_order = int(self.config.optimization_backend['discretization_options']['collocation_order']) + 1

            for i in range(1, time_points, 1):
                # set control
                control_num = int((i*dt // self.config.time_step - (i*dt % self.config.time_step == 0)) * collocation_order)
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
            store_results_df = self.flex_results.copy(deep=True)
            store_results_df.index = multi_index.tolist()

            # save results
            if not os.path.exists(self.res_file_flex):
                store_results_df.to_csv(self.res_file_flex)
            else:
                store_results_df.to_csv(self.res_file_flex, mode='a', header=False)


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

