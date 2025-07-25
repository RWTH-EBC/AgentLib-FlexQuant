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
from flexibility_quantification.data_structures import globals as glbs

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
            self.set("rel_start", self.get("_P_external").value.index[0] -
                     self.env.time)
            # the provision profile gives a value for the start of a time step.
            # For the end of the flex interval add time step!
            self.set("rel_end", self.get("_P_external").value.index[-1] -
                     self.env.time)

    def set_output(self, solution):
        """Takes the solution from optimization backend and sends it to AgentVariables."""
        # Output must be defined in the conig as "type"="pd.Series"
        if not self.config.set_outputs:
            return
        self.logger.info("Sending optimal output values to data_broker.")

        # simulate with the casadi simulator
        self.sim_flex_model(solution)

        df = solution.df
        if hasattr(self, "flex_results"):
            storage_variable_name = next(var.name for var in self.variables if var.alias == glbs.STORED_ENERGY_ALIAS_BASE)
            for output in self.var_ref.outputs:
                if output not in [self.config.power_variable_name, storage_variable_name]:
                    series = df.variable[output]
                    self.set(output, series)
            # send the power and storage variable value from simulation results
            upsampled_output_power = self.flex_results[self.config.power_variable_name]
            upsampled_output_storage = self.flex_results[storage_variable_name]
            self.set(self.config.power_variable_name, upsampled_output_power)
            self.set(storage_variable_name, upsampled_output_storage.dropna())
        else:
            for output in self.var_ref.outputs:
                series = df.variable[output]
                self.set(output, series)

    def sim_flex_model(self, solution):
        '''simulate the flex model over the preditcion horizon and save results'''

        # return if sim_time_step is not a positive integer and system is in provision
        if not (self.config.casadi_sim_time_step > 0 and not self.get("in_provision").value):
            return

        # read the defined simulation time step
        sim_time_step = self.config.casadi_sim_time_step
        mpc_time_step = self.config.time_step

        # set the horizon length and the number of simulation steps
        total_horizon_time = int(self.config.prediction_horizon * self.config.time_step)
        n_simulation_steps = math.ceil(total_horizon_time / sim_time_step)

        # read the current optimization result
        result_df = solution.df

        # initialize the flex sim results Dataframe
        self._initialize_flex_results(n_simulation_steps, total_horizon_time, sim_time_step, result_df)

        # Update model parameters and initial states
        self._update_model_parameters()
        self._update_initial_states(result_df)

        # Run simulation
        self._run_simulation(n_simulation_steps, sim_time_step, mpc_time_step, result_df, total_horizon_time)

        # set index of flex results to the same as mpc result
        store_results_df = self.flex_results.copy(deep=True)
        store_results_df.index = self.flex_results.index.tolist()

        # save results
        if not os.path.exists(self.res_file_flex):
            store_results_df.to_csv(self.res_file_flex)
        else:
            store_results_df.to_csv(self.res_file_flex, mode='a', header=False)

        # set the flex results format same as mpc result while updating Agengvariable
        self.flex_results.index = self.flex_results.index.get_level_values(1)

    def _initialize_flex_results(self, n_simulation_steps, horizon_length, sim_time_step, result_df):
        '''Initialize the flex results dataframe with the correct dimension and index and fill with existing results from optimization'''

        # create MultiIndex for collocation points
        index_coll = pd.MultiIndex.from_arrays(
            [[self.env.now] * len(result_df.index), result_df.index],
            names=['time_step', 'time']
            # Match the names with multi_index but note they're reversed
        )
        # create Multiindex for full simulation sample times
        index_full_sample = pd.MultiIndex.from_tuples(
            zip([self.env.now] * (n_simulation_steps + 1), range(0, horizon_length + sim_time_step, sim_time_step)),
            names=['time_step', 'time'])
        # merge indexes
        new_index = index_coll.union(index_full_sample).sort_values()
        # initialize the flex results
        self.flex_results = pd.DataFrame(np.nan, index=new_index,
                                         columns=self.var_ref.outputs)

        # Get the optimization outputs and create a series for fixed optimization outputs with the correct MultiIndex format
        opti_outputs = result_df.variable[self.config.power_variable_name]
        fixed_opti_output = pd.Series(
            opti_outputs.values,
            index=index_coll,
        )
        # fill the output value at the time step where it already exists in optimization output
        for idx in fixed_opti_output.index:
            if idx in self.flex_results.index:
                self.flex_results.loc[idx, self.config.power_variable_name] = fixed_opti_output[idx]

    def _update_model_parameters(self):
        '''update the value of module parameters with value from config,
           since creating a model just reads the value in the model class but not the config
        '''

        for par in self.config.parameters:
            self.flex_model.set(par.name, par.value)

    def _update_initial_states(self, result_df):
        '''set the initial value of states'''

        # get state values from the mpc optimization result
        state_values = result_df.variable[self.var_ref.states]
        # update state values with last measurement
        for state, value in zip(self.var_ref.states, state_values.iloc[0]):
            self.flex_model.set(state, value)

    def _run_simulation(self, n_simulation_steps, sim_time_step, mpc_time_step, result_df, total_horizon_time):
        '''simulate with flex model over the prediction horizon'''

        # get control and input values from the mpc optimization result
        control_values = result_df.variable[self.var_ref.controls].dropna()
        input_values = result_df.parameter[self.var_ref.inputs].dropna()

        # For each simulation step, determine which MPC interval we're in
        current_control_idx = 0
        last_control_time = 0

        control_dict = {}

        for i in range(0, n_simulation_steps):
            current_sim_time = i * sim_time_step

            # Check if the control values need to be updated
            if current_sim_time >= last_control_time + mpc_time_step and current_control_idx < len(control_values) - 1:
                current_control_idx += 1
                last_control_time += mpc_time_step

            # Apply control and input values from the appropriate MPC step
            for control, value in zip(self.var_ref.controls, control_values.iloc[current_control_idx]):
                self.flex_model.set(control, value)
            control_dict[current_sim_time] = value

            for input_var, value in zip(self.var_ref.inputs, input_values.iloc[current_control_idx]):
                self.flex_model.set(input_var, value)

            # do integration
            # reduce the simultion time step so that the total horizon time will not be exceeded
            if current_sim_time + sim_time_step <= total_horizon_time:
                t_sample = sim_time_step
            else:
                t_sample = total_horizon_time - current_sim_time
            self.flex_model.do_step(t_start=0, t_sample=t_sample)

            # save output
            for output in self.var_ref.outputs:
                self.flex_results.loc[(
                    self.env.now, current_sim_time + t_sample), output] = self.flex_model.get_output(
                    output).value


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

