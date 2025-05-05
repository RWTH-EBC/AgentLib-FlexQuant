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
    power_variable_name: str = Field(default=None, description="Name of the power variable in the baseline mpc model.")


class FlexibilityShadowMPC(mpc_full.MPC):
    config: FlexibilityShadowMPCConfig

    def __init__(self, *args, **kwargs):
        # create instance variable
        self._full_controls: Dict[str, Union[AgentVariable, None]] = {}
        super().__init__(*args, **kwargs)
        # clear the casadi simulator result at the first time step
        self.res_file_flex = self.config.optimization_backend['results_file'].replace('mpc', 'mpc_sim')
        # initialize the flex_model for integration
        self.flex_model = type(self.model)(dt=self.config.casadi_sim_time_step)
        try:
            os.remove(self.res_file_flex)
        except:
            pass

    def set_output(self, solution):
        """Takes the solution from optimization backend and sends it to AgentVariables."""
        # Output must be defined in the conig as "type"="pd.Series"
        if not self.config.set_outputs:
            return
        self.logger.info("Sending optimal output values to data_broker.")
        df = solution.df
        self.sim_flex_model(solution)
        if hasattr(self, "flex_results"):
            for output in self.var_ref.outputs:
                if not output == self.config.power_variable_name:
                    series = df.variable[output]
                    self.set(output, series)
            upsampled_output = self.flex_results[self.config.power_variable_name]
            self.set(self.config.power_variable_name, upsampled_output)
        else:
            for output in self.var_ref.outputs:
                series = df.variable[output]
                self.set(output, series)

    def sim_flex_model(self, solution):
        # read the high resolution time step
        dt = self.config.casadi_sim_time_step
        mpc_time_step = self.config.time_step

        # simulate the flex_model if dt is a positive integer and system is not in provision
        if dt > 0 and not self.get("in_provision").value:

            # initialize flex result
            horizon_length = int(self.config.prediction_horizon * (self.config.time_step))
            n_simulation_steps = math.floor((horizon_length) / dt)  # if int then plus one
            index_first_level = [self.env.now] * (n_simulation_steps + 1)

            # TODO: celanup + check with newer agentlib version (eval)
            # create collocation index
            parsed_tuples = []
            for idx in solution.df.index:
                parsed_tuples.append(eval(idx))
            # Create a proper MultiIndex from the parsed tuples
            index_coll = pd.MultiIndex.from_tuples(
                parsed_tuples,
                names=['time_step', 'time']
                # Match the names with multi_index but note they're reversed
            )
            # create index for full sample times
            index_full_sample = pd.MultiIndex.from_tuples(
                zip(index_first_level, range(0, horizon_length + dt, dt)),
                names=['time_step', 'time'])
            # merge indexes
            new_index = index_coll.union(index_full_sample).sort_values()
            self.flex_results = pd.DataFrame(np.nan, index=new_index,
                                             columns=self.var_ref.outputs)
            # Get the optimization outputs
            opti_outputs = solution.df.variable[self.config.power_variable_name]
            # Fix the index on opti_outputs to match the structure
            parsed_opti_tuples = []
            for idx in opti_outputs.index:
                # Check if idx is already a tuple or needs parsing
                if isinstance(idx, str):
                    parsed_opti_tuples.append(eval(idx))
                else:
                    parsed_opti_tuples.append(idx)
            # Create a series with the correct MultiIndex format
            fixed_series = pd.Series(
                opti_outputs.values,
                index=pd.MultiIndex.from_tuples(parsed_opti_tuples,
                                                names=['time_step', 'time'])
            )

            for idx in fixed_series.index:
                if idx in self.flex_results.index:
                    self.flex_results.loc[idx, self.config.power_variable_name] = fixed_series[idx]

            # update the value of module inputs and parameters with value from config, since creating a model just reads the value in the model class but not the config
            for inp in self.config.inputs + self.config.parameters:
                if not isinstance(inp.value, Iterable):
                    self.flex_model.set(inp.name, inp.value)

            # read the current optimization result
            result_df = solution.df

            # get electrical power at collocation points
            power_coll = result_df.variable[self.config.power_variable_name]

            # get control values from the mpc optimization result (only at main timesteps, not collocation points)
            # Filter out collocation points - keep only the main timestep indices
            collocation_order = int(
                self.config.optimization_backend['discretization_options'][
                    'collocation_order'])
            n_points_per_interval = collocation_order + 1

            # Extract only the main control points (skip collocation points)
            main_indices = range(0, len(result_df), n_points_per_interval)
            control_values = result_df.variable[self.var_ref.controls].iloc[
                main_indices]
            input_values = result_df.parameter[self.var_ref.inputs].iloc[main_indices]

            # get state values from the mpc optimization result
            state_values = result_df.variable[self.var_ref.states]
            # update state values with last measurement
            for state, value in zip(self.var_ref.states, state_values.iloc[0]):
                self.flex_model.set(state, value)

            # For each simulation step, determine which MPC interval we're in
            current_control_idx = 0
            last_control_time = 0

            control_dict = {}

            for i in range(0, n_simulation_steps):
                current_sim_time = i * dt

                # Check if we need to update the control values
                if current_sim_time >= last_control_time + mpc_time_step and current_control_idx < len(
                        control_values) - 1:
                    current_control_idx += 1
                    last_control_time += mpc_time_step

                # Apply control and input values from the appropriate MPC step
                for control, value in zip(self.var_ref.controls,
                                          control_values.iloc[current_control_idx]):
                    self.flex_model.set(control, value)
                control_dict[current_sim_time] = value

                for input_var, value in zip(self.var_ref.inputs,
                                            input_values.iloc[current_control_idx]):
                    self.flex_model.set(input_var, value)

                # set t_sample
                t_sample = self.flex_model.dt

                # do integration
                self.flex_model.do_step(t_start=0, t_sample=t_sample)
                # save output
                for output in self.var_ref.outputs:
                    self.flex_results.loc[(
                        self.env.now, t_sample * (i + 1)), output] = self.flex_model.get_output(
                        output).value

            # set index to the same as mpc result
            store_results_df = self.flex_results.copy(deep=True)
            store_results_df.index = new_index.tolist()

            # save results
            if not os.path.exists(self.res_file_flex):
                store_results_df.to_csv(self.res_file_flex)
            else:
                store_results_df.to_csv(self.res_file_flex, mode='a', header=False)


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
