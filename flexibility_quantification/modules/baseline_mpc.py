from typing import Optional

from agentlib_mpc.modules import mpc_full, minlp_mpc
import pandas as pd

class FlexibilityBaselineMPC(mpc_full.MPC):
    config: mpc_full.MPCConfig
    flex_results: pd.DataFrame

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

    def sim_fex_model(self):
        results = self.flex_model.do_step() # high res power profile
        results.to_csv()
        # results für t_step=60
        # time      P_flex_sim
        #  0        100
        # 180.711        200
        # 760.12214       ..
        # 900           ..
        formatted = self.format_results(results)
        # results für
        # time      P_flex_sim
        #  (0.0, 0)        100
        # (0.0,180.711)        200
        # (0.0,760.12214)      ..
        # (0.0,900)           ..
        self.flex_results = results

    def get_results(self) -> Optional[pd.DataFrame]:
        """Read the results that were saved from the optimization backend and
        returns them as Dataframe.

        Returns:
            (results, stats) tuple of Dataframes.
        """
        results_file = self.optimization_backend.config.results_file
        if results_file is None or not self.optimization_backend.config.save_results:
            self.logger.info("All results were saved .")
            return None
        try:
            result, stat = self.read_results_file(results_file)
            result=result.append(self.formatted)
            result.write_result_file()
            self.warn_for_missed_solves(stat)
            return result
        except FileNotFoundError:
            self.logger.error("Results file %s was not found.", results_file)
            return None

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
    #             t_start = float(state_values.index[i - 1].strip("()").split(", ")[
    #                                 1])  # or a dummy value? is t_start itself used at all, or do we just need inputs at t_start?
    #             t_sample = float(state_values.index[i].strip("()").split(", ")[1])
    #             # only integrate to get outputs at the discretization points
    #             if not t_sample % self.config.time_step:
    #                 # set the state to the one at the last collocation/discretization point
    #                 for j in range(len(self.var_ref.states)):
    #                     self.model.set(self.var_ref.states[j], state_values.iloc[i - 1, j])
    #                 # set the control to the one at the last discretization point
    #                 for j in range(len(self.var_ref.controls)):
    #                     control_inx_shift = int(
    #                         self.config.optimization_backend['discretization_options']['collocation_order']) + 1
    #                     self.model.set(self.var_ref.controls[j], control_values.iloc[i - control_inx_shift, j])
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

