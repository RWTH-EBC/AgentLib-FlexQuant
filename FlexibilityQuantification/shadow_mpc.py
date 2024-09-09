from agentlib_mpc.modules import mpc_full
from agentlib_mpc.data_structures.mpc_datamodels import InitStatus
from agentlib import AgentVariable, Agent
import numpy as np
import pandas as pd
import casadi as ca
from math import inf


def _set_mean_values(arr):
    """Helper function to set the mean values for the collocation arrays
    #TODO: find a better solution, like using the simulator for the values
    """
    def count_false_after_true(lst):
        count = 0
        found_true = False
        for item in lst:
            if item:
                if found_true:
                    break
                found_true = True
            elif found_true:
                count += 1
        return count
    missing_indices = np.isnan(arr)
    m = count_false_after_true(missing_indices)
    result = []
    values = arr.values[:-1]

    for i in range(0, len(values), m + 1):
        if np.isnan(values[i]):
            data = values[i:i + m + 1]
            non_nan_values = np.nan_to_num(data, nan=0)
            mean_value = np.sum(non_nan_values)/m
            result.append(mean_value)
            result.extend(data[1:])
        else:
            result.extend(arr[i:i + m + 1])

    return result


class FlexibilityShadowMPC(mpc_full.MPC):
    def register_callbacks(self):
        self.__controls = {}
        for control in self.var_ref.controls:
            self.agent.data_broker.register_callback(
                name=f"{control}_full", alias=f"{control}_full", callback=self.calc_flex_callback
            )
            self.__controls[control] = None
        super().register_callbacks()

    def calc_flex_callback(self, inp, name):
        """
        set the control trajectories before calculating the flexibility offer.
        self.model should account for flexibility in its cost function
        """
        # during provision dont calculate flex
        if self.get("in_provision").value:
            return
        vals = pd.Series(_set_mean_values(inp.value), index=inp.value.index[:-1])
        # The MPC Predictions starts at t=env.now not t=0!
        vals.index += self.env.time
        self.__controls[name.replace("_full", "")] = vals
        self.set(f"_{name.replace('_full', '')}", vals)
        # make sure all controls are set
        if all(x is not None for x in self.__controls.values()):
            self.do_step()
            for x in self.__controls.keys():
                self.__controls[x.replace("_full", "")] = None

    def process(self):
        # the shadow mpc should only be run after the results of the baseline are sent
        yield self.env.event()

class FlexibilityProvisorMPC(mpc_full.MPC):
    def pre_computation_hook(self):
        if self.get("in_provision").value:
            timestep = self.get("_P_external").value.index[1] - self.get("_P_external").value.index[0]
            self.set("rel_start", self.get("_P_external").value.index[0] - self.env.time)
            # the provision profile gives a value for the start of a time step. For the end of the flex interval add time step!
            self.set("rel_end", self.get("_P_external").value.index[-1] - self.env.time + timestep)