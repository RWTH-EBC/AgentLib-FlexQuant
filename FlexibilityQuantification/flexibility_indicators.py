import agentlib
import numpy as np
import pandas as pd
import os, sys
from typing import Optional, List
sys.path.append(os.path.dirname(__file__))

from flex_offer import PowerFlexOffer
from matplotlib import pyplot as plt
class FlexibilityIndicatorConfig(agentlib.BaseModuleConfig):
    inputs: List[agentlib.AgentVariable] = [
            agentlib.AgentVariable(name="__P_el_min",unit="W",
                description="The power input to the system"),
            agentlib.AgentVariable(name="__P_el_base",unit="W",
                description="The power input to the system"),
            agentlib.AgentVariable(name="__P_el_max",unit="W",
                description="The power input to the system"),

            agentlib.AgentVariable(name="r_pel", unit="ct/kWh", type="pd.Series", description="Weight for P_el in objective function")
    ]
    outputs: List[agentlib.AgentVariable] = [
        agentlib.AgentVariable(name="PowerFlexibilityOffer", type="PowerFlexOffer"),
        agentlib.AgentVariable(
            name="powerflex_flex_neg", unit='W', type="pd.Series", description="Negative Powerflexibility"
        ),
        agentlib.AgentVariable(
            name="powerflex_flex_pos", unit='W', type="pd.Series", description="Positive Powerflexibility"
        ),
        agentlib.AgentVariable(
            name="powerflex_avg_neg", unit='kW', type="pd.Series", description="Negative Average Powerflexibility"
        ),
        agentlib.AgentVariable(
            name="powerflex_avg_pos", unit='kW', type="pd.Series", description="Positive Average Powerflexibility"
        ),
        agentlib.AgentVariable(
            name="powerflex_neg_max", unit='kW', type="pd.Series", description="Negative Maximal Powerflexibility"
        ),
        agentlib.AgentVariable(
            name="powerflex_neg_min", unit='kW', type="pd.Series", description="Negative Minimal Powerflexibility"
        ),
        agentlib.AgentVariable(
            name="powerflex_pos_max", unit='kW', type="pd.Series", description="Positive Maximal Powerflexibility"
        ),
        agentlib.AgentVariable(
            name="powerflex_pos_min", unit='kW', type="pd.Series", description="Positive Minimal Powerflexibility"
        ),
        agentlib.AgentVariable(
            name="energyflex_neg", unit='kWh', type="pd.Series", description="Negative Energyflexibility"
        ),
        agentlib.AgentVariable(
            name="energyflex_pos", unit='kWh', type="pd.Series", description="Positive Energyflexibility"
        ),
        agentlib.AgentVariable(
            name="timeflex_neg", unit='s', type="pd.Series", description="Negative Timeflexibility nach Komfortgrenzen"
        ),
        agentlib.AgentVariable(
            name="timeflex_pos", unit='s', type="pd.Series", description="Positive Timeflexibility nach Komfortgrenzen"
        ),
        agentlib.AgentVariable(
            name="costs_neg", unit='ct', type="pd.Series", description="Saved costs due to baseline"
        ),
        agentlib.AgentVariable(
            name="costs_pos", unit='ct', type="pd.Series", description="Saved costs due to baseline"
        ),
        agentlib.AgentVariable(
            name="costs_neg_rel", unit='ct/kWh', type="pd.Series", description="Saved costs due to baseline"
        ),
        agentlib.AgentVariable(
            name="costs_pos_rel", unit='ct/kWh', type="pd.Series", description="Saved costs due to baseline"
        )
    ]
    
    parameters: List[agentlib.AgentVariable] = [
        agentlib.AgentVariable(name="prep_time",unit="s",
                description="time to switch objective"),
        agentlib.AgentVariable(name="flex_event_duration",unit="s",
            description="time to switch objective"),
        agentlib.AgentVariable(name="time_step",unit="s",
            description="timestep of the mpc solution"),
        agentlib.AgentVariable(name="prediction_horizon",unit="-",
            description="prediction horizon of the mpc solution"),
        
    ]
    shared_variable_fields:List[str] = ["outputs"]

class FlexibilityIndicator(agentlib.BaseModule):
    config: FlexibilityIndicatorConfig
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.results_file: Optional[str] = kwargs.get("results_file") or "flexibility_results.csv"
        self.var_list = []
        for variable in self.variables:
            if variable == "PowerFlexibilityOffer":
                continue
            self.var_list.append(variable.name)
        self.time = []
        self.base_vals = None
        self.min_vals = None
        self.max_vals = None
        self._r_pel = None
        self.in_provision = False
        self.df = None
        self.offer_count = 0

    def send_flex_offer(
            self, name, base_power_profile, pos_price, pos_diff_profile,
            pos_time_flex, neg_price, neg_diff_profile, neg_time_flex, 
            timestamp:float=None):
        """
        Send a flex offer as an agent Variable. The first offer is dismissed,
        because the 

        Inputs:

        name: name of the agent variable
        base_power_profile: power profile from the base MPC
        pos_price: price to provise the positive flex profile
        pos_diff_profile: difference profile (pos flex MPC - base MPC)
        neg_price: price to provise the negative flex profile
        neg_diff_profile: difference profile (neg flex MPC - base MPC)
        timestamp: the time offer was generated

        """
        if self.offer_count > 0:
            var = self._variables_dict[name]
            var.value = PowerFlexOffer(
                base_power_profile=base_power_profile,
                pos_price=pos_price,
                pos_diff_profile=pos_diff_profile,
                pos_time_flex=pos_time_flex,
                neg_price=neg_price,
                neg_diff_profile=neg_diff_profile,
                neg_time_flex=neg_time_flex
                )
            if timestamp is None:
                timestamp = self.env.time
            var.timestamp = timestamp
            self.agent.data_broker.send_variable(
                variable=var.copy(update={"source": self.source}),
                copy=False,
            )
        self.offer_count += 1

    def register_callbacks(self):
        inputs = self.config.inputs
        for var in inputs:
            self.agent.data_broker.register_callback(
                name=var.name, alias=var.name, callback=self.callback
            )
        self.agent.data_broker.register_callback(
            name="in_provision", alias="in_provision", callback=self.callback
        )    
    def process(self):
        yield self.env.event()

    def callback(self, inp, name):
        if name == "in_provision":
            self.in_provision = inp.value
            if self.in_provision:
                self.base_vals = None
                self.max_vals = None
                self.min_vals = None
                self._r_pel = None
        if not self.in_provision:
            if name == "__P_el_base":
                self.base_vals = inp.value
            elif name == "__P_el_max":
                self.max_vals = inp.value
            elif name == "__P_el_min":
                self.min_vals = inp.value
            elif name == "r_pel":
                self._r_pel = inp.value
            
            if all(var is not None for var in (self.base_vals, self.max_vals, self.min_vals, self._r_pel)):
                self.calc_flex()


    def get_results(self) -> Optional[pd.DataFrame]:
        """
        Opens results file of flexibilityindicators.py
        results_file defined in __init__
        """
        results_file = self.results_file
        try:
            results = pd.read_csv(results_file, header=[0], index_col=[0,1])
            return results
        except FileNotFoundError:
            self.logger.error("Results file %s was not found.", results_file)
            return None

    def write_results(self, df, ts, n):
        """
        Write every data of variables in self.var_list in an DataFrame
        DataFrame will be updated every time step

        Args:
            df: DataFrame which is initialised as an empty DataFrame with columns according to self.var_list
            ts: time step
            n: number of time steps during prediction horizon
        Returns:
            DataFrame with results of every variable in self.var_list
        """
        results = []
        now=self.env.now
        for name in self.var_list:
            # Use the power variables averaged for each timestep, not the collocation values
            if name == "__P_el_base":
                values = self.base_vals
            elif name == "__P_el_max":
                values = self.max_vals
            elif name == "__P_el_min":
                values = self.min_vals
            else:
                values = self.get(name).value

            if isinstance(values, pd.Series):
                traj = values.reindex(np.arange(0, n * ts, ts))
            else:
                traj = pd.Series(values).reindex(np.arange(0, n * ts, ts))
            results.append(traj)

        if not now % ts:
            self.time.append(now)
            new_df = pd.DataFrame(results).T
            new_df.columns = self.var_list
            new_df.index.name = "time"
            new_df['time_step'] = now
            new_df.set_index(['time_step', new_df.index], inplace=True)
            df = pd.concat([df, new_df])
            # set the indices once again as concat cant handle indices properly
            indices = pd.MultiIndex.from_tuples(df.index, names=["time_step", "time"])
            df.set_index(indices, inplace=True)

        return df

    def cleanup_results(self):
        results_file = self.results_file
        if not results_file:
            return
        os.remove(results_file)
    
    def _mean_value(self, arr, m):
        """
        Calculate the mean value for segments of length @m in the input array @arr 
            and replace NaN values with segment mean.
        """
    
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
    

    def calc_flex(self):
        if self.df is None:
            self.df = pd.DataFrame(columns=self.var_list)
        prep_time = self.get("prep_time").value
        flex_event_duration = self.get("flex_event_duration").value
        horizon = self.get("prediction_horizon").value
        time_step = self.get("time_step").value
        
        # generate horizons
        # 1. for the flexibility range
        flex_horizon = np.arange(
            prep_time, prep_time + flex_event_duration, time_step)
        # 2. for the full range of prediction
        full_horizon = np.arange(
            0, horizon*time_step, time_step)
    
        # As the collocation uses the values after each time step, the last value is always none
        time = self.base_vals.index[:-1]

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
        # for each time_step, get the mean of the collocation values
        missing_indices = np.isnan(self.base_vals)
        m = count_false_after_true(missing_indices)
        self.base_vals = self._mean_value(arr=self.base_vals, m=m)
        self.max_vals = self._mean_value(arr=self.max_vals, m=m)
        self.min_vals = self._mean_value(arr=self.min_vals, m=m)
        # use only the values of the full time steps
        self.base_vals = pd.Series(self.base_vals, index=time).reindex(index=full_horizon)
        self.max_vals = pd.Series(self.max_vals, index=time).reindex(index=full_horizon)
        self.min_vals = pd.Series(self.min_vals, index=time).reindex(index=full_horizon)

        powerflex_flex_neg = []
        for i in range(len(self.max_vals)):
            diff = self.max_vals.values[i] - self.base_vals.values[i]

            if diff < 0:
                percentage_diff = (abs(diff) / self.base_vals.values[i]) * 100

                if percentage_diff < 1:
                    powerflex_flex_neg.append(0)
                else:
                    powerflex_flex_neg.append(diff)
            else:
                powerflex_flex_neg.append(diff)
        # save this variable for the cost flexibilty
        powerflex_flex_neg_full = pd.Series(powerflex_flex_neg, index=full_horizon)
        # the powerflex is defined only in the flexibility region
        powerflex_profile_neg = powerflex_flex_neg_full.reindex(index=flex_horizon)
        powerflex_flex_neg = powerflex_profile_neg.reindex(index=full_horizon)
        self.set("powerflex_flex_neg", powerflex_flex_neg)
        # W -> kW
        self.set("powerflex_avg_neg", str(np.average(powerflex_flex_neg.dropna())/1e3))
        self.set("powerflex_neg_min", str(min(powerflex_flex_neg.dropna()) / 1e3))
        self.set("powerflex_neg_max", str(max(powerflex_flex_neg.dropna()) / 1e3))
        # J -> kWh 
        energyflex_neg = (np.sum(powerflex_flex_neg*time_step) / 3.6e6).round(4)
        self.set("energyflex_neg", str(energyflex_neg))

        powerflex_flex_pos = []
        for i in range(len(self.min_vals)):
            diff = self.base_vals.values[i] - self.min_vals.values[i]

            if diff < 0:
                percentage_diff = (abs(diff) / self.base_vals.values[i]) * 100

                if percentage_diff < 1:
                    powerflex_flex_pos.append(0)
                else:
                    powerflex_flex_pos.append(diff)
            else:
                powerflex_flex_pos.append(diff)
        # save this variable for the cost flexibilty
        powerflex_flex_pos_full = pd.Series(powerflex_flex_pos, index=full_horizon)
        # the powerflex is defined only in the flexibility region
        powerflex_profile_pos = powerflex_flex_pos_full.reindex(index=flex_horizon)
        powerflex_flex_pos = powerflex_profile_pos.reindex(index=full_horizon)

        self.set("powerflex_flex_pos", powerflex_flex_pos)
        # W -> kW
        self.set("powerflex_avg_pos", str(np.average(powerflex_flex_pos.dropna())/1e3))
        self.set("powerflex_pos_min", str(min(powerflex_flex_pos.dropna()) / 1e3))
        self.set("powerflex_pos_max", str(max(powerflex_flex_pos.dropna()) / 1e3))
        # J -> kWh 
        energyflex_pos = (np.sum(powerflex_flex_pos*time_step) / 3.6e6).round(4)

        self.set("energyflex_pos", str(energyflex_pos))

        def calc_timeflex(powerflex, ts):
            sum = 0
            sum_weighted = 0
            for i, val in enumerate(powerflex):
                sum += val * ts
                t_i = powerflex.index.values[i] - (powerflex.index.values[0] / 2)
                sum_weighted += val*ts*t_i
            timeflex = sum_weighted / sum
            if pd.isna(timeflex):
                return 0
            else:
                return timeflex

        
        timeflex_neg = calc_timeflex(powerflex_flex_neg.reindex(flex_horizon), time_step)
        self.set("timeflex_neg", str(timeflex_neg))

        timeflex_pos = calc_timeflex(powerflex_flex_pos.reindex(flex_horizon), time_step)
        self.set("timeflex_pos", str(timeflex_pos))
        
        elec_prices = self._r_pel.iloc[:horizon]
        elec_prices.index = full_horizon
        flex_price_neg = sum(powerflex_flex_neg_full * elec_prices * time_step / 3.6e6)
        flex_price_pos = -1 * sum(powerflex_flex_pos_full * elec_prices * time_step / 3.6e6)

        self.set("costs_neg", str(flex_price_neg))
        self.set("costs_pos", str(flex_price_pos))
        
        # Relative Flexibility Costs as deviation of absolute costs for whole prediction horizon
        # and energy flexibility during flexibility event

        if energyflex_neg == 0:
            costs_neg_rel = 0
        else:
            costs_neg_rel = flex_price_neg/energyflex_neg

        if energyflex_pos == 0:
            costs_pos_rel = 0
        else:
            costs_pos_rel = flex_price_pos/energyflex_pos

        self.set("costs_neg_rel", str(costs_neg_rel))
        self.set("costs_pos_rel", str(costs_pos_rel))
        
        base_profile = self.base_vals.reindex(index=flex_horizon)
        self.send_flex_offer("PowerFlexibilityOffer", base_profile, 
                             flex_price_pos, powerflex_profile_pos, timeflex_pos, 
                             flex_price_neg, powerflex_profile_neg, timeflex_neg)
        
        self.df = self.write_results(df=self.df, ts=time_step, n=horizon)
        self.df.to_csv(self.results_file)
        # set the values to None to reset the callback
        self.base_vals = None
        self.max_vals = None
        self.min_vals = None
        self._r_pel = None