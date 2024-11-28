import os
import sys
from typing import Optional, Union, List
import agentlib
import numpy as np
import pandas as pd
from pathlib import Path
import pydantic
import flexibility_quantification.data_structures.globals as glbs
from flexibility_quantification.data_structures.indicator import IndicatorCalculator

from flexibility_quantification.utils.data_handling import strip_multi_index
sys.path.append(os.path.dirname(__file__))

from flexibility_quantification.data_structures.flex_offer import FlexOffer


class FlexibilityIndicatorModuleConfig(agentlib.BaseModuleConfig):
    inputs: List[agentlib.AgentVariable] = [
        agentlib.AgentVariable(name="__P_el_pos", unit="W",
                               description="The power input to the system"),
        agentlib.AgentVariable(name="__P_el_base", unit="W",
                               description="The power input to the system"),
        agentlib.AgentVariable(name="__P_el_neg", unit="W",
                               description="The power input to the system"),
        agentlib.AgentVariable(name="r_pel", unit="ct/kWh", type="pd.Series",
                               description="electricity price")
    ]
    outputs: List[agentlib.AgentVariable] = [
        agentlib.AgentVariable(name="FlexibilityOffer", type="FlexOffer"),
        agentlib.AgentVariable(
            name="powerflex_flex_neg", unit='W', type="pd.Series",
            description="Negative Powerflexibility"
        ),
        agentlib.AgentVariable(
            name="powerflex_flex_pos", unit='W', type="pd.Series",
            description="Positive Powerflexibility"
        ),
        agentlib.AgentVariable(
            name="powerflex_avg_neg", unit='kW', type="pd.Series",
            description="Negative Average Powerflexibility"
        ),
        agentlib.AgentVariable(
            name="powerflex_avg_pos", unit='kW', type="pd.Series",
            description="Positive Average Powerflexibility"
        ),
        agentlib.AgentVariable(
            name="powerflex_neg_max", unit='kW', type="pd.Series",
            description="Negative Maximal Powerflexibility"
        ),
        agentlib.AgentVariable(
            name="powerflex_neg_min", unit='kW', type="pd.Series",
            description="Negative Minimal Powerflexibility"
        ),
        agentlib.AgentVariable(
            name="powerflex_pos_max", unit='kW', type="pd.Series",
            description="Positive Maximal Powerflexibility"
        ),
        agentlib.AgentVariable(
            name="powerflex_pos_min", unit='kW', type="pd.Series",
            description="Positive Minimal Powerflexibility"
        ),
        agentlib.AgentVariable(
            name="energyflex_neg", unit='kWh', type="pd.Series",
            description="Negative Energyflexibility"
        ),
        agentlib.AgentVariable(
            name="energyflex_pos", unit='kWh', type="pd.Series",
            description="Positive Energyflexibility"
        ),
        agentlib.AgentVariable(
            name="costs_neg", unit='ct', type="pd.Series", description="Saved costs due to baseline"
        ),
        agentlib.AgentVariable(
            name="costs_pos", unit='ct', type="pd.Series", description="Saved costs due to baseline"
        ),
        agentlib.AgentVariable(
            name="costs_neg_rel", unit='ct/kWh', type="pd.Series",
            description="Saved costs due to baseline"
        ),
        agentlib.AgentVariable(
            name="costs_pos_rel", unit='ct/kWh', type="pd.Series",
            description="Saved costs due to baseline"
        )
    ]

    # TODO: don't use parameters list, but create a IndicatorSpecifications class (see market)
    parameters: List[agentlib.AgentVariable] = [
        agentlib.AgentVariable(name=glbs.PREP_TIME, unit="s",
                               description="Preparation time"),
        agentlib.AgentVariable(name=glbs.MARKET_TIME, unit="s",
                               description="Market time"),
        agentlib.AgentVariable(name=glbs.FLEX_EVENT_DURATION, unit="s",
                               description="time to switch objective"),
        agentlib.AgentVariable(name=glbs.TIME_STEP, unit="s",
                               description="timestep of the mpc solution"),
        agentlib.AgentVariable(name="prediction_horizon", unit="-",
                               description="prediction horizon of the mpc solution"),

    ]

    results_file: Optional[Path] = pydantic.Field(default=None)
    # TODO: use these two
    save_results: Optional[bool] = pydantic.Field(validate_default=True, default=None)
    overwrite_result_file: Optional[bool] = pydantic.Field(default=False, validate_default=True)

    price_variable: str = pydantic.Field(
        default="r_pel",
        description="Name of the price variable send by a predictor",
    )
    power_unit: str = pydantic.Field(
        default="kW",
        description="Unit of the power variable"
    )
    discretization: str = pydantic.Field(
        default="collocation",
        description="Name of the discretization method",
    )

    shared_variable_fields: List[str] = ["outputs"]

    # # Quatsch von felix
    # indicator_meta: IndicatorMeta(df).energy_flex.calc()
    #         energy_flex_pos(IndicatorKPI)
    #         energy_flex_neg(IndicatorKPI)
    #                 IndicatorKPI
    #                     name
    #                     value
    #                     unit
    #                     type
    #
    # for output in self.config.outputs:
    #     if output.name in [kpi.name for kpi in self.indicator_meta.energylfex]
    #         self.set("name", value)


class FlexibilityIndicatorModule(agentlib.BaseModule):
    config: FlexibilityIndicatorModuleConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.var_list = []
        for variable in self.variables:
            if variable == "FlexibilityOffer":
                continue
            self.var_list.append(variable.name)
        self.time = []
        self.base_vals = None
        self.pos_vals = None
        self.neg_vals = None
        self._r_pel = None
        self.in_provision = False
        self.df = None
        self.offer_count = 0

    def send_flex_offer(
            self, name, base_power_profile, pos_price, pos_diff_profile,
            neg_price, neg_diff_profile, timestamp: float = None):
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
            var.value = FlexOffer(
                base_power_profile=base_power_profile,
                pos_price=pos_price,
                pos_diff_profile=pos_diff_profile,
                neg_price=neg_price,
                neg_diff_profile=neg_diff_profile
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
                self.neg_vals = None
                self.pos_vals = None
                self._r_pel = None
        # TODO: remove hardcoded strings
        if not self.in_provision:
            if name == "__P_el_base":
                self.base_vals = inp.value
                self.base_vals = strip_multi_index(self.base_vals)
            elif name == "__P_el_neg":
                self.neg_vals = inp.value
                self.neg_vals = strip_multi_index(self.neg_vals)
            elif name == "__P_el_pos":
                self.pos_vals = inp.value
                self.pos_vals = strip_multi_index(self.pos_vals)
            elif name == self.config.price_variable:
                # price comes from predictor, so no stripping needed
                # TODO: add other sources for price signal?
                self._r_pel = inp.value

            if all(var is not None for var in
                   (self.base_vals, self.neg_vals, self.pos_vals, self._r_pel)):
                self.flexibility()

    def get_results(self) -> Optional[pd.DataFrame]:
        """
        Opens results file of flexibilityindicators.py
        results_file defined in __init__
        """
        results_file = self.config.results_file
        try:
            results = pd.read_csv(results_file, header=[0], index_col=[0, 1])
            return results
        except FileNotFoundError:
            self.logger.error("Results file %s was not found.", results_file)
            return None

    def flexibility(self):
        """
        Calculate the flexibility, the costs, send the offer and write and save the results
        """
        # check if the dataframe is already initialized
        if self.df is None:
            self.df = pd.DataFrame(columns=self.var_list)

        # get parameter
        prep_time = self.get(glbs.PREP_TIME).value
        market_time = self.get(glbs.MARKET_TIME).value
        switch_time = prep_time + market_time
        flex_event_duration = self.get(glbs.FLEX_EVENT_DURATION).value
        horizon = self.get("prediction_horizon").value
        time_step = self.get(glbs.TIME_STEP).value

        # test = IndicatorCalculator(
            # prep_time=self.get(glbs.PREP_TIME).value,
            # market_time=self.get(glbs.MARKET_TIME).value,
            # flex_event_duration=self.get(glbs.FLEX_EVENT_DURATION).value,
            #  time_step=self.get(glbs.TIME_STEP).value,
            #     prediction_horizon=self.get("prediction_horizon").value
        # )
        # print(test)
        # test.calculate(
        #     power_profile_base=self.base_vals,
        #     power_profile_flex_pos=self.pos_vals,
        #     power_profile_flex_neg=self.neg_vals,
        #     costs_profile_electricity=self._r_pel
        # )
        # print(test)
        # print()

        # generate horizons
        # 1. for the flexibility range
        flex_horizon = np.arange(
            switch_time, switch_time + flex_event_duration, time_step)
        # 2. for the full range of prediction
        full_horizon = np.arange(
            0, horizon * time_step, time_step)

        self.uniform_data(full_horizon=full_horizon)
        self.calculate_and_send_offer(full_horizon=full_horizon, flex_horizon=flex_horizon, time_step=time_step, horizon=horizon)
        self.safe_results()

        # set the values to None to reset the callback
        self.base_vals = None
        self.neg_vals = None
        self.pos_vals = None
        self._r_pel = None

    def uniform_data(self, full_horizon: np.ndarray):   # todo
        """Uniform data to the same length
        """
        # uniform regarding discretization
        if self.config.discretization == "collocation":
            # As the collocation uses the values after each time step, the last value is always none
            time = self.base_vals.index[:-1]

            # use only the values of the full time steps
            self.base_vals = pd.Series(self.base_vals, index=time).reindex(index=full_horizon)
            self.neg_vals = pd.Series(self.neg_vals, index=time).reindex(index=full_horizon)
            self.pos_vals = pd.Series(self.pos_vals, index=time).reindex(index=full_horizon)

        # TODO: units anpassen (über Agentvars?)
        # convert unit of power variables in kW
        if self.config.power_unit == "kW":
            scaler = 1
        else:
            scaler = 1000
        self.base_vals = self.base_vals / scaler
        self.neg_vals = self.neg_vals / scaler
        self.pos_vals = self.pos_vals / scaler

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
        now = self.env.now
        for name in self.var_list:
            # Use the power variables averaged for each timestep, not the collocation values
            if name == "__P_el_base":
                values = self.base_vals
            elif name == "__P_el_neg":
                values = self.neg_vals
            elif name == "__P_el_pos":
                values = self.pos_vals
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
            new_df.index.type = "time"
            new_df[glbs.TIME_STEP] = now
            new_df.set_index([glbs.TIME_STEP, new_df.index], inplace=True)
            df = pd.concat([df, new_df])
            # set the indices once again as concat cant handle indices properly
            indices = pd.MultiIndex.from_tuples(df.index, names=[glbs.TIME_STEP, "time"])
            df.set_index(indices, inplace=True)

        return df

    def safe_results(self):
        self.df.to_csv(self.config.results_file)

    def cleanup_results(self):
        results_file = self.config.results_file
        if not results_file:
            return
        os.remove(results_file)

    def _diff_negative_flexibility(self) -> np.ndarray:
        diff = self.neg_vals.values - self.base_vals.values
        return diff

    def _diff_positive_flexibility(self) -> np.ndarray:
        diff = self.base_vals.values - self.pos_vals.values
        return diff

    def calculate_flexibility(self, abbrev: str, diff_func: callable, full_horizon: np.ndarray, flex_horizon: np.ndarray, ts: float):
        """calculate flexibility by differentiate the baseline and shadow mpc
        """
        powerflex_flex = []
        difference = diff_func
        for i in range(len(self.neg_vals)):
            diff = difference[i]

            if diff < 0:
                percentage_diff = (abs(diff) / self.base_vals.values[i]) * 100

                if percentage_diff < 1:
                    powerflex_flex.append(0)
                else:
                    powerflex_flex.append(diff)
            else:
                powerflex_flex.append(diff)

        # save this variable for the cost flexibilty
        powerflex_flex_full = pd.Series(powerflex_flex, index=full_horizon)
        # the powerflex is defined only in the flexibility region
        powerflex_profile = powerflex_flex_full.reindex(index=flex_horizon)
        powerflex_flex = powerflex_profile.reindex(index=full_horizon)

        # calculate characteristics
        powerflex_avg = np.average(powerflex_flex.dropna())
        powerflex_min = min(powerflex_flex.dropna())
        powerflex_max = max(powerflex_flex.dropna())
        energyflex = (np.sum(powerflex_flex * ts)).round(4)

        # set characteristics
        self.set(f"powerflex_flex_{abbrev}", powerflex_flex)
        self.set(f"powerflex_avg_{abbrev}", str(powerflex_avg))
        self.set(f"powerflex_{abbrev}_min", str(powerflex_min))
        self.set(f"powerflex_{abbrev}_max", str(powerflex_max))
        self.set(f"energyflex_{abbrev}", str(energyflex))

        return powerflex_profile, powerflex_flex_full, energyflex

    def calculate_price(self, abbrev, horizon, full_horizon, powerflex_flex_full, ts, energyflex):
        elec_prices = self._r_pel.iloc[:horizon]
        elec_prices.index = full_horizon

        flex_price = sum(powerflex_flex_full * elec_prices * ts)
        self.set(f"costs_{abbrev}", str(flex_price))

        # Relative Flexibility Costs as deviation of absolute costs for whole prediction horizon
        # and energy flexibility during flexibility event
        if energyflex == 0:
            costs_neg_rel = 0
        else:
            costs_neg_rel = flex_price / energyflex

        self.set(f"costs_{abbrev}_rel", str(costs_neg_rel))

        return flex_price

    def calculate_and_send_offer(self, full_horizon: np.ndarray, flex_horizon: np.ndarray, time_step: float, horizon):
        # convert timestep from seconds to hours for calculations
        ts = time_step / 3600

        # calculate flexibility
        if len(self.base_vals) != len(self.neg_vals) or len(self.base_vals) != len(self.pos_vals):
            raise ValueError("Length of power profiles do not match")
        powerflex_profile_neg, powerflex_flex_neg_full, energyflex_neg = self.calculate_flexibility(
            abbrev="neg", diff_func=self._diff_negative_flexibility(),
            full_horizon=full_horizon, flex_horizon=flex_horizon, ts=ts
        )
        powerflex_profile_pos, powerflex_flex_pos_full, energyflex_pos = self.calculate_flexibility(
            abbrev="pos", diff_func=self._diff_positive_flexibility(),
            full_horizon=full_horizon, flex_horizon=flex_horizon, ts=ts
        )

        # calculate prices
        flex_price_neg = self.calculate_price(
            abbrev="neg", powerflex_flex_full=powerflex_flex_neg_full, energyflex=energyflex_neg,
            horizon=horizon, full_horizon=full_horizon, ts=ts
        )
        flex_price_pos = self.calculate_price(
            abbrev="pos", powerflex_flex_full=powerflex_flex_pos_full, energyflex=energyflex_pos,
            horizon=horizon, full_horizon=full_horizon, ts=ts
        )

        # send flex offer
        base_profile = self.base_vals.reindex(index=flex_horizon)
        self.send_flex_offer("FlexibilityOffer", base_profile,
                             flex_price_pos, powerflex_profile_pos,
                             flex_price_neg, powerflex_profile_neg)

        # write results
        self.df = self.write_results(df=self.df, ts=time_step, n=horizon)
