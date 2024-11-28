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
            name="power_flex_neg", unit='W', type="pd.Series",
            description="Negative Powerflexibility"
        ),
        agentlib.AgentVariable(
            name="power_flex_pos", unit='W', type="pd.Series",
            description="Positive Powerflexibility"
        ),
        agentlib.AgentVariable(
            name="power_flex_neg_avg", unit='kW', type="float",
            description="Negative Average Powerflexibility"
        ),
        agentlib.AgentVariable(
            name="power_flex_pos_avg", unit='kW', type="float",
            description="Positive Average Powerflexibility"
        ),
        agentlib.AgentVariable(
            name="power_flex_neg_max", unit='kW', type="float",
            description="Negative Maximal Powerflexibility"
        ),
        agentlib.AgentVariable(
            name="power_flex_neg_min", unit='kW', type="float",
            description="Negative Minimal Powerflexibility"
        ),
        agentlib.AgentVariable(
            name="power_flex_pos_max", unit='kW', type="float",
            description="Positive Maximal Powerflexibility"
        ),
        agentlib.AgentVariable(
            name="power_flex_pos_min", unit='kW', type="float",
            description="Positive Minimal Powerflexibility"
        ),
        agentlib.AgentVariable(
            name="energyflex_neg", unit='kWh', type="float",
            description="Negative Energyflexibility"
        ),
        agentlib.AgentVariable(
            name="energyflex_pos", unit='kWh', type="float",
            description="Positive Energyflexibility"
        ),
        agentlib.AgentVariable(
            name="costs_neg", unit='ct', type="float", description="Saved costs due to baseline"
        ),
        agentlib.AgentVariable(
            name="costs_pos", unit='ct', type="float", description="Saved costs due to baseline"
        ),
        agentlib.AgentVariable(
            name="costs_neg_rel", unit='ct/kWh', type="float",
            description="Saved costs due to baseline"
        ),
        agentlib.AgentVariable(
            name="costs_pos_rel", unit='ct/kWh', type="float",
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
        agentlib.AgentVariable(name=glbs.PREDICTION_HORIZON, unit="-",
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
    indicator_meta: IndicatorCalculator = None


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
        self.df = pd.DataFrame(columns=self.var_list)
        self.offer_count = 0
        self.indicator_meta = IndicatorCalculator(
            prep_time=self.get(glbs.PREP_TIME).value,
            market_time=self.get(glbs.MARKET_TIME).value,
            flex_event_duration=self.get(glbs.FLEX_EVENT_DURATION).value,
            time_step=self.get(glbs.TIME_STEP).value,
            prediction_horizon=self.get(glbs.PREDICTION_HORIZON).value
        )

    def send_flex_offer(
            self, name,
            power_profile_base,
            power_profile_diff_neg, neg_price,
            power_profile_diff_pos, pos_price,
            timestamp: float = None
    ):
        """
        Send a flex offer as an agent Variable. The first offer is dismissed,
        because the 

        Inputs:

        name: name of the agent variable
        power_profile_base: power profile from the base MPC
        pos_price: price to provise the positive flex profile
        power_profile_diff_pos: difference profile (pos flex MPC - base MPC)
        neg_price: price to provise the negative flex profile
        power_profile_diff_neg: difference profile (neg flex MPC - base MPC)
        timestamp: the time offer was generated

        """
        if self.offer_count > 0:
            var = self._variables_dict[name]
            var.value = FlexOffer(
                base_power_profile=power_profile_base,
                pos_price=pos_price,
                pos_diff_profile=power_profile_diff_pos,
                neg_price=neg_price,
                neg_diff_profile=power_profile_diff_neg
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

                # set the values to None to reset the callback
                self.base_vals = None
                self.neg_vals = None
                self.pos_vals = None
                self._r_pel = None

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
        # Calculate the flexibility KPIs for current predictions
        self.indicator_meta.calculate(
            power_profile_base=self.base_vals,
            power_profile_flex_pos=self.pos_vals,
            power_profile_flex_neg=self.neg_vals,
            costs_profile_electricity=self._r_pel
        )

        # Send flex offer
        self.send_flex_offer(
            name="FlexibilityOffer",
            power_profile_base=self.indicator_meta.power_profile_base,
            power_profile_diff_neg=self.indicator_meta.neg_kpis.power_flex.value, neg_price=self.indicator_meta.neg_kpis.costs.integrate(),
            power_profile_diff_pos=self.indicator_meta.pos_kpis.power_flex.value, pos_price=self.indicator_meta.pos_kpis.costs.integrate(),
        )

        # set outputs
        # todo: remove hardcoded strings
        # todo: loop over all outputs
        self.set("power_flex_neg", self.indicator_meta.neg_kpis.power_flex.value)
        self.set("power_flex_neg_avg", self.indicator_meta.neg_kpis.power_flex.mean())
        self.set("power_flex_neg_max", self.indicator_meta.neg_kpis.power_flex.max())
        self.set("power_flex_neg_min", self.indicator_meta.neg_kpis.power_flex.min())
        self.set("energyflex_neg", self.indicator_meta.neg_kpis.energy_flex.value)
        self.set("costs_neg", self.indicator_meta.neg_kpis.costs.integrate())
        self.set("costs_neg_rel", self.indicator_meta.neg_kpis.costs_rel.value)

        self.set("power_flex_pos", self.indicator_meta.pos_kpis.power_flex.value)
        self.set("power_flex_pos_avg", self.indicator_meta.pos_kpis.power_flex.mean())
        self.set("power_flex_pos_max", self.indicator_meta.pos_kpis.power_flex.max())
        self.set("power_flex_pos_min", self.indicator_meta.pos_kpis.power_flex.min())
        self.set("energyflex_pos", self.indicator_meta.pos_kpis.energy_flex.value)
        self.set("costs_pos", self.indicator_meta.pos_kpis.costs.integrate())
        self.set("costs_pos_rel", self.indicator_meta.pos_kpis.costs_rel.value)

        # write results
        self.df = self.write_results(df=self.df, ts=self.get(glbs.TIME_STEP).value, n=self.get(glbs.PREDICTION_HORIZON).value)
        self.df.to_csv(self.config.results_file)

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

    def cleanup_results(self):
        results_file = self.config.results_file
        if not results_file:
            return
        os.remove(results_file)

