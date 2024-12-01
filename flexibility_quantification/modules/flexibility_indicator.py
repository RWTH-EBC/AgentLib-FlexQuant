import os
import sys
from typing import Optional, List
import agentlib
import numpy as np
import pandas as pd
from pathlib import Path
import pydantic
import flexibility_quantification.data_structures.globals as glbs
from flexibility_quantification.data_structures.indicator import IndicatorData, IndicatorKPIs, DiscretizationTypes

from flexibility_quantification.utils.data_handling import strip_multi_index
sys.path.append(os.path.dirname(__file__))

from flexibility_quantification.data_structures.flex_offer import FlexOffer

kpis_pos = IndicatorKPIs(direction="positive")
kpis_neg = IndicatorKPIs(direction="negative")

class FlexibilityIndicatorModuleConfig(agentlib.BaseModuleConfig):
    inputs: List[agentlib.AgentVariable] = [
        agentlib.AgentVariable(name="__P_el_pos", unit="W", type="pd.Series",
                               description="The power input to the system"),
        agentlib.AgentVariable(name="__P_el_base", unit="W", type="pd.Series",
                               description="The power input to the system"),
        agentlib.AgentVariable(name="__P_el_neg", unit="W", type="pd.Series",
                               description="The power input to the system"),
        agentlib.AgentVariable(name="r_pel", unit="ct/kWh", type="pd.Series",
                               description="electricity price")
    ]
    outputs: List[agentlib.AgentVariable] = [
        # Flexibility offer
        agentlib.AgentVariable(name=glbs.FlexibilityOffer, type="FlexOffer"),

        # Power KPIs
        agentlib.AgentVariable(
            name=kpis_neg.power_flex_full.get_name(), unit='W', type="pd.Series",
            description="Negative power flexibility"
        ),
        agentlib.AgentVariable(
            name=kpis_pos.power_flex_full.get_name(), unit='W', type="pd.Series",
            description="Positive power flexibility"
        ),
        agentlib.AgentVariable(
            name=kpis_neg.power_flex_offer.get_name(), unit='W', type="pd.Series",
            description="Negative power flexibility"
        ),
        agentlib.AgentVariable(
            name=kpis_pos.power_flex_offer.get_name(), unit='W', type="pd.Series",
            description="Positive power flexibility"
        ),

        # Energy KPIs
        agentlib.AgentVariable(
            name=kpis_neg.energy_flex.get_name(), unit='kWh', type="float",
            description="Negative energy flexibility"
        ),
        agentlib.AgentVariable(
            name=kpis_pos.energy_flex.get_name(), unit='kWh', type="float",
            description="Positive energy flexibility"
        ),

        # Costs KPIs
        agentlib.AgentVariable(
            name=kpis_neg.costs.get_name(), unit="ct", type="float",
            description="Saved costs due to baseline"
        ),
        agentlib.AgentVariable(
            name=kpis_pos.costs.get_name(), unit="ct", type="float",
            description="Saved costs due to baseline"
        ),
        agentlib.AgentVariable(
            name=kpis_neg.costs_rel.get_name(), unit='ct/kWh', type="float",
            description="Saved costs due to baseline"
        ),
        agentlib.AgentVariable(
            name=kpis_pos.costs_rel.get_name(), unit='ct/kWh', type="float",
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
    discretization: DiscretizationTypes = pydantic.Field(
        default="collocation",
        description="Name of the discretization method",
    )

    shared_variable_fields: List[str] = ["outputs"]


class FlexibilityIndicatorModule(agentlib.BaseModule):
    config: FlexibilityIndicatorModuleConfig

    data: IndicatorData

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.var_list = []
        for variable in self.variables:
            if variable == glbs.FlexibilityOffer:
                continue
            self.var_list.append(variable.name)
        self.time = []
        self.base_vals = None
        self.pos_vals = None
        self.neg_vals = None
        self._r_pel = None
        self.in_provision = False
        self.df = pd.DataFrame(columns=pd.Series(self.var_list))
        self.offer_count = 0
        self.data = IndicatorData(
            prep_time=self.get(glbs.PREP_TIME).value,
            market_time=self.get(glbs.MARKET_TIME).value,
            flex_event_duration=self.get(glbs.FLEX_EVENT_DURATION).value,
            time_step=self.get(glbs.TIME_STEP).value,
            prediction_horizon=self.get(glbs.PREDICTION_HORIZON).value,
            discretisation_type=self.config.discretization,
        )

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
        Opens results file of flexibility_indicator.py
        results_file defined in __init__
        """
        results_file = self.config.results_file
        try:
            results = pd.read_csv(results_file, header=[0], index_col=[0, 1])
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
            new_df.index.direction = "time"
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

    def flexibility(self):
        # Calculate the flexibility KPIs for current predictions
        self.data.calculate(
            power_profile_base=self.base_vals,
            power_profile_flex_pos=self.pos_vals,
            power_profile_flex_neg=self.neg_vals,
            power_unit=self.config.power_unit,
            costs_profile_electricity=self._r_pel
        )

        # Send flex offer
        self.send_flex_offer(
            name=glbs.FlexibilityOffer,
            base_power_profile=self.data.power_profile_base,
            pos_diff_profile=self.data.kpis_pos.power_flex_offer.value,
            pos_price=self.data.kpis_pos.costs.value,
            neg_diff_profile=self.data.kpis_neg.power_flex_offer.value,
            neg_price=self.data.kpis_neg.costs.value,
        )

        # set outputs
        for kpi in self.data.get_kpis().values():
            for output in self.config.outputs:
                if output.name == kpi.get_name():
                    self.set(output.name, kpi.value)

        # write results
        self.df = self.write_results(
            df=self.df,
            ts=self.get(glbs.TIME_STEP).value,
            n=self.get(glbs.PREDICTION_HORIZON).value
        )

        # save results
        self.df.to_csv(self.config.results_file)

    def send_flex_offer(
            self, name,
            base_power_profile: pd.Series,
            pos_diff_profile: pd.Series, pos_price: float,
            neg_diff_profile: pd.Series, neg_price: float,
            timestamp: float = None
    ):
        """
        Send a flex offer as an agent Variable. The first offer is dismissed,
        because the

        Inputs:

        name: name of the agent variable
        indicator_data: the indicator data object
        timestamp: the time offer was generated

        """
        if self.offer_count > 0:
            var = self._variables_dict[name]
            var.value = FlexOffer(
                base_power_profile=base_power_profile,
                pos_diff_profile=pos_diff_profile, pos_price=pos_price,
                neg_diff_profile=neg_diff_profile, neg_price=neg_price,
            )
            if timestamp is None:
                timestamp = self.env.time
            var.timestamp = timestamp
            self.agent.data_broker.send_variable(
                variable=var.copy(update={"source": self.source}),
                copy=False,
            )
        self.offer_count += 1
