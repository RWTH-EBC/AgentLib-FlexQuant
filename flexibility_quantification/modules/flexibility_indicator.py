import os
import sys
from typing import Optional, List, Literal
import agentlib
import numpy as np
import pandas as pd
from pathlib import Path
import pydantic
import flexibility_quantification.data_structures.globals as glbs
from flexibility_quantification.data_structures.indicator import FlexibilityData, FlexibilityKPIs

DiscretizationTypes = Literal["collocation", "multiple_shooting"]

sys.path.append(os.path.dirname(__file__))

from flexibility_quantification.data_structures.flex_offer import FlexOffer

kpis_pos = FlexibilityKPIs(direction="positive")
kpis_neg = FlexibilityKPIs(direction="negative")

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
    discretization: str = pydantic.Field(
        default=glbs.COLLOCATION,
        description="Name of the discretization method",
    )

    shared_variable_fields: List[str] = ["outputs"]


class FlexibilityIndicatorModule(agentlib.BaseModule):
    config: FlexibilityIndicatorModuleConfig

    data: FlexibilityData

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.var_list = []
        for variable in self.variables:
            if variable == glbs.FlexibilityOffer:
                continue
            self.var_list.append(variable.name)
        self.time = []
        self.in_provision = False
        self.offer_count = 0
        self.data = FlexibilityData(
            prep_time=self.get(glbs.PREP_TIME).value,
            market_time=self.get(glbs.MARKET_TIME).value,
            flex_event_duration=self.get(glbs.FLEX_EVENT_DURATION).value,
            time_step=self.get(glbs.TIME_STEP).value,
            prediction_horizon=self.get(glbs.PREDICTION_HORIZON).value,
            discretisation_type=self.config.discretization
        )
        self.df = pd.DataFrame(columns=pd.Series(self.var_list))

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
                self._set_inputs_to_none()

        # TODO: remove hardcoded strings
        if not self.in_provision:
            if name == "__P_el_base":
                self.data.power_profile_base = self.data.format_mpc_inputs(inp.value)
            elif name == "__P_el_neg":
                self.data.power_profile_flex_neg = self.data.format_mpc_inputs(inp.value)
            elif name == "__P_el_pos":
                self.data.power_profile_flex_pos = self.data.format_mpc_inputs(inp.value)
            elif name == self.config.price_variable:
                # price comes from predictor, so no stripping needed
                # TODO: add other sources for price signal?
                self.data.costs_profile_electricity = self.data.format_predictor_inputs(inp.value)

            if all(var is not None for var in (
                    self.data.power_profile_base,
                    self.data.power_profile_flex_neg,
                    self.data.power_profile_flex_pos,
                    self.data.costs_profile_electricity
            )):
                # Calculate the flexibility, send the offer, write and save the results
                self.flexibility()

                # set the values to None to reset the callback
                self._set_inputs_to_none()

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
                values = self.data.power_profile_base
            elif name == "__P_el_neg":
                values = self.data.power_profile_flex_neg
            elif name == "__P_el_pos":
                values = self.data.power_profile_flex_pos
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
        self.data.calculate()

        # Send flex offer
        self.send_flex_offer(
            name=glbs.FlexibilityOffer,
            base_power_profile=self.data.power_profile_base,
            pos_diff_profile=self.data.kpis_pos.power_flex_offer.value,
            pos_price=self.data.kpis_pos.costs.value,
            neg_diff_profile=self.data.kpis_neg.power_flex_offer.value,
            neg_price=self.data.kpis_neg.costs.value,
        )

        if self.config.discretization == glbs.COLLOCATION:
            # As the collocation uses the values after each time step, the last value is always none
            time = self.base_vals.index[:-1]

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

    def _set_inputs_to_none(self):
        self.data.power_profile_base = None
        self.data.power_profile_flex_neg = None
        self.data.power_profile_flex_pos = None
        self.data.costs_profile_electricity = None
