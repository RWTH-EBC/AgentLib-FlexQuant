import os
import logging
from typing import Optional, List
import agentlib
import numpy as np
import pandas as pd
from pathlib import Path
from pydantic import BaseModel, Field, field_validator


import flexibility_quantification.data_structures.globals as glbs
from flexibility_quantification.data_structures.flex_kpis import FlexibilityData, FlexibilityKPIs
from flexibility_quantification.data_structures.flex_offer import FlexOffer


class InputsForCorrectFlexCosts(BaseModel):
    enable_energy_costs_correction: bool = Field(
        name="enable_energy_costs_correction",
        description="Variable determining whether to correct the costs of the flexible energy"
                    "Define the storage variable in the base MPC model and config as output if the correction of costs is enabled",
        default=False
    )

    absolute_power_deviation_tolerance: float = Field(
        name="absolute_power_deviation_tolerance",
        default=0.1,
        description="Absolute tolerance in kW within which no warning is thrown"
    )

    stored_energy_variable: str = Field(
        name="stored_energy_variable",
        default="E_stored",
        description="Name of the variable representing the stored electrical energy in the baseline config"
    )


# Pos and neg kpis to get the right names for plotting
kpis_pos = FlexibilityKPIs(direction="positive")
kpis_neg = FlexibilityKPIs(direction="negative")


class FlexibilityIndicatorModuleConfig(agentlib.BaseModuleConfig):
    inputs: List[agentlib.AgentVariable] = [
        agentlib.AgentVariable(name=glbs.POWER_ALIAS_BASE, unit="W", type="pd.Series",
                               description="The power input to the system"),
        agentlib.AgentVariable(name=glbs.POWER_ALIAS_NEG, unit="W", type="pd.Series",
                               description="The power input to the system"),
        agentlib.AgentVariable(name=glbs.POWER_ALIAS_POS, unit="W", type="pd.Series",
                               description="The power input to the system"),
        agentlib.AgentVariable(name="r_pel", unit="ct/kWh", type="pd.Series",
                               description="electricity price"),
        agentlib.AgentVariable(name=glbs.STORED_ENERGY_ALIAS_BASE, unit="kWh", type="pd.Series",
                               description="Energy stored in the system w.r.t. 0K"),
        agentlib.AgentVariable(name=glbs.STORED_ENERGY_ALIAS_NEG, unit="kWh", type="pd.Series",
                               description="Energy stored in the system w.r.t. 0K"),
        agentlib.AgentVariable(name=glbs.STORED_ENERGY_ALIAS_POS, unit="kWh", type="pd.Series",
                               description="Energy stored in the system w.r.t. 0K")
    ]

    outputs: List[agentlib.AgentVariable] = [
        # Flexibility offer
        agentlib.AgentVariable(name=glbs.FlexibilityOffer, type="FlexOffer"),

        # Power KPIs
        agentlib.AgentVariable(
            name=kpis_neg.power_flex_full.get_kpi_identifier(), unit='W', type="pd.Series",
            description="Negative power flexibility"
        ),
        agentlib.AgentVariable(
            name=kpis_pos.power_flex_full.get_kpi_identifier(), unit='W', type="pd.Series",
            description="Positive power flexibility"
        ),
        agentlib.AgentVariable(
            name=kpis_neg.power_flex_offer.get_kpi_identifier(), unit='W', type="pd.Series",
            description="Negative power flexibility"
        ),
        agentlib.AgentVariable(
            name=kpis_pos.power_flex_offer.get_kpi_identifier(), unit='W', type="pd.Series",
            description="Positive power flexibility"
        ),
        agentlib.AgentVariable(
            name=kpis_neg.power_flex_offer_min.get_kpi_identifier(), unit='W', type="float",
            description="Minimum of negative power flexibility"
        ),
        agentlib.AgentVariable(
            name=kpis_pos.power_flex_offer_min.get_kpi_identifier(), unit='W', type="float",
            description="Minimum of positive power flexibility"
        ),
        agentlib.AgentVariable(
            name=kpis_neg.power_flex_offer_max.get_kpi_identifier(), unit='W', type="float",
            description="Maximum of negative power flexibility"
        ),
        agentlib.AgentVariable(
            name=kpis_pos.power_flex_offer_max.get_kpi_identifier(), unit='W', type="float",
            description="Maximum of positive power flexibility"
        ),
        agentlib.AgentVariable(
            name=kpis_neg.power_flex_offer_avg.get_kpi_identifier(), unit='W', type="float",
            description="Average of negative power flexibility"
        ),
        agentlib.AgentVariable(
            name=kpis_pos.power_flex_offer_avg.get_kpi_identifier(), unit='W', type="float",
            description="Average of positive power flexibility"
        ),
        agentlib.AgentVariable(
            name=kpis_neg.power_flex_within_boundary.get_kpi_identifier(), unit='-', type="bool",
            description="Variable indicating whether the baseline power and flex power align at the horizon end"
        ),
        agentlib.AgentVariable(
            name=kpis_pos.power_flex_within_boundary.get_kpi_identifier(), unit='-', type="bool",
            description="Variable indicating whether the baseline power and flex power align at the horizon end"
        ),

        # Energy KPIs
        agentlib.AgentVariable(
            name=kpis_neg.energy_flex.get_kpi_identifier(), unit='kWh', type="float",
            description="Negative energy flexibility"
        ),
        agentlib.AgentVariable(
            name=kpis_pos.energy_flex.get_kpi_identifier(), unit='kWh', type="float",
            description="Positive energy flexibility"
        ),

        # Costs KPIs
        agentlib.AgentVariable(
            name=kpis_neg.costs.get_kpi_identifier(), unit="ct", type="float",
            description="Saved costs due to baseline"
        ),
        agentlib.AgentVariable(
            name=kpis_pos.costs.get_kpi_identifier(), unit="ct", type="float",
            description="Saved costs due to baseline"
        ),
        agentlib.AgentVariable(
            name=kpis_neg.corrected_costs.get_kpi_identifier(), unit="ct", type="float",
            description="Corrected saved costs due to baseline"
        ),
        agentlib.AgentVariable(
            name=kpis_pos.corrected_costs.get_kpi_identifier(), unit="ct", type="float",
            description="Corrected saved costs due to baseline"
        ),
        agentlib.AgentVariable(
            name=kpis_neg.costs_rel.get_kpi_identifier(), unit='ct/kWh', type="float",
            description="Saved costs due to baseline"
        ),
        agentlib.AgentVariable(
            name=kpis_pos.costs_rel.get_kpi_identifier(), unit='ct/kWh', type="float",
            description="Saved costs due to baseline"
        ),
        agentlib.AgentVariable(
            name=kpis_neg.corrected_costs_rel.get_kpi_identifier(), unit='ct/kWh', type="float",
            description="Corrected saved costs per energy due to baseline"
        ),
        agentlib.AgentVariable(
            name=kpis_pos.corrected_costs_rel.get_kpi_identifier(), unit='ct/kWh', type="float",
            description="Corrected saved costs per energy due to baseline"
        )
    ]


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
                               description="prediction horizon of the mpc solution")
    ]

    results_file: Optional[Path] = Field(default=None)

    save_results: Optional[bool] = Field(validate_default=True, default=None)
    overwrite_result_file: Optional[bool] = Field(default=False, validate_default=True)

    price_variable: str = Field(
        default="r_pel",
        description="Name of the price variable sent by a predictor",
    )
    power_unit: str = Field(
        default="kW",
        description="Unit of the power variable"
    )

    shared_variable_fields: List[str] = ["outputs"]

    correct_costs: InputsForCorrectFlexCosts = InputsForCorrectFlexCosts()


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
            prediction_horizon=self.get(glbs.PREDICTION_HORIZON).value
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

        if not self.in_provision:
            if name == glbs.POWER_ALIAS_BASE:
                self.data.power_profile_base = self.data.format_mpc_inputs(inp.value)
            elif name == glbs.POWER_ALIAS_NEG:
                self.data.power_profile_flex_neg = self.data.format_mpc_inputs(inp.value)
            elif name == glbs.POWER_ALIAS_POS:
                self.data.power_profile_flex_pos = self.data.format_mpc_inputs(inp.value)
            elif name == glbs.STORED_ENERGY_ALIAS_BASE:
                self.data.stored_energy_profile_base = self.data.format_mpc_inputs(inp.value)
            elif name == glbs.STORED_ENERGY_ALIAS_NEG:
                self.data.stored_energy_profile_flex_neg = self.data.format_mpc_inputs(inp.value)
            elif name == glbs.STORED_ENERGY_ALIAS_POS:
                self.data.stored_energy_profile_flex_pos = self.data.format_mpc_inputs(inp.value)
            elif name == self.config.price_variable:
                # price comes from predictor, so no stripping needed
                self.data.electricity_price_series = self.data.format_predictor_inputs(inp.value)

            necessary_input_for_calc_flex = [self.data.power_profile_base,
                                             self.data.power_profile_flex_neg,
                                             self.data.power_profile_flex_pos,
                                             self.data.electricity_price_series]
            if self.config.correct_costs.enable_energy_costs_correction:
                necessary_input_for_calc_flex.extend(
                                                [self.data.stored_energy_profile_base,
                                                 self.data.stored_energy_profile_flex_neg,
                                                 self.data.stored_energy_profile_flex_pos])

            if all(var is not None for var in necessary_input_for_calc_flex):

                # check the power profile end deviation
                if not self.config.correct_costs.enable_energy_costs_correction:
                    self.check_power_end_deviation(tol=self.config.correct_costs.absolute_power_deviation_tolerance)

                # Calculate the flexibility, send the offer, write and save the results
                self.calc_and_send_offer()

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
            if name == glbs.POWER_ALIAS_BASE:
                values = self.data.power_profile_base
            elif name == glbs.POWER_ALIAS_NEG:
                values = self.data.power_profile_flex_neg
            elif name == glbs.POWER_ALIAS_POS:
                values = self.data.power_profile_flex_pos
            elif name == glbs.STORED_ENERGY_ALIAS_BASE:
                values = self.data.stored_energy_profile_base
            elif name == glbs.STORED_ENERGY_ALIAS_NEG:
                values = self.data.stored_energy_profile_flex_neg
            elif name == glbs.STORED_ENERGY_ALIAS_POS:
                values = self.data.stored_energy_profile_flex_pos
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

    def calc_and_send_offer(self):
        """
        Calculate the flexibility KPIs for current predictions, send the flex offer and set the outputs, write and save the results.
        """
        # Calculate the flexibility KPIs for current predictions
        self.data.calculate(enable_energy_costs_correction=self.config.correct_costs.enable_energy_costs_correction)

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
            if kpi.get_kpi_identifier() not in [kpis_pos.power_flex_within_boundary.get_kpi_identifier(), kpis_neg.power_flex_within_boundary.get_kpi_identifier()]:
                for output in self.config.outputs:
                    if output.name == kpi.get_kpi_identifier():
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
        since the different MPCs need one time step to fully initialize.

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
        self.data.electricity_price_series = None
        self.data.stored_energy_profile_base = None
        self.data.stored_energy_profile_flex_neg = None
        self.data.stored_energy_profile_flex_pos = None

    def check_power_end_deviation(self, tol: float):
        """
        calculates the deviation of the final value of the power profiles and warn the user if it exceeds the tolerance
        """
        logger = logging.getLogger(__name__)
        dev_pos = np.mean(self.data.power_profile_flex_pos.values[-4:] - self.data.power_profile_base.values[-4:])
        dev_neg = np.mean(self.data.power_profile_flex_neg.values[-4:] - self.data.power_profile_base.values[-4:])
        if abs(dev_pos) > tol:
            logger.warning(f"There is an average deviation of {dev_pos:.6f} kW between the final values of power profiles of positive shadow MPC and the baseline. Correction of energy costs might be necessary.")
            self.set(kpis_pos.power_flex_within_boundary.get_kpi_identifier(), False)
        else:
            self.set(kpis_pos.power_flex_within_boundary.get_kpi_identifier(), True)
        if abs(dev_neg) > tol:
            logger.warning(f"There is an average deviation of {dev_pos:.6f} kW between the final values of power profiles of negative shadow MPC and the baseline. Correction of energy costs might be necessary.")
            self.set(kpis_neg.power_flex_within_boundary.get_kpi_identifier(), False)
        else:
            self.set(kpis_neg.power_flex_within_boundary.get_kpi_identifier(), True)

