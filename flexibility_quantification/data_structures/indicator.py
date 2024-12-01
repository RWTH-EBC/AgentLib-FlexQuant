from typing import Literal, Union

import pydantic
import numpy as np
import pandas as pd

from pint import UnitRegistry, Quantity

from agentlib_mpc.utils import TimeConversionTypes, TIME_CONVERSION

ShadowDirections = Literal["positive", "negative"]

Collocation: str = "collocation"
MultipleShooting: str = "multiple_shooting"
DiscretizationTypes = Literal["collocation", "multiple_shooting"]

# todo: Units
ureg = UnitRegistry()


class KPI(pydantic.BaseModel):
    """
    Class defining attributes of the indicator KPI.
    """
    name: str = pydantic.Field(
        default=None,
        description="Name of the indicator KPI",
    )
    value: Union[float, pd.Series] = pydantic.Field(
        default=None,
        description="Value of the indicator KPI",
    )
    unit: str = pydantic.Field(
        default=None,
        description="Unit of the indicator KPI",
    )
    direction: ShadowDirections = pydantic.Field(
        default=None,
        description="Direction of the shadow mpc"
    )

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, name: str, value: Union[float, pd.Series], unit: str, direction: ShadowDirections, **data):
        super().__init__(**data)
        self.name = name
        self.value = value
        self.unit = unit
        self.direction = direction

    def integrate(self, time_conv: TimeConversionTypes = "seconds") -> float:
        """
        Integrate the value of the KPI over time by summing up the product of values and the time difference.
        Only possible for pd.Series
        """
        if isinstance(self.value, pd.Series):
            integral = (self.value * self._get_dt(unit=time_conv)).sum()
            return integral
        else:
            raise ValueError("Integration only possible for pd.Series")

    def _get_dt(self, unit: TimeConversionTypes = "seconds") -> pd.Series:
        """
        Calculate the time difference between the values of the series.
        """
        if isinstance(self.value, pd.Series):
            dt = pd.Series(index=self.value.index, data=self.value.index).diff().shift(-1).ffill()
            dt = dt  / TIME_CONVERSION[unit]
            return dt
        else:
            raise ValueError("Time difference only possible for Series")

    def get_name(self):
        name = f"{self.direction}_{self.name}"
        return name


class IndicatorKPIs(pydantic.BaseModel):
    """
    Class defining the indicator KPIs.
    """
    # Direction
    direction: ShadowDirections = pydantic.Field(
        default=None,
        description="Direction of the shadow mpc"
    )

    # Power / energy KPIs
    power_flex_full: KPI = pydantic.Field(
        default=KPI(
            name="power_flex_full",
            value=pd.Series(),
            unit="kW",
            direction=""
        ),
        description="Power flexibility",
    )
    power_flex_offer: KPI = pydantic.Field(
        default=KPI(
            name="power_flex_offer",
            value=pd.Series(),
            unit="kW",
            direction=""
        ),
        description="Power flexibility",
    )
    energy_flex: KPI = pydantic.Field(
        default=KPI(
            name="energy_flex",
            value=0,
            unit="kWh",
            direction=""
        ),
        description="Energy flexibility",
    )

    # Costs KPIs
    costs: KPI = pydantic.Field(
        default=KPI(
            name="costs",
            value=0,
            unit="ct",
            direction=""
        ),
        description="Costs of flexibility",
    )
    costs_rel: KPI = pydantic.Field(
        default=KPI(
            name="costs_rel",
            value=0,
            unit="ct/kWh",
            direction=""
        ),
        description="Costs of flexibility per energy",
    )

    def __init__(self, direction: ShadowDirections, **data):
        super().__init__(**data)
        self.direction = direction
        for kpi in vars(self).values():
            if isinstance(kpi, KPI):
                kpi.direction = self.direction

    def calculate(
            self,
            power_profile_base: pd.Series,
            power_profile_flex: pd.Series,
            power_unit: str,
            costs_profile_electricity: pd.Series,
            horizon_full: np.ndarray,
            horizon_offer: np.ndarray
    ):
        # Power / energy KPIs
        self.power_flex_full.value = self._calculate_power_flex(power_profile_base=power_profile_base, power_profile_flex=power_profile_flex, power_unit=power_unit)
        self.power_flex_offer.value = self.power_flex_full.value.reindex(horizon_offer)
        self.energy_flex.value = self._calculate_energy_flex()

        # Costs KPIs
        self.costs.value = self._calculate_costs(costs_profile_electricity=costs_profile_electricity)
        self.costs_rel.value = self._calculate_costs_rel()

    def _calculate_power_flex(self, power_profile_base: pd.Series, power_profile_flex: pd.Series, power_unit: str) -> pd.Series:
        # Check if indices of profiles match
        if not power_profile_flex.index.equals(power_profile_base.index):
            raise ValueError("Indices of power profiles do not match")

        if self.direction == "positive":
            power_flex = power_profile_base - power_profile_flex
        elif self.direction == "negative":
            power_flex = power_profile_flex - power_profile_base
        else:
            raise ValueError("Direction of KPIs not defined")

        # Set values to zero if the difference is below 1% of the base profile
        relative_difference = (power_flex / power_profile_base).abs()
        power_flex.loc[(power_flex < 0) & (relative_difference < 0.01)] = 0
        return power_flex

    def _calculate_energy_flex(self) -> float:
        energy_flex = self.power_flex_offer.integrate(time_conv="hours")
        return energy_flex

    def _calculate_costs(self, costs_profile_electricity: pd.Series) -> float:
        # Check if indices of profiles match
        if not self.power_flex_full.value.index.equals(costs_profile_electricity.index):
            raise ValueError("Indices of profiles do not match")

        # Calculate costs
        costs_series = costs_profile_electricity * self.power_flex_full.value
        costs = KPI(name=self.costs.name, value=costs_series, unit=self.costs.unit, direction=self.direction).integrate(time_conv="hours")
        return costs

    def _calculate_costs_rel(self) -> float:
        if self.energy_flex == 0:
            costs_rel = 0
        else:
            costs_rel = self.costs.value / self.energy_flex.value
        return costs_rel

    def get_kpi_dict(self) -> dict[str, KPI]:
        kpi_dict = {}
        for kpi in vars(self).values():
            if isinstance(kpi, KPI):
                kpi_dict[kpi.get_name()] = kpi
        return kpi_dict

class IndicatorData(pydantic.BaseModel):
    """
    Class
    """
    # Time parameters
    prep_time: int = pydantic.Field(
        default=1800,
        ge=0,
        unit="s",
        description="Preparation time before the flexibility event",
    )
    market_time: int = pydantic.Field(
        default=900,
        ge=0,
        unit="s",
        description="Time for market interaction",
    )
    flex_event_duration: int = pydantic.Field(
        default=7200,
        ge=0,
        unit="s",
        description="Flexibility event duration",
    )
    time_step: int = pydantic.Field(
        default=900,
        ge=0,
        unit="s",
        description="Time step of the simulation",
    )
    prediction_horizon: int = pydantic.Field(
        default=96,
        ge=0,
        unit="-",
        description="prediction horizon of the mpc solution"
    )
    full_horizon: np.ndarray = pydantic.Field(
        default=None,
        description="Full horizon of the simulation",
    )
    flex_horizon: np.ndarray = pydantic.Field(
        default=None,
        description="Flexibility horizon",
    )
    # Discretisation
    discretisation_type: DiscretizationTypes = pydantic.Field(
        default=Collocation,
        description="Type of discretisation",
    )

    # Profiles
    power_profile_base: pd.Series = pydantic.Field(
        default=None,
        description="Base profile of the flexibility event",
    )
    power_profile_flex_neg: pd.Series = pydantic.Field(
        default=None,
        description="Negative flexibility profile",
    )
    power_profile_flex_pos: pd.Series = pydantic.Field(
        default=None,
        description="Positive flexibility profile",
    )
    costs_profile_electricity: pd.Series = pydantic.Field(
        default=None,
        description="Costs of the flexibility event",
    )

    # KPIs
    kpis_pos: IndicatorKPIs = pydantic.Field(
        default=IndicatorKPIs(direction="positive"),
        description="KPIs for positive flexibility",
    )
    kpis_neg: IndicatorKPIs = pydantic.Field(
        default=IndicatorKPIs(direction="negative"),
        description="KPIs for negative flexibility",
    )

    class Config:
        arbitrary_types_allowed = True

    def __init__(
            self, prep_time: int, market_time: int, flex_event_duration: int,
            time_step: int, prediction_horizon: int,
            discretisation_type: DiscretizationTypes,
            **data):
        super().__init__(**data)

        self.prep_time = prep_time
        self.market_time = market_time
        self.flex_event_duration = flex_event_duration
        self.time_step = time_step
        self.prediction_horizon = prediction_horizon
        
        switch_time = prep_time + market_time
        self.flex_horizon = np.arange(switch_time, switch_time + flex_event_duration, time_step)
        self.full_horizon = np.arange(0, prediction_horizon * time_step, time_step)

        self.discretisation_type = discretisation_type

    def calculate(
            self,
            power_profile_base: pd.Series,
            power_profile_flex_pos: pd.Series,
            power_profile_flex_neg: pd.Series,
            power_unit: str,
            costs_profile_electricity: pd.Series
    ):
        self.uniform_input_data(
            power_profile_base=power_profile_base,
            power_profile_flex_pos=power_profile_flex_pos,
            power_profile_flex_neg=power_profile_flex_neg,
            costs_profile_electricity=costs_profile_electricity
        )

        self.kpis_pos.calculate(
            power_profile_base=self.power_profile_base,
            power_profile_flex=self.power_profile_flex_pos,
            power_unit=power_unit,
            costs_profile_electricity=self.costs_profile_electricity,
            horizon_full=self.full_horizon, horizon_offer=self.flex_horizon
        )
        self.kpis_neg.calculate(
            power_profile_base=self.power_profile_base,
            power_profile_flex=self.power_profile_flex_neg,
            power_unit=power_unit,
            costs_profile_electricity=self.costs_profile_electricity,
            horizon_full=self.full_horizon, horizon_offer=self.flex_horizon
        )
        return self

    def uniform_input_data(
            self,
            power_profile_base: pd.Series,
            power_profile_flex_pos: pd.Series,
            power_profile_flex_neg: pd.Series,
            costs_profile_electricity: pd.Series
    ):
        # todo
        if self.discretisation_type == Collocation:
            self.power_profile_base = power_profile_base.reindex(self.full_horizon)
            self.power_profile_flex_pos = power_profile_flex_pos.reindex(self.full_horizon)
            self.power_profile_flex_neg = power_profile_flex_neg.reindex(self.full_horizon)
            self.costs_profile_electricity = costs_profile_electricity.reindex(self.full_horizon)

    def get_kpis(self) -> dict[str, KPI]:
        kpis_dict = self.kpis_pos.get_kpi_dict() | self.kpis_neg.get_kpi_dict()
        return kpis_dict