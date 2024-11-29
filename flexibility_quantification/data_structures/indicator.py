from typing import Union, Literal, Optional

import pydantic
import numpy as np
import pandas as pd

from pint import UnitRegistry, Quantity, Unit

from agentlib_mpc.utils import TimeConversionTypes, TIME_CONVERSION

KPIDirections = Literal["positive", "negative"]

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

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, name: str, value: Union[float, pd.Series], unit: str, **data):
        super().__init__(**data)
        self.name = name
        self.value = value
        self.unit = unit

    def min(self) -> float:
        if isinstance(self.value, pd.Series):
            return self.value.min()
        else:
            return self.value

    def max(self) -> float:
        if isinstance(self.value, pd.Series):
            return self.value.max()
        else:
            return self.value

    def mean(self) -> float:
        if isinstance(self.value, pd.Series):
            return self.integrate() / (self.value.index[-1] - self.value.index[0])
        else:
            return self.value

    def integrate(self, time_conv: TimeConversionTypes = "seconds") -> float:
        """
        Integrate the value of the KPI over time by summing up the product of values and the time difference.
        Only possible for pd.Series
        """
        if isinstance(self.value, pd.Series):
            return (self.value * self._get_dt()).sum() / TIME_CONVERSION[time_conv]
        else:
            raise ValueError("Integration only possible for pd.Series")

    def _get_dt(self) -> pd.Series:
        """
        Calculate the time difference between the values of the series.
        """
        if isinstance(self.value, pd.Series):
            dt = pd.Series(index=self.value.index, data=self.value.index).diff().shift(-1).ffill()
            return dt
        else:
            raise ValueError("Time difference only possible for Series")

    def _convert_units(self, magnitude: Union[float, pd.Series], is_unit: str, to_unit: str) -> Union[float, pd.Series]:
        if isinstance(magnitude, pd.Series):
            series = magnitude.apply(self._convert_units, is_unit=is_unit, to_unit=to_unit)
            return series
        else:
            scalar = Quantity(magnitude, is_unit).to(to_unit).magnitude
            return scalar


class IndicatorKPIs(pydantic.BaseModel):
    """
    Class defining the indicator KPIs.
    """
    # Parameter
    direction: KPIDirections = pydantic.Field(
        default=None,
        description="Direction of the shadow mpc",
    )

    # KPIs
    power_flex_full: KPI = pydantic.Field(
        default=KPI(
            name="power_flex_full",
            value=pd.Series(),
            unit="kW"
        ),
        description="Power flexibility",
    )
    power_flex_offer: KPI = pydantic.Field(
        default=KPI(
            name="power_flex_offer",
            value=pd.Series(),
            unit="kW"
        ),
        description="Power flexibility",
    )
    energy_flex: KPI = pydantic.Field(
        default=KPI(
            name="energy_flex",
            value=0,
            unit="kWh"
        ),
        description="Energy flexibility",
    )
    costs: KPI = pydantic.Field(
        default=KPI(
            name="costs",
            value=0,
            unit="ct"
        ),
        description="Costs of flexibility",
    )
    costs_rel: KPI = pydantic.Field(
        default=KPI(
            name="costs_rel",
            value=0,
            unit="ct/kWh"
        ),
        description="Costs of flexibility per energy",
    )

    def __init__(self, direction: KPIDirections, **data):
        super().__init__(**data)
        self.direction = direction

    def calculate(self, power_profile_base: pd.Series, power_profile_flex: pd.Series, power_unit: str,
                  costs_profile_electricity: pd.Series,
                  horizon_full: np.ndarray, horizon_offer: np.ndarray):
        self.power_flex_full.value = self._calculate_powerflex(power_profile_base=power_profile_base, power_profile_flex=power_profile_flex, power_unit=power_unit)
        self.power_flex_offer.value = self.power_flex_full.value.reindex(horizon_offer)
        self.energy_flex.value = self._calculate_energyflex()
        self.costs.value = self._calculate_costs(costs_profile_electricity=costs_profile_electricity)
        self.costs_rel.value = self._calculate_costs_rel()

    def _calculate_powerflex(self, power_profile_base: pd.Series, power_profile_flex: pd.Series, power_unit: str) -> pd.Series:
        if not power_profile_flex.index.equals(power_profile_base.index):
            raise ValueError("Indices of power profiles do not match")

        if self.direction == "positive":
            difference = power_profile_base - power_profile_flex
        elif self.direction == "negative":
            difference = power_profile_flex - power_profile_base
        else:
            raise ValueError("Direction of KPIs not defined")

        difference_rel = (difference / power_profile_base).abs()
        difference.loc[(difference < 0) & (difference_rel < 0.01)] = 0
        difference = difference.apply(self._convert_units, is_unit=power_unit, to_unit=self.power_flex_full.unit)
        print(difference)

        return difference

    def _calculate_energyflex(self) -> float:
        energy_flex = self.power_flex_offer.integrate(time_conv="hours")
        return energy_flex

    def _calculate_costs(self, costs_profile_electricity: pd.Series) -> float:
        costs_series = costs_profile_electricity * self.power_flex_full.value
        costs = KPI(name=self.costs.name, value=costs_series, unit=self.costs.unit).integrate(time_conv="hours")
        return costs

    def _calculate_costs_rel(self) -> float:
        if self.energy_flex == 0:
            costs_rel = 0
        else:
            costs_rel = self.costs.value / self.energy_flex.value
        return costs_rel

    def _convert_units(self, x: float, is_unit: str, to_unit: str) -> float:
        return Quantity(x, is_unit).to(to_unit).magnitude


class IndicatorData(pydantic.BaseModel):
    """
    Class
    """
    # Parameter
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

    def __init__(self, prep_time: int, market_time: int, flex_event_duration: int, time_step: int, prediction_horizon: int, **data):
        super().__init__(**data)

        self.prep_time = prep_time
        self.market_time = market_time
        self.flex_event_duration = flex_event_duration
        self.time_step = time_step
        self.prediction_horizon = prediction_horizon
        
        switch_time = prep_time + market_time
        self.flex_horizon = np.arange(switch_time, switch_time + flex_event_duration, time_step)
        self.full_horizon = np.arange(0, prediction_horizon * time_step, time_step)

    def calculate(
            self,
            power_profile_base: pd.Series, power_profile_flex_pos: pd.Series, power_profile_flex_neg: pd.Series, power_unit: str,
            costs_profile_electricity: pd.Series):
        self.power_profile_base = power_profile_base
        self.power_profile_flex_pos = power_profile_flex_pos
        self.power_profile_flex_neg = power_profile_flex_neg
        self.costs_profile_electricity = costs_profile_electricity

        self.kpis_pos.calculate(
            power_profile_base=self.power_profile_base, power_profile_flex=self.power_profile_flex_pos, power_unit=power_unit,
            costs_profile_electricity=self.costs_profile_electricity,
            horizon_full=self.full_horizon, horizon_offer=self.flex_horizon
        )
        self.kpis_neg.calculate(
            power_profile_base=self.power_profile_base, power_profile_flex=self.power_profile_flex_neg, power_unit=power_unit,
            costs_profile_electricity=self.costs_profile_electricity,
            horizon_full=self.full_horizon, horizon_offer=self.flex_horizon
        )
        return self
