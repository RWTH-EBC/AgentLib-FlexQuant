from typing import Union, Literal, Optional

import pydantic
import numpy as np
import pandas as pd

KPITypes = Literal["positive", "negative"]


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
    type: str = pydantic.Field(
        default=None,
        description="Type of the indicator KPI",
    )

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, name: str, value: Union[float, pd.Series], unit: str, type: str, **data):
        super().__init__(**data)
        self.name = name
        self.value = value
        self.unit = unit
        self.type = type

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

    def integrate(self) -> float:
        """
        Integrate the value of the KPI over time by summing up the product of values and the time difference.
        Only possible for pd.Series
        """
        if isinstance(self.value, pd.Series):
            return (self.value * self._get_dt()).sum()
        else:
            raise ValueError("Integration only possible for pd.Series")

    def _get_dt(self) -> float:     # -> pd.Series:
        """
        Calculate the time difference between the values of the series.
        """
        if isinstance(self.value, pd.Series):
            dt = self.value.index[1] - self.value.index[0]  # pd.Series(index=self.value.index, data=self.value.index).diff().shift(-1).ffill()
            return dt
        else:
            raise ValueError("Integration only possible for pd.Series")


class IndicatorKPIs(pydantic.BaseModel):
    """
    Class defining the indicator KPIs.
    """
    power_flex: KPI = pydantic.Field(
        default=KPI(
            name="power_flex",
            value=pd.Series(),
            unit="kW",
            type="positive"
        ),
        description="Power flexibility",
    )
    energy_flex: KPI = pydantic.Field(
        default=KPI(
            name="energy_flex",
            value=0,
            unit="kWh",
            type="positive"
        ),
        description="Energy flexibility",
    )
    costs: KPI = pydantic.Field(
        default=KPI(
            name="costs",
            value=0,
            unit="ct",
            type="positive"
        ),
        description="Costs of flexibility",
    )
    costs_rel: KPI = pydantic.Field(
        default=KPI(
            name="costs_rel",
            value=0,
            unit="ct/kWh",
            type="positive"
        ),
        description="Costs of flexibility per energy",
    )

    def __init__(self, type: KPITypes, **data):
        super().__init__(**data)
        self.power_flex.type = type
        self.energy_flex.type = type
        self.costs.type = type
        self.costs_rel.type = type

    def calculate(self, power_profile_base: pd.Series, power_profile_flex: pd.Series, costs_profile_electricity: pd.Series):
        self.power_flex.value = self._calculate_powerflex(power_profile_base=power_profile_base, power_profile_flex=power_profile_flex)
        self.energy_flex.value = self._calculate_energyflex()
        self.costs.value = self._calculate_costs(costs_profile_electricity=costs_profile_electricity)
        self.costs_rel.value = self._calculate_costs_rel()

    def _calculate_powerflex(self, power_profile_base: pd.Series, power_profile_flex: pd.Series) -> pd.Series:
        difference = power_profile_flex - power_profile_base
        difference_rel = (difference / power_profile_base).abs() * 100
        difference.loc[(difference < 0) & (difference_rel < 1)] = 0
        return difference

    def _calculate_energyflex(self) -> float:
        energyflex = self.power_flex.integrate()
        return energyflex

    def _calculate_costs(self, costs_profile_electricity: pd.Series) -> pd.Series:
        costs = costs_profile_electricity * self.power_flex.value
        return costs

    def _calculate_costs_rel(self) -> float:
        if self.energy_flex == 0:
            costs_rel = 0
        else:
            costs_rel = self.costs.integrate() / self.energy_flex.value
        return costs_rel


class IndicatorCalculator(pydantic.BaseModel):
    """
    Class
    """
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
    pos_kpis: IndicatorKPIs = pydantic.Field(
        default=IndicatorKPIs(type="positive"),
        description="KPIs for positive flexibility",
    )
    neg_kpis: IndicatorKPIs = pydantic.Field(
        default=IndicatorKPIs(type="negative"),
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
        self.flex_horizon, self.full_horizon = self._generate_horizons()

    def _generate_horizons(self) -> tuple[np.ndarray, np.ndarray]:
        switch_time = self.prep_time + self.market_time
        # generate horizons
        # 1. for the flexibility range
        flex_horizon = np.arange(
            switch_time, switch_time + self.flex_event_duration, self.time_step)
        # 2. for the full range of prediction
        full_horizon = np.arange(
            0, self.prediction_horizon * self.time_step, self.time_step)
        return flex_horizon, full_horizon
    
    def _uniform_data(self):   # todo
        """Uniform data to the same length
        """
        # uniform regarding discretization
        if True: # self.config.discretization == "collocation":
            # As the collocation uses the values after each time step, the last value is always none
            time = self.power_profile_base.index[:-1]

            # use only the values of the full time steps
            self.power_profile_base = pd.Series(self.power_profile_base, index=time).reindex(index=self.full_horizon)
            self.power_profile_flex_neg = pd.Series(self.power_profile_flex_neg, index=time).reindex(index=self.full_horizon)
            self.power_profile_flex_pos = pd.Series(self.power_profile_flex_pos, index=time).reindex(index=self.full_horizon)
            self.costs_profile_electricity = pd.Series(self.costs_profile_electricity, index=time).reindex(index=self.full_horizon)

    def calculate(self, power_profile_base: pd.Series, power_profile_flex_pos: pd.Series, power_profile_flex_neg: pd.Series, costs_profile_electricity: pd.Series):
        self.power_profile_base = power_profile_base
        self.power_profile_flex_pos = power_profile_flex_pos
        self.power_profile_flex_neg = power_profile_flex_neg
        self.costs_profile_electricity = costs_profile_electricity

        self._uniform_data()

        self.pos_kpis.calculate(power_profile_base=self.power_profile_base, power_profile_flex=self.power_profile_flex_pos, costs_profile_electricity=self.costs_profile_electricity)
        self.neg_kpis.calculate(power_profile_base=self.power_profile_base, power_profile_flex=self.power_profile_flex_pos, costs_profile_electricity=self.costs_profile_electricity)
        pass

