from typing import Literal, Union

import pydantic
import numpy as np
import pandas as pd

from agentlib_mpc.utils import TimeConversionTypes, TIME_CONVERSION
from flexibility_quantification.data_structures.globals import FlexibilityDirections
from flexibility_quantification.utils.data_handling import strip_multi_index, fill_nans, MEAN, INTERPOLATE




class KPI(pydantic.BaseModel):
    """ Class defining attributes of the indicator KPI. """

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
    direction: FlexibilityDirections = pydantic.Field(
        default=None,
        description="Direction of the shadow mpc"
    )

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, name: str, value: Union[float, pd.Series], unit: str, direction: FlexibilityDirections, **data):
        super().__init__(**data)
        self.name = name
        self.value = value
        self.unit = unit
        self.direction = direction

    def get_name(self):
        name = f"{self.direction}_{self.name}"
        return name

    def integrate(self, time_unit: TimeConversionTypes = "seconds") -> float:
        """
        Integrate the value of the KPI over time by summing up the product of values and the time difference.
        Only possible for pd.Series
        """
        if isinstance(self.value, pd.Series):
            integral = (self.value * self._get_dt(time_unit=time_unit)).sum()
            return integral
        else:
            raise ValueError("Integration only possible for pd.Series")

    def _get_dt(self, time_unit: TimeConversionTypes = "seconds") -> pd.Series:
        """
        Calculate the time difference between the values of the series.
        """
        if isinstance(self.value, pd.Series):
            dt = pd.Series(index=self.value.index, data=self.value.index).diff().shift(-1).ffill()
            dt = dt / TIME_CONVERSION[time_unit]
            return dt
        else:
            raise ValueError("Time difference only possible for Series")


class FlexibilityKPIs(pydantic.BaseModel):
    """
    Class defining the indicator KPIs.
    """
    # Direction
    direction: FlexibilityDirections = pydantic.Field(
        default=None,
        description="Direction of the shadow mpc"
    )

    # Power / energy KPIs
    power_flex_full: KPI = pydantic.Field(
        default=KPI(
            name="power_flex_full",
            direction="",
            value=pd.Series(),
            unit="kW"
        ),
        description="Power flexibility",
    )
    power_flex_offer: KPI = pydantic.Field(
        default=KPI(
            name="power_flex_offer",
            direction="",
            value=pd.Series(),
            unit="kW"
        ),
        description="Power flexibility",
    )
    energy_flex: KPI = pydantic.Field(
        default=KPI(
            name="energy_flex",
            direction="",
            value=0,
            unit="kWh"
        ),
        description="Energy flexibility",
    )

    # Costs KPIs
    costs: KPI = pydantic.Field(
        default=KPI(
            name="costs",
            direction="",
            value=0,
            unit="ct"
        ),
        description="Costs of flexibility",
    )
    costs_rel: KPI = pydantic.Field(
        default=KPI(
            name="costs_rel",
            direction="",
            value=0,
            unit="ct/kWh"
        ),
        description="Costs of flexibility per energy",
    )

    def __init__(self, direction: FlexibilityDirections, **data):
        super().__init__(**data)
        self.direction = direction
        for kpi in vars(self).values():
            if isinstance(kpi, KPI):
                kpi.direction = self.direction

    def calculate(
            self,
            power_profile_base: pd.Series,
            power_profile_flex: pd.Series,
            costs_profile_electricity: pd.Series,
            horizon_full: np.ndarray,
            horizon_offer: np.ndarray
    ):
        """
        Calculate the KPIs based on the power and electricity input profiles.
        Horizons needed for indexing of the power flexibility profiles.
        """
        # Power / energy KPIs
        self.power_flex_full.value = self._calculate_power_flex(power_profile_base=power_profile_base, power_profile_flex=power_profile_flex)
        self.power_flex_offer.value = self.power_flex_full.value.reindex(horizon_offer)
        self.energy_flex.value = self._calculate_energy_flex()

        # Costs KPIs
        self.costs.value = self._calculate_costs(costs_profile_electricity=costs_profile_electricity)
        self.costs_rel.value = self._calculate_costs_rel()

        # change index for offer
        self.power_flex_offer.value = self.power_flex_full.value.reindex(horizon_full)

    def _calculate_power_flex(self, power_profile_base: pd.Series, power_profile_flex: pd.Series) -> pd.Series:
        """
        Calculate the power flexibility based on the base and flexibility power profiles.
        """
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
        """
        Calculate the energy flexibility by integrating the power flexibility of the offer window.
        """
        energy_flex = self.power_flex_offer.integrate(time_unit="hours")
        return energy_flex

    def _calculate_costs(self, costs_profile_electricity: pd.Series) -> float:
        """
        Calculate the costs of the flexibility event based on the electricity costs profile and the power flexibility profile.
        """
        # Check if indices of profiles match
        if not self.power_flex_full.value.index.equals(costs_profile_electricity.index):
            raise ValueError("Indices of profiles do not match")

        # Calculate costs
        costs_series = costs_profile_electricity * self.power_flex_full.value
        costs = KPI(name="", value=costs_series, unit="", direction=self.direction).integrate(time_unit="hours")
        return abs(costs)

    def _calculate_costs_rel(self) -> float:
        """
        Calculate the relative costs of the flexibility event per energy flexibility.
        """
        if self.energy_flex == 0:
            costs_rel = 0
        else:
            costs_rel = self.costs.value / self.energy_flex.value
        return costs_rel

    def get_kpi_dict(self, direction_name: bool = False) -> dict[str, KPI]:
        """
        Get the KPIs as a dictionary with names depending on the direction as keys.
        """
        kpi_dict = {}
        for kpi in vars(self).values():
            if isinstance(kpi, KPI):
                if direction_name:
                    kpi_dict[kpi.get_name()] = kpi
                else:
                    kpi_dict[kpi.name] = kpi
        return kpi_dict

    def get_name_dict(self) -> dict[str, str]:
        """
        Get the KPIs as a dictionary with names depending on the direction as keys.
        """
        name_dict = {}
        for name, kpi in self.get_kpi_dict(direction_name=False).items():
            name_dict[name] = kpi.get_name()
        return name_dict


class FlexibilityData(pydantic.BaseModel):
    """
    Class containing the data for the calculation of the flexibility.
    """

    # Time parameters
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
        description="Base power profile",
    )
    power_profile_flex_neg: pd.Series = pydantic.Field(
        default=None,
        description="Power profile of the negative flexibility",
    )
    power_profile_flex_pos: pd.Series = pydantic.Field(
        default=None,
        description="Power profile of the positive flexibility",
    )
    costs_profile_electricity: pd.Series = pydantic.Field(
        default=None,
        description="Profile of the electricity costs",
    )

    # KPIs
    kpis_pos: FlexibilityKPIs = pydantic.Field(
        default=FlexibilityKPIs(direction="positive"),
        description="KPIs for positive flexibility",
    )
    kpis_neg: FlexibilityKPIs = pydantic.Field(
        default=FlexibilityKPIs(direction="negative"),
        description="KPIs for negative flexibility",
    )

    class Config:
        arbitrary_types_allowed = True

    def __init__(
            self, prep_time: int, market_time: int, flex_event_duration: int,
            time_step: int, prediction_horizon: int,
            **data):
        super().__init__(**data)

        # Define horizons
        switch_time = prep_time + market_time
        self.flex_horizon = np.arange(switch_time, switch_time + flex_event_duration, time_step)
        self.full_horizon = np.arange(0, prediction_horizon * time_step, time_step)

    def format_predictor_inputs(self, series: pd.Series) -> pd.Series:
        series = series.reindex(self.full_horizon)
        return series

    def format_mpc_inputs(self, series: pd.Series) -> pd.Series:
        series = strip_multi_index(series)
        series = fill_nans(series=series, method=MEAN)
        series = series.reindex(self.full_horizon)
        return series

    def calculate(self):
        self.kpis_pos.calculate(
            power_profile_base=self.power_profile_base,
            power_profile_flex=self.power_profile_flex_pos,
            costs_profile_electricity=self.costs_profile_electricity,
            horizon_full=self.full_horizon, horizon_offer=self.flex_horizon
        )
        self.kpis_neg.calculate(
            power_profile_base=self.power_profile_base,
            power_profile_flex=self.power_profile_flex_neg,
            costs_profile_electricity=self.costs_profile_electricity,
            horizon_full=self.full_horizon, horizon_offer=self.flex_horizon
        )
        return self

    def get_kpis(self) -> dict[str, KPI]:
        kpis_dict = self.kpis_pos.get_kpi_dict(direction_name=True) | self.kpis_neg.get_kpi_dict(direction_name=True)
        return kpis_dict
