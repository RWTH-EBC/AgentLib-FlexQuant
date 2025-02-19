from typing import Union, Optional

import numpy
import pydantic
import numpy as np
import pandas as pd

from agentlib_mpc.utils import TimeConversionTypes, TIME_CONVERSION
from flexibility_quantification.data_structures.globals import FlexibilityDirections
from flexibility_quantification.utils.data_handling import strip_multi_index, fill_nans, MEAN


class KPI(pydantic.BaseModel):
    """ Class defining attributes of the indicator KPI. """

    name: str = pydantic.Field(
        default=None,
        description="Name of the flexibility KPI",
    )
    value: Union[float, None] = pydantic.Field(
        default=None,
        description="Value of the flexibility KPI",
    )
    unit: str = pydantic.Field(
        default=None,
        description="Unit of the flexibility KPI",
    )
    direction: Union[FlexibilityDirections, None] = pydantic.Field(
        default=None,
        description="Direction of the shadow mpc / flexibility"
    )

    class Config:
        arbitrary_types_allowed = True

    def get_kpi_identifier(self):
        name = f"{self.direction}_{self.name}"
        return name


class KPISeries(KPI):
    value: Union[pd.Series, None] = pydantic.Field(
        default=None,
        description="Value of the flexibility KPI",
    )
    dt: Union[pd.Series, None] = pydantic.Field(
        default=None,
        description="Time differences between the timestamps of the series in seconds",
    )

    def _get_dt(self) -> pd.Series:
        """
        Get the time differences between the timestamps of the series.
        """
        dt = pd.Series(index=self.value.index, data=self.value.index).diff().shift(-1).ffill()
        self.dt = dt
        return dt

    def min(self) -> float:
        return self.value.min()

    def max(self) -> float:
        return self.value.max()

    def avg(self) -> float:
        """
        Calculate the average value of the KPI over time.
        """
        if self.dt is None:
            self._get_dt()
        delta_t = self.dt.sum()
        avg = self.integrate() / delta_t
        return avg

    def integrate(self, time_unit: TimeConversionTypes = "seconds") -> float:
        """
        Integrate the value of the KPI over time by summing up the product of values and the time difference.
        """
        if self.dt is None:
            self._get_dt()
        products = self.value * self.dt / TIME_CONVERSION[time_unit]
        integral = products.sum()
        return integral


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
    power_flex_full: KPISeries = pydantic.Field(
        default=KPISeries(
            name="power_flex_full",
            unit="kW"
        ),
        description="Power flexibility",
    )
    power_flex_offer: KPISeries = pydantic.Field(
        default=KPISeries(
            name="power_flex_offer",
            unit="kW"
        ),
        description="Power flexibility",
    )
    power_flex_offer_max: KPI = pydantic.Field(
        default=KPI(
            name="power_flex_offer_max",
            unit="kW"
        ),
        description="Maximum power flexibility",
    )
    power_flex_offer_min: KPI = pydantic.Field(
        default=KPI(
            name="power_flex_offer_min",
            unit="kW"
        ),
        description="Minimum power flexibility",
    )
    power_flex_offer_avg: KPI = pydantic.Field(
        default=KPI(
            name="power_flex_offer_avg",
            unit="kW"
        ),
        description="Average power flexibility",
    )
    energy_flex: KPI = pydantic.Field(
        default=KPI(
            name="energy_flex",
            unit="kWh"
        ),
        description="Energy flexibility equals the integral of the power flexibility",
    )

    # Costs KPIs
    costs_series: KPISeries = pydantic.Field(
        default=KPISeries(
            name="costs_series",
            unit="ct"
        ),
        description="Costs of flexibility",
    )
    costs: KPI = pydantic.Field(
        default=KPI(
            name="costs",
            unit="ct"
        ),
        description="Costs of flexibility",
    )
    corrected_costs: KPI = pydantic.Field(
        default=KPI(
            name="corrected_costs",
            unit="ct"
        ),
        description="Corrected costs of flexibility considering the stored thermal energy in the system",
    )
    costs_rel: KPI = pydantic.Field(
        default=KPI(
            name="costs_rel",
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
            power_profile_shadow: pd.Series,
            power_costs_profile: pd.Series,
            mpc_time_grid: np.ndarray,
            flex_offer_time_grid: np.ndarray,
            stored_energy_base: pd.Series,
            stored_energy_shadow: pd.Series
    ):
        """
        Calculate the KPIs based on the power and electricity input profiles.
        Time grids needed for indexing of the power flexibility profiles.
        """
        # Power / energy KPIs
        self._calculate_power_flex(power_profile_base=power_profile_base, power_profile_shadow=power_profile_shadow, flex_offer_time_grid=flex_offer_time_grid)
        self._calculate_power_flex_stats()
        self._calculate_energy_flex()

        # Costs KPIs
        end_power_diff = numpy.mean(power_profile_shadow.values[-4:-1] - power_profile_base.values[-4:-1])
        if abs(end_power_diff) > 0.01:
            correct_cost = True
        else:
            correct_cost = False
        stored_energy_diff = stored_energy_shadow.values[-1] - stored_energy_base.values[-1]
        self._calculate_costs(power_costs_profile=power_costs_profile, stored_energy_diff=stored_energy_diff, correct_cost=correct_cost)
        self._calculate_costs_rel()

    def _calculate_power_flex(self, power_profile_base: pd.Series, power_profile_shadow: pd.Series,
                              flex_offer_time_grid: np.ndarray,
                              relative_error_acceptance: float = 0.01) -> pd.Series:
        """
        Calculate the power flexibility based on the base and flexibility power profiles.

        Args:
            relative_error_acceptance: threshold for the relative error between the baseline and shadow mpc to set the power flexibility to zero
        """
        if not power_profile_shadow.index.equals(power_profile_base.index):
            raise ValueError(f"Indices of power profiles do not match.\n"
                             f"Baseline: {power_profile_base.index}\n"
                             f"Shadow: {power_profile_shadow.index}")

        # Calculate flexibility
        if self.direction == "positive":
            power_flex = power_profile_base - power_profile_shadow
        elif self.direction == "negative":
            power_flex = power_profile_shadow - power_profile_base
        else:
            raise ValueError(f"Direction of KPIs not properly defined: {self.direction}")

        # Set values to zero if the difference is small
        relative_difference = (power_flex / power_profile_base).abs()
        power_flex.loc[relative_difference < relative_error_acceptance] = 0

        # Set values
        self.power_flex_full.value = power_flex
        self.power_flex_offer.value = power_flex.loc[flex_offer_time_grid[0]:flex_offer_time_grid[-1]]
        return power_flex

    def _calculate_power_flex_stats(self) -> [float]:
        """
        Calculate the characteristic values of the power flexibility for the offer.
        """
        if self.power_flex_offer.value is None:
            raise ValueError("Power flexibility value is empty.")

        # Calculate characteristic values
        power_flex_offer_max = self.power_flex_offer.max()
        power_flex_offer_min = self.power_flex_offer.min()
        power_flex_offer_avg = self.power_flex_offer.avg()

        # Set values
        self.power_flex_offer_max.value = power_flex_offer_max
        self.power_flex_offer_min.value = power_flex_offer_min
        self.power_flex_offer_avg.value = power_flex_offer_avg
        return power_flex_offer_max, power_flex_offer_min, power_flex_offer_avg

    def _calculate_energy_flex(self) -> float:
        """
        Calculate the energy flexibility by integrating the power flexibility of the offer window.
        """
        if self.power_flex_offer.value is None:
            raise ValueError("Power flexibility value of the offer is empty.")

        # Calculate flexibility
        energy_flex = self.power_flex_offer.integrate(time_unit="hours")

        # Set value
        self.energy_flex.value = energy_flex
        return energy_flex

    def _calculate_costs(self, power_costs_profile: pd.Series, stored_energy_diff: float, correct_cost: bool) -> [float, pd.Series]:
        """
        Calculate the costs of the flexibility event based on the electricity costs profile and the power flexibility profile.
        """
        # Calculate series
        costs_series = power_costs_profile * self.power_flex_full.value
        self.costs_series.value = costs_series

        # Calculate scalar
        costs = abs(self.costs_series.integrate(time_unit="hours"))
        corrected_costs = costs
        if correct_cost:
            corrected_costs = costs - stored_energy_diff * power_costs_profile.values[-1]
        self.costs.value = costs
        self.corrected_costs.value = corrected_costs
        return costs, costs_series, corrected_costs

    def _calculate_costs_rel(self) -> float:
        """
        Calculate the relative costs of the flexibility event per energy flexibility.
        """
        if self.energy_flex == 0:
            costs_rel = 0
        else:
            costs_rel = self.costs.value / self.energy_flex.value

        # Set value
        self.costs_rel.value = costs_rel
        return costs_rel

    def get_kpi_dict(self, identifier: bool = False) -> dict[str, KPI]:
        """
        Get the KPIs as a dictionary with names or identifier as keys.
        
        Args:
            identifier: If True, the keys are the identifiers of the KPIs, otherwise the name of the kpi.
        """
        kpi_dict = {}
        for kpi in vars(self).values():
            if isinstance(kpi, KPI):
                if identifier:
                    kpi_dict[kpi.get_kpi_identifier()] = kpi
                else:
                    kpi_dict[kpi.name] = kpi
        return kpi_dict

    def get_name_dict(self) -> dict[str, str]:
        """
        Returns:
            Dictionary of the kpis with names as keys and the identifiers as values.
        """
        name_dict = {}
        for name, kpi in self.get_kpi_dict(identifier=False).items():
            name_dict[name] = kpi.get_kpi_identifier()
        return name_dict


class FlexibilityData(pydantic.BaseModel):
    """
    Class containing the data for the calculation of the flexibility.
    """
    # Time parameters
    mpc_time_grid: np.ndarray = pydantic.Field(
        default=None,
        description="Time grid of the mpcs",
    )
    flex_offer_time_grid: np.ndarray = pydantic.Field(
        default=None,
        description="Time grid of the flexibility offer",
    )
    switch_time: Optional[float] = pydantic.Field(
        default=None,
        description="Time of the switch between the preparation and the market time",
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
    stored_energy_profile_base: pd.Series = pydantic.Field(
        default=None,
        description="Base profile of the stored thermal energy w.r.t. 0K",
    )
    stored_energy_profile_neg: pd.Series = pydantic.Field(
        default=None,
        description="Profile of the stored thermal energy w.r.t. 0K for negative flexibility",
    )
    stored_energy_profile_pos: pd.Series = pydantic.Field(
        default=None,
        description="Profile of the stored thermal energy w.r.t. 0K for positive flexibility",
    )
    power_costs_profile: pd.Series = pydantic.Field(
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

    def __init__(self, prep_time: int, market_time: int, flex_event_duration: int,
                 time_step: int, prediction_horizon: int, **data):
        super().__init__(**data)
        self.switch_time = prep_time + market_time
        self.flex_offer_time_grid = np.arange(self.switch_time, self.switch_time + flex_event_duration, time_step)
        self.mpc_time_grid = np.arange(0, prediction_horizon * time_step, time_step)

    def format_predictor_inputs(self, series: pd.Series) -> pd.Series:
        """
        Format the input of the predictor to unify the data.

        Args:
            series: Input series from a predictor.
            
        Returns:
            Formatted series.
        """
        series.index = series.index - series.index[0]
        series = series.reindex(self.mpc_time_grid)
        if any(series.isna()):
            raise ValueError(f"The mpc time grid is not compatible with the predictor input, which leads to NaN values in the series.\n"
                             f"MPC time grid:{self.mpc_time_grid}\n"
                             f"Series index:{series.index}")
        return series

    def format_mpc_inputs(self, series: pd.Series) -> pd.Series:
        """
        Format the input of the mpc to unify the data.
        
        Args:
            series: Input series from a mpc.
            
        Returns:
            Formatted series.
        """
        series = strip_multi_index(series)
        if any(series.isna()):
            series = fill_nans(series=series, method=MEAN)
        series = series.reindex(self.mpc_time_grid)
        if any(series.isna()):
            raise ValueError(f"The mpc time grid is not compatible with the mpc input, which leads to NaN values in the series.\n"
                             f"MPC time grid:{self.mpc_time_grid}\n"
                             f"Series index:{series.index}")
        return series

    def calculate(self) -> [FlexibilityKPIs, FlexibilityKPIs]:
        """
        Calculate the KPIs for the positive and negative flexibility.

        Returns:
            positive KPIs, negative KPIs
        """
        self.kpis_pos.calculate(
            power_profile_base=self.power_profile_base,
            power_profile_shadow=self.power_profile_flex_pos,
            power_costs_profile=self.power_costs_profile,
            mpc_time_grid=self.mpc_time_grid,
            flex_offer_time_grid=self.flex_offer_time_grid,
            stored_energy_base=self.stored_energy_profile_base,
            stored_energy_shadow=self.stored_energy_profile_pos
        )
        self.kpis_neg.calculate(
            power_profile_base=self.power_profile_base,
            power_profile_shadow=self.power_profile_flex_neg,
            power_costs_profile=self.power_costs_profile,
            mpc_time_grid=self.mpc_time_grid,
            flex_offer_time_grid=self.flex_offer_time_grid,
            stored_energy_base=self.stored_energy_profile_base,
            stored_energy_shadow=self.stored_energy_profile_neg
        )
        return self.kpis_pos, self.kpis_neg

    def get_kpis(self) -> dict[str, KPI]:
        kpis_dict = self.kpis_pos.get_kpi_dict(identifier=True) | self.kpis_neg.get_kpi_dict(identifier=True)
        return kpis_dict
