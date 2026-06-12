"""
User module for sending comfort boundaries to MPCs.

Handles baseline MPC comfort bounds and shadow MPC bounds during flexibility events.
"""

from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field
import agentlib as al
from agentlib.core import BaseModule, BaseModuleConfig, Agent

from flexmod.predictor.utils.temperature_schedule import (
    generate_comfort_bounds_for_horizon,
    apply_flexibility_shift,
    get_active_mode_from_schedule,
    TEMPERATURE_BOUNDS,
    SCHEDULES,
    build_comfort_map,
)


class SimulationConfig(BaseModel):
    """Simulation timing parameters."""
    year: int = 2015
    time_step: int = 900  # Module execution interval in seconds
    sample_time: int = 900  # Trajectory sample time in seconds
    prediction_horizon: int = 96


class ComfortConfig(BaseModel):
    """Schedule selection and configuration."""
    schedule_type: Literal[
        "wolisz", "constant_setpoint", "constant_bound",
        "dynamic_range", "gradual_wakeup"] = "constant_setpoint"
    shift_upper_bound: float = 0.0
    shift_lower_bound: float = 0.0
    setpoint_preference: float = 21.0  # User's preferred temperature in Celsius


class PMVPPDConfig(BaseModel):
    """Comfort constraint configuration."""
    use_ppd_constraints: bool = False
    ppd_threshold: float = 10.0


class FlexibilityConfig(BaseModel):
    """Configuration for a flexibility event."""
    flexibility_event_duration: float = 9000
    prep_time: float = 0
    market_time: float = 0


class UserConfig(BaseModuleConfig):
    """User module configuration."""

    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    comfort: ComfortConfig = Field(default_factory=ComfortConfig)
    pmv_ppd: PMVPPDConfig = Field(default_factory=PMVPPDConfig)
    flexibility: FlexibilityConfig = Field(default_factory=FlexibilityConfig)

    outputs: al.AgentVariables = [
        al.AgentVariable(name="T_Air_lb", unit="K",
                         description="Lower bound of temperature constraint"),
        al.AgentVariable(name="T_Air_ub", unit="K",
                         description="Upper bound of temperature constraint"),
        al.AgentVariable(name="T_Air_lb_sim", unit="K",
                         description="Lower bound of temperature constraint"),
        al.AgentVariable(name="T_Air_ub_sim", unit="K",
                         description="Upper bound of temperature constraint"),
        al.AgentVariable(name="T_Air_ub_shadow", unit="K",
                         description="Upper comfort bound for shadow MPCs"),
        al.AgentVariable(name="T_Air_lb_shadow", unit="K",
                         description="Lower comfort bound for shadow MPCs"),
    ]


class UserModule(BaseModule):
    """User module that sends comfort boundaries to MPCs."""

    config: UserConfig

    def __init__(self, *, config: dict, agent: Agent):
        super().__init__(config=config, agent=agent)
        self._initialize_schedule_template()
        self._initialize_comfort_map()
        self._initialize_ppd_bounds()

    def _initialize_schedule_template(self):
        """Load the schedule template for the selected schedule type."""
        self.schedule_template = SCHEDULES.get(
            self.config.comfort.schedule_type,
            SCHEDULES["constant_setpoint"]
        )

    def _initialize_comfort_map(self):
        """Build comfort map with user's setpoint preference for constant_setpoint."""
        self.comfort_map = build_comfort_map(
            setpoint_preference=self.config.comfort.setpoint_preference
        )

    def _initialize_ppd_bounds(self):
        """Calculate PPD-based bounds for each comfort mode if enabled."""
        self.ppd_bounds = None

        if self.config.pmv_ppd.use_ppd_constraints:
            from utils.pmv_ppd_comfort import calculate_ppd_bounds_for_modes

            self.ppd_bounds = calculate_ppd_bounds_for_modes(
                ppd_threshold=self.config.pmv_ppd.ppd_threshold,
                modes=["sleeping", "active", "inactive", "not_present"],
                fallback_bounds=self.comfort_map["not_present"],
            )

    def register_callbacks(self):
        pass

    def _get_comfort_bounds_for_horizon(self, current_time: float) \
            -> tuple[pd.Series, pd.Series]:
        """
        Generate comfort bounds for the current prediction horizon.

        Automatically handles week rollovers and gradual transitions.
        Values are computed at each sample_time point, with gradual transitions
        properly interpolated.

        Args:
            current_time: Current simulation time in seconds since start of year.

        Returns:
            Tuple of (upper_series, lower_series) covering the prediction horizon.
        """
        return generate_comfort_bounds_for_horizon(
            schedule_template=self.schedule_template,
            comfort_map=self.comfort_map,
            current_time=current_time,
            sample_time=self.config.simulation.sample_time,
            prediction_horizon=self.config.simulation.prediction_horizon,
            reference_year=self.config.simulation.year,
        )

    def get_active_mode_at_time(self, timestamp: float) -> str | None:
        """Determine which comfort mode is active at a specific time."""
        return get_active_mode_from_schedule(
            self.schedule_template,
            timestamp,
            self.config.simulation.year
        )

    def process(self):
        """Main loop: periodically send comfort bounds."""
        time_step = self.config.simulation.time_step

        while True:
            current_time = self.env.time

            # Generate comfort bounds dynamically for current horizon
            t_upper, t_lower = self._get_comfort_bounds_for_horizon(current_time)

            self._send_baseline_bounds(t_upper, t_lower)
            self._send_shadow_bounds(t_upper, t_lower, current_time)

            yield self.env.timeout(time_step)

    def _send_baseline_bounds(self, t_upper: pd.Series, t_lower: pd.Series):
        """Send comfort bounds for the baseline MPC."""
        self.set("T_Air_ub", t_upper)
        self.set("T_Air_ub_sim", t_upper.iloc[0])
        self.set("T_Air_lb", t_lower)
        self.set("T_Air_lb_sim", t_upper.iloc[0])

    def _send_shadow_bounds(
        self,
        t_upper: pd.Series,
        t_lower: pd.Series,
        current_time: float
    ):
        """Send bounds for shadow MPCs during flexibility events."""
        flex = self.config.flexibility
        comfort = self.config.comfort

        # Calculate flex event timing relative to current time
        time_to_start = flex.prep_time + flex.market_time
        flex_start_time = current_time + time_to_start
        flex_end_time = current_time + time_to_start + flex.flexibility_event_duration

        # Apply flexibility shifts
        t_upper_shadow, t_lower_shadow = apply_flexibility_shift(
            t_upper=t_upper,
            t_lower=t_lower,
            flex_start=flex_start_time,
            flex_end=flex_end_time,
            shift_upper=comfort.shift_upper_bound,
            shift_lower=comfort.shift_lower_bound,
            sample_time=self.config.simulation.sample_time,
        )

        # Apply PPD constraints if enabled
        if self.config.pmv_ppd.use_ppd_constraints and self.ppd_bounds:
            self._apply_ppd_constraints(
                t_upper_shadow, t_lower_shadow,
                flex_start_time, flex_end_time
            )

        self.set("T_Air_ub_shadow", t_upper_shadow)
        self.set("T_Air_lb_shadow", t_lower_shadow)

    def _apply_ppd_constraints(
        self,
        t_upper: pd.Series,
        t_lower: pd.Series,
        flex_start: float,
        flex_end: float
    ):
        """Constrain temperatures by PPD-based bounds for each mode."""
        mask = (t_upper.index >= flex_start) & (t_upper.index <= flex_end)

        for idx in t_upper.index[mask]:
            mode = self.get_active_mode_at_time(idx)
            if mode and mode != "not_present" and mode in self.ppd_bounds:
                t_upper[idx] = min(t_upper[idx], self.ppd_bounds[mode]["T_upper"])
                t_lower[idx] = max(t_lower[idx], self.ppd_bounds[mode]["T_lower"])
