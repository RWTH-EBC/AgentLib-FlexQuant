import casadi as ca
import pandas as pd
from agentlib_mpc.models.casadi_model import (
    CasadiModel,
    CasadiInput,
    CasadiState,
    CasadiParameter,
    CasadiOutput,
    CasadiModelConfig,
)
from math import inf


class BaselineMPCModelConfig(CasadiModelConfig):
    inputs: list[CasadiInput] = [
        CasadiInput(
            name="mDot",
            value=0.0225,
            unit="kg/s",
            description="Air mass flow into zone",
        ),
        CasadiInput(
            name="load", value=150, unit="W", description="Heat load into zone"
        ),
        CasadiInput(
            name="T_in", value=280.15, unit="K", description="Inflow air temperature"
        ),
        CasadiInput(
            name="T_upper",
            value=294.15,
            unit="K",
            description="Upper boundary (soft) for T.",
        ),
        CasadiInput(
            name="T_lower",
            value=292.15,
            unit="K",
            description="Upper boundary (soft) for T.",
        ),
        CasadiInput(
            name="_P_external",
            value=0,
            unit="W",
            type="pd.Series",
            description="External power profile to be provided",
        ),
        CasadiInput(
            name="in_provision",
            value=False,
            unit="-",
            type="bool",
            description="Flag signaling if the flexibility is in provision",
        ),
        CasadiInput(
            name="rel_start",
            value=0,
            unit="s",
            type="int",
            description="relative start time of the flexibility event",
        ),
        CasadiInput(
            name="rel_end",
            value=0,
            unit="s",
            type="int",
            description="relative end time of the flexibility event",
        ),
    ]
    states: list[CasadiState] = [
        CasadiState(
            name="T", value=293.15, unit="K", description="Temperature of zone"
        ),
        CasadiState(
            name="T_slack",
            value=0,
            unit="K",
            description="Slack variable of temperature of zone",
        ),
    ]
    parameters: list[CasadiParameter] = [
        CasadiParameter(
            name="cp",
            value=1000,
            unit="J/kg*K",
            description="thermal capacity of the air",
        ),
        CasadiParameter(
            name="C", value=100000, unit="J/K", description="thermal capacity of zone"
        ),
        CasadiParameter(
            name="s_T",
            value=1,
            unit="-",
            description="Weight for T in constraint function",
        ),
        CasadiParameter(
            name="r_mDot",
            value=1,
            unit="-",
            description="Weight for mDot in objective function",
        ),
        CasadiParameter(
            name="profile_deviation_weight",
            value=0,
            unit="-",
            description="Weight of soft constraint for deviation from accepted flexible profile",
        ),
        CasadiParameter(
            name="profile_comfort_weight",
            value=0,
            unit="-",
            description="Weight of soft constraint for discomfort",
        ),
    ]
    outputs: list[CasadiOutput] = [
        CasadiOutput(name="T_out", unit="K", description="Temperature of zone"),
        CasadiOutput(
            name="E_out", unit="kWh", description="Stored energy in the zone w.r.t. 0K"
        ),
        CasadiOutput(
            name="P_el", unit="W", description="The power input to the system"
        ),
    ]


class BaselineMPCModel(CasadiModel):
    config: BaselineMPCModelConfig

    def setup_system(self):
        self.T.ode = (
            self.cp * self.mDot / self.C * (self.T_in - self.T) + self.load / self.C
        )
        self.P_el.alg = self.cp * self.mDot * (self.T - self.T_in) / 1000
        self.T_out.alg = self.T
        self.E_out.alg = -self.T * self.C / (3600 * 1000)
        self.constraints = [
            (self.T_lower, self.T + self.T_slack, inf),
            (-inf, self.T - self.T_slack, self.T_upper),
            (0, self.T_slack, inf),
        ]
        objective = sum([self.r_mDot * self.mDot, self.s_T * self.T_slack**2])
        obj_std = objective
        return ca.if_else(
            self.in_provision.sym,
            ca.if_else(
                self.time < self.rel_start.sym,
                obj_std,
                ca.if_else(
                    self.time >= self.rel_end.sym,
                    obj_std,
                    sum(
                        [
                            self.profile_deviation_weight
                            * (self.P_el - self._P_external) ** 2,
                            self.T_slack**2 * self.profile_comfort_weight,
                        ]
                    ),
                ),
            ),
            obj_std,
        )


class PosFlexModelConfig(CasadiModelConfig):
    inputs: list[CasadiInput] = [
        CasadiInput(
            name="mDot",
            value=0.0225,
            unit="kg/s",
            description="Air mass flow into zone",
        ),
        CasadiInput(
            name="load", value=150, unit="W", description="Heat load into zone"
        ),
        CasadiInput(
            name="T_in", value=280.15, unit="K", description="Inflow air temperature"
        ),
        CasadiInput(
            name="T_upper",
            value=294.15,
            unit="K",
            description="Upper boundary (soft) for T.",
        ),
        CasadiInput(
            name="T_lower",
            value=292.15,
            unit="K",
            description="Upper boundary (soft) for T.",
        ),
        CasadiInput(
            name="mDot_full",
            value=None,
            unit="Not defined",
            type="pd.Series",
            description="full control trajectory output of baseline mpc",
        ),
        CasadiInput(
            name="in_provision",
            value=False,
            unit="-",
            type="bool",
            description="provision flag",
        ),
    ]
    states: list[CasadiState] = [
        CasadiState(
            name="T", value=293.15, unit="K", description="Temperature of zone"
        ),
        CasadiState(
            name="T_slack",
            value=0,
            unit="K",
            description="Slack variable of temperature of zone",
        ),
    ]
    parameters: list[CasadiParameter] = [
        CasadiParameter(
            name="cp",
            value=1000,
            unit="J/kg*K",
            description="thermal capacity of the air",
        ),
        CasadiParameter(
            name="C", value=100000, unit="J/K", description="thermal capacity of zone"
        ),
        CasadiParameter(
            name="s_T",
            value=1,
            unit="-",
            description="Weight for T in constraint function",
        ),
        CasadiParameter(
            name="r_mDot",
            value=1,
            unit="-",
            description="Weight for mDot in objective function",
        ),
        CasadiParameter(
            name="prep_time", value=0, unit="s", description="time to switch objective"
        ),
        CasadiParameter(
            name="flex_event_duration",
            value=0,
            unit="s",
            description="time to switch objective",
        ),
        CasadiParameter(
            name="market_time",
            value=0,
            unit="s",
            description="time to switch objective",
        ),
        CasadiParameter(
            name="s_P",
            value=10,
            unit="-",
            description="Weight for P in objective function",
        ),
    ]
    outputs: list[CasadiOutput] = [
        CasadiOutput(name="T_out", unit="K", description="Temperature of zone"),
        CasadiOutput(
            name="E_out", unit="kWh", description="Stored energy in the zone w.r.t. 0K"
        ),
        CasadiOutput(
            name="P_el", unit="W", description="The power input to the system"
        ),
    ]


class PosFlexModel(CasadiModel):
    config: PosFlexModelConfig

    def setup_system(self):
        mDot_lower = ca.if_else(
            self.time < self.market_time.sym, self.mDot_full.sym, self.mDot.lb
        )
        mDot_upper = ca.if_else(
            self.time < self.market_time.sym, self.mDot_full.sym, self.mDot.ub
        )
        self.T.ode = (
            self.cp * self.mDot / self.C * (self.T_in - self.T) + self.load / self.C
        )
        self.P_el.alg = self.cp * self.mDot * (self.T - self.T_in) / 1000
        self.T_out.alg = self.T
        self.E_out.alg = -self.T * self.C / (3600 * 1000)
        self.constraints = [
            (self.T_lower, self.T + self.T_slack, inf),
            (-inf, self.T - self.T_slack, self.T_upper),
            (0, self.T_slack, inf),
            (mDot_lower, self.mDot, mDot_upper),
        ]
        objective = sum([self.r_mDot * self.mDot, self.s_T * self.T_slack**2])
        obj_std = objective
        obj_flex = sum([self.s_T * self.T_slack**2, self.s_P * self.P_el])
        return ca.if_else(
            self.time < self.prep_time.sym + self.market_time.sym,
            obj_std,
            ca.if_else(
                self.time
                < self.prep_time.sym
                + self.flex_event_duration.sym
                + self.market_time.sym,
                obj_flex,
                obj_std,
            ),
        )


class NegFlexModelConfig(CasadiModelConfig):
    inputs: list[CasadiInput] = [
        CasadiInput(
            name="mDot",
            value=0.0225,
            unit="kg/s",
            description="Air mass flow into zone",
        ),
        CasadiInput(
            name="load", value=150, unit="W", description="Heat load into zone"
        ),
        CasadiInput(
            name="T_in", value=280.15, unit="K", description="Inflow air temperature"
        ),
        CasadiInput(
            name="T_upper",
            value=294.15,
            unit="K",
            description="Upper boundary (soft) for T.",
        ),
        CasadiInput(
            name="T_lower",
            value=292.15,
            unit="K",
            description="Upper boundary (soft) for T.",
        ),
        CasadiInput(
            name="mDot_full",
            value=None,
            unit="Not defined",
            type="pd.Series",
            description="full control trajectory output of baseline mpc",
        ),
        CasadiInput(
            name="in_provision",
            value=False,
            unit="-",
            type="bool",
            description="provision flag",
        ),
    ]
    states: list[CasadiState] = [
        CasadiState(
            name="T", value=293.15, unit="K", description="Temperature of zone"
        ),
        CasadiState(
            name="T_slack",
            value=0,
            unit="K",
            description="Slack variable of temperature of zone",
        ),
    ]
    parameters: list[CasadiParameter] = [
        CasadiParameter(
            name="cp",
            value=1000,
            unit="J/kg*K",
            description="thermal capacity of the air",
        ),
        CasadiParameter(
            name="C", value=100000, unit="J/K", description="thermal capacity of zone"
        ),
        CasadiParameter(
            name="s_T",
            value=1,
            unit="-",
            description="Weight for T in constraint function",
        ),
        CasadiParameter(
            name="r_mDot",
            value=1,
            unit="-",
            description="Weight for mDot in objective function",
        ),
        CasadiParameter(
            name="prep_time", value=0, unit="s", description="time to switch objective"
        ),
        CasadiParameter(
            name="flex_event_duration",
            value=0,
            unit="s",
            description="time to switch objective",
        ),
        CasadiParameter(
            name="market_time",
            value=0,
            unit="s",
            description="time to switch objective",
        ),
        CasadiParameter(
            name="s_P",
            value=10,
            unit="-",
            description="Weight for P in objective function",
        ),
    ]
    outputs: list[CasadiOutput] = [
        CasadiOutput(name="T_out", unit="K", description="Temperature of zone"),
        CasadiOutput(
            name="E_out", unit="kWh", description="Stored energy in the zone w.r.t. 0K"
        ),
        CasadiOutput(
            name="P_el", unit="W", description="The power input to the system"
        ),
    ]


class NegFlexModel(CasadiModel):
    config: NegFlexModelConfig

    def setup_system(self):
        mDot_lower = ca.if_else(
            self.time < self.market_time.sym, self.mDot_full.sym, self.mDot.lb
        )
        mDot_upper = ca.if_else(
            self.time < self.market_time.sym, self.mDot_full.sym, self.mDot.ub
        )
        self.T.ode = (
            self.cp * self.mDot / self.C * (self.T_in - self.T) + self.load / self.C
        )
        self.P_el.alg = self.cp * self.mDot * (self.T - self.T_in) / 1000
        self.T_out.alg = self.T
        self.E_out.alg = -self.T * self.C / (3600 * 1000)
        self.constraints = [
            (self.T_lower, self.T + self.T_slack, inf),
            (-inf, self.T - self.T_slack, self.T_upper),
            (0, self.T_slack, inf),
            (mDot_lower, self.mDot, mDot_upper),
        ]
        objective = sum([self.r_mDot * self.mDot, self.s_T * self.T_slack**2])
        obj_std = objective
        obj_flex = sum([self.s_T * self.T_slack**2, -self.s_P * self.P_el])
        return ca.if_else(
            self.time < self.prep_time.sym + self.market_time.sym,
            obj_std,
            ca.if_else(
                self.time
                < self.prep_time.sym
                + self.flex_event_duration.sym
                + self.market_time.sym,
                obj_flex,
                obj_std,
            ),
        )
