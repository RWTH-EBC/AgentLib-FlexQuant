from agentlib_mpc.models.casadi_model import (
    CasadiModel,
    CasadiInput,
    CasadiState,
    CasadiParameter,
    CasadiOutput,
    CasadiModelConfig,
)
from typing import List
from math import inf


class BaselineMPCModelConfig(CasadiModelConfig):
    inputs: List[CasadiInput] = [
        # controls
        CasadiInput(
            name="mDot", value=0.0225, unit="kg/s", description="Air mass flow into zone"
        ),
        # disturbances
        CasadiInput(
            name="load", value=150, unit="W", description="Heat " "load into zone"
        ),
        CasadiInput(
            name="T_in", value=280.15, unit="K", description="Inflow air temperature"
        ),
        # settings
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

    ]

    states: List[CasadiState] = [
        CasadiState(name="t_sim", value=0, unit="sec", description="simulation time"),

        # differential
        CasadiState(
            name="T", value=293.15, unit="K", description="Temperature of zone"
        ),
        # algebraic
        # slack variables
        CasadiState(
            name="T_slack",
            value=0,
            unit="K",
            description="Slack variable of temperature of zone",
        ),
        
    ]

    parameters: List[CasadiParameter] = [
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
            value=100,
            unit="-",
            description="Weight of soft constraint for deviation from accepted flexible profile",
        ),

    ]
    outputs: List[CasadiOutput] = [
        CasadiOutput(name="T_out", unit="K", description="Temperature of zone"),
        CasadiOutput(
            name="P_el",
            unit="W",
            description="The power input to the system",
        ),
    ]

class BaselineMPCModel(CasadiModel):
    config: BaselineMPCModelConfig
                
    def setup_system(self):
        # Define ode
        self.T.ode = (
            self.cp * self.mDot / self.C * (self.T_in - self.T) + self.load / self.C
        )
        self.P_el.alg = self.cp * self.mDot * (self.T - self.T_in)/1000

        # Define ae
        self.T_out.alg = self.T  # math operation to get the symbolic variable
        # Constraints: List[(lower bound, function, upper bound)]
        self.constraints = [
            # soft constraints
            (self.T_lower, self.T + self.T_slack, inf),
            (-inf, self.T - self.T_slack, self.T_upper),
            (0, self.T_slack, inf)
        ]
        # Objective function
        objective = sum(
                [
                    self.r_mDot * self.mDot,
                    self.s_T * self.T_slack**2,
                ]
            )
        return objective


