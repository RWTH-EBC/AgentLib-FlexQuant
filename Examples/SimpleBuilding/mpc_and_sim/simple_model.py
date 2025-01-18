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
            #TODO: Change to another name "P_in"
            name="P_in",
            value=100,
            unit="W",
            description="Electrical power of heating rod",
        ),
        # disturbances
        CasadiInput(
            name="T_amb",
            value=290.15,
            unit="K",
            description="Ambient air temperature",
        ),
        # settings
        CasadiInput(
            name="T_upper",
            value=294.15,
            unit="K",
            description="Upper boundary (soft) for T",
        ),
        CasadiInput(
            name="T_lower",
            value=290.15,
            unit="K",
            description="Lower boundary (soft) for T",
        ),
    ]

    states: List[CasadiState] = [
        # differential
        CasadiState(
            name="T_zone",
            value=293.15,
            unit="K",
            description="Temperature of zone",
        ),
        # algebraic
        # slack variables
        CasadiState(
            name="T_slack_upper",
            value=0,
            unit="K",
            description="Slack variable of upper temperature of zone",
        ),
        CasadiState(
            name="T_slack_lower",
            value=0,
            unit="K",
            description="Slack variable of lower temperature of zone",
        ),
    ]

    parameters: List[CasadiParameter] = [
        CasadiParameter(
            name="C",
            value=10000,
            unit="J/K",
            description="thermal capacity of zone",
        ),
        CasadiParameter(
            name="U",
            value=5,
            unit="W/K",
            description="thermal conductivity of zone",
        ),
        CasadiParameter(
            name="s_T_upper",
            value=1,
            unit="-",
            description="Weight for T in upper constraint function",
        ),
        CasadiParameter(
            name="s_T_lower",
            value=1,
            unit="-",
            description="Weight for T in lower constraint function",
        ),
        CasadiParameter(
            name="r_pel",
            value=1,
            unit="-",
            description="Weight for P_el in objective function",
        )
    ]

    outputs: List[CasadiParameter] = [
        CasadiParameter(
            #TODO: alias: "P_in"?
            name="P_el",
            unit="W",
            description="Electrical power of heating rod (system input)",
        )
    ]

class BaselineMPCModel(CasadiModel):

    config: BaselineMPCModelConfig
                
    def setup_system(self):
        # Define ode
        self.T_zone.ode = (self.P_in - self.U * (self.T_zone - self.T_amb)) / self.C

        #Define ae for outputs
        self.P_el.alg = self.P_in

        # Constraints: List[(lower bound, function, upper bound)]
        self.constraints = [
            # soft constraints
            (self.T_upper, self.T_zone + self.T_slack_upper, self.T_upper),
            (self.T_lower, self.T_zone - self.T_slack_lower, self.T_lower),
            (0, self.T_slack_upper, inf),
            (0, self.T_slack_lower, inf),
            # hard constraints
            (-100, self.P_in, 200),
        ]

        # Objective function
        objective = sum(
            [
                self.s_T_upper * self.T_slack_upper**2,
                self.s_T_lower * self.T_slack_lower**2,
                self.r_pel * self.P_in,
            ]
        )

        return objective