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
        # Power var needs to be declared as an output. Here power var is also a model-input 
        # So two separate var names are used for power var: P_in as input, P_el as output
        # P_el set to P_in below
        CasadiInput(
            name="P_in",
            value=100,
            unit="W",
            description="Electrical power of heating rod (equivalent to P_el)",
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
            name="T_slack",
            value=0,
            unit="K",
            description="Slack variable for zone temperature",
        ),
    ]

    parameters: List[CasadiParameter] = [
        CasadiParameter(
            name="C",
            value=10000,
            unit="J/K",
            description="Thermal capacity of zone",
        ),
        CasadiParameter(
            name="U",
            value=5,
            unit="W/K",
            description="Thermal conductivity of zone",
        ),
        CasadiParameter(
            name="s_T",
            value=1,
            unit="-",
            description="Weight for zone temperature slack var in constraint function",
        ),
        CasadiParameter(
            name="r_pel",
            value=1,
            unit="-",
            description="Weight for P_el in objective function",
        )
    ]

    outputs: List[CasadiOutput] = [
        CasadiOutput(
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
            (-inf, self.T_zone - self.T_slack, self.T_upper),
            (self.T_lower, self.T_zone + self.T_slack, inf),
            (0, self.T_slack, inf),
            # hard constraints
            (0, self.P_in, 200),
        ]

        # Objective function
        objective = sum(
            [
                self.s_T * self.T_slack**2,
                self.r_pel * self.P_in,
            ]
        )

        return objective