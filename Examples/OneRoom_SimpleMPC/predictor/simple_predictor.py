import agentlib as al
import numpy as np
import pandas as pd
from agentlib.core import Agent
from typing import List
import json
import csv
from datetime import datetime

class PredictorModuleConfig(al.BaseModuleConfig):
    """Module that outputs a prediction of the heat load at a specified
    interval."""
    outputs: al.AgentVariables = [
        al.AgentVariable(
            name="r_pel", unit="ct/kWh", type="pd.Series", description="Weight for P_el in objective function"
        ),
        # al.AgentVariable(
        #     name="mDot", unit="kg/s", type="float", description="Air mass flow into zone"
        # )
    ]
    parameters: al.AgentVariables = [
        al.AgentVariable(
            name="time_step", value=900, description="Sampling time for prediction."
        ),
        al.AgentVariable(
            name="prediction_horizon",
            value=8,
            description="Number of sampling points for prediction.",
        )
    ]


    shared_variable_fields:List[str] = ["outputs"]


class PredictorModule(al.BaseModule):
    """Module that outputs a prediction of the heat load at a specified
    interval."""

    config: PredictorModuleConfig

    def register_callbacks(self):
        pass

    def process(self):
        while True:
            sample_time = self.env.config.t_sample
            ts = self.get("time_step").value
            k = self.get("prediction_horizon").value
            now = self.env.now

            grid = np.arange(now, now + k * ts + 1, sample_time)
            p_traj = pd.Series([1 for i in grid], index=list(grid))

            # if now < 14000:
            #     self.set("mDot", 0.05)
            # elif now < 14900:
            #     self.set("mDot", 0)
            # else:
            #     self.set("mDot", 0.05)

            self.set("r_pel", p_traj)

            yield self.env.timeout(sample_time)
