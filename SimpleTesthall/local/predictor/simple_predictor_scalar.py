import agentlib as al
import numpy as np
import pandas as pd
from agentlib.core import Agent
import pickle
import json
import csv
from datetime import datetime
from typing import List
from local.utils.disturbance_generator import DisturbanceGenerator

class PredictorModuleConfig(al.BaseModuleConfig):
    """Module that outputs a prediction of the heat load at a specified
    interval."""
    outputs: al.AgentVariables = [
        al.AgentVariable(
            name="T_flow_in", description="Input temperature of radiator"
        ),
        al.AgentVariable(
            name="T_amb_scalar", description="Ambient temperature"
        ),
        al.AgentVariable(
            name="Q_RadSol_scalar", description="Radiative solar heat"
        ),
        al.AgentVariable(
            name="T_preTemWin_scalar", description="Outdoor surface temperature of window"
        ),
        al.AgentVariable(
            name="T_preTemWall_scalar",  description="Outdoor surface temperature of wall"
        ),
        al.AgentVariable(
            name="T_preTemRoof_scalar",  description="Outdoor surface temperature of roof"
        ),
        al.AgentVariable(
            name="schedule_human_scalar", description="Internal gains caused by humans"
        ),
        al.AgentVariable(
            name="schedule_dev_scalar", description="Internal gains caused by devices"
        ),
        al.AgentVariable(
            name="schedule_light_scalar",  description="Internal gains caused by light"
        ),
        al.AgentVariable(
            name="T_upper_scalar", description="Upper temperature boundary"
        ),
        al.AgentVariable(
            name="T_lower_scalar", description="Lower temperature boundary"
        )
    ]

    parameters: al.AgentVariables = [
        al.AgentVariable(
            name="rt"
        ),
        al.AgentVariable(
            name="time_step", value=900, description="Sampling time for prediction."
        ),
        al.AgentVariable(
            name="prediction_horizon",
            value=8,
            description="Number of sampling points for prediction.",
        ),
        al.AgentVariable(
            name="sampling_time",
            value=10,
            description="Time between prediction updates.",
        ),
        al.AgentVariable(
            name="comfort_interval_morning_start",
            value=6.5,
            description="Hour of day when stricter comfort starts",
        ),
        al.AgentVariable(
            name="comfort_interval_morning_end",
            value=8,
            description="Hour of day when stricter comfort ends",
        ),
        al.AgentVariable(
            name="comfort_interval_evening_start",
            value=16.5,
            description="Hour of day when stricter comfort starts",
        ),
        al.AgentVariable(
            name="comfort_interval_evening_end",
            value=22,
            description="Hour of day when stricter comfort ends",
        ),
        al.AgentVariable(
            name="upper_comfort_high",
            value=300.15,
            description="High value in the comfort set point trajectory.",
        ),
        al.AgentVariable(
            name="upper_comfort_low",
            value=296.15,
            description="Low value in the comfort set point trajectory.",
        ),
        al.AgentVariable(
            name="lower_comfort_high",
            value=294.15,
            description="High value in the comfort set point trajectory.",
        ),
        al.AgentVariable(
            name="lower_comfort_low",
            value=290.15,
            description="Low value in the comfort set point trajectory.",
        ),
        al.AgentVariable(
            name="base_price",
            value=1,
            description="Base price for electricity signal.",
        ),
        al.AgentVariable(
            name="varying_price_signal",
            value=0,
            description="0, for const signal. 1, for varying signal",
        ),
        al.AgentVariable(
            name="path_data_disturbances",
            value=r"predictor\Disturbances_ASHRAE",
            description="Path to disturbance data file",
        ),
        al.AgentVariable(
            name="path_mapping_disturbances",
            value=r"predictor\mapping_disturbance_generation.json",
            description="Path to disturbance mapping file",
        ),
        al.AgentVariable(
            name="path_setup_disturbances",
            value=r"predictor\setup_disturbances.json",
            description="Path to set up disturbances file",
        )
    ]

    # shared_variable_fields: List[str] = ["outputs"]


class PredictorModule(al.BaseModule):
    """Module that outputs a prediction of the heat load at a specified
    interval."""

    config: PredictorModuleConfig

    def register_callbacks(self):
        pass

    def __init__(self, *, config: dict, agent: Agent):
        super().__init__(config=config, agent=agent)
        # disturbance data
        # new disturbance.csv considered that the TRY 2018 starts at a monday,
        #  This is important for the internal gain profile

        # generate disturbance.csv from FMU
        disturbance_gen = DisturbanceGenerator(setup=self.get('path_setup_disturbances').value)
        disturbance_path = disturbance_gen.create_disturbances()
        self.disturbances = self.read_csv(disturbance_path)  # disturbances: a dataframe

        self.disturbances.index = self.disturbances['SimTime']

        # mapping disturbances
        # mapping_disturbances is a dic including alias as key, and name from model as value
        with open(self.get("path_mapping_disturbances").value, 'r') as f:
            self.setup_disturbances = json.load(f)
            self.mapping_disturbances = self.setup_disturbances['disturbances']

    def process(self):
        """Sets a new prediction at each time step."""
        # self.env.process(self.send_comfort_trajectory())
        self.env.process(self.send_modelica_parse_data())

        while True:
            sample_time = self.get("sampling_time").value
            # delete following. only one value at a time point is needed from casadi simulator
            ts = self.get("time_step").value
            k = self.get("prediction_horizon").value
            now = self.env.now
            grid = np.arange(now, now + k * ts + 1, sample_time)
            # add output for T_flow_in, only for testing
            values = impuls_signal(grid)
            traj = pd.Series(values, index=list(grid))
            self.set("T_flow_in", traj.iloc[0])
            yield self.env.timeout(sample_time)


    def send_comfort_trajectory(self):
        """Sends the series for the comfort condition."""
        sample_time = self.get("sampling_time").value
        comfort_interval_morning_start = self.get("comfort_interval_morning_start").value
        comfort_interval_morning_end = self.get("comfort_interval_morning_end").value
        comfort_interval_evening_start = self.get("comfort_interval_evening_start").value
        comfort_interval_evening_end = self.get("comfort_interval_evening_end").value
        ts = self.get("time_step").value
        k = self.get("prediction_horizon").value

        def calc_comfort_boundary(t):
            """
            Calculates the comfort boundary based on a time value.
            Function especially build for TRY 2015 (should be updated, if TRY changes).

            Args:
                t: actual time
            """

            day_of_week = int((t / 3600 / 24) % 7)  # Starts at Thurs specific at 2015
            # day0=Thurs. day1=Fri. : 2 3 Boundarys on weekend
            if day_of_week == 2 or day_of_week == 3:
                return self.get("upper_comfort_low").value, self.get("lower_comfort_high").value
            else:
                if round(comfort_interval_morning_start / 24, 5) <= round((t / 3600 / 24) % 1, 5) < round(
                        comfort_interval_morning_end / 24, 5) or \
                        round(comfort_interval_evening_start / 24, 5) <= round((t / 3600 / 24) % 1, 5) < round(
                    comfort_interval_evening_end / 24, 5):
                    return self.get("upper_comfort_low").value, self.get("lower_comfort_high").value
                else:
                    return self.get("upper_comfort_high").value, self.get("lower_comfort_low").value

        while True:
            now = self.env.now

            # temperature prediction
            #todo: isnt it  np.arange(now, now + k * ts + 1, ts)?
            grid = np.arange(now, now + k * ts + 1, sample_time)
            values_upper = []
            values_lower = []
            for t in grid:
                val_upper, val_lower = calc_comfort_boundary(t)
                values_upper.append(val_upper)
                values_lower.append(val_lower)

            traj_upper = pd.Series(values_upper, index=list(grid))
            traj_lower = pd.Series(values_lower, index=list(grid))
            self.set("T_upper", traj_upper)
            self.set("T_lower", traj_lower)
            yield self.env.timeout(sample_time)

    def send_modelica_parse_data(self):
        """Sends the disturbance calculated by the modelica model"""
        ts = self.get("time_step").value
        k = self.get("prediction_horizon").value
        sample_time = self.get("sampling_time").value

        # add key Q_RadSol for mapping_mapping_disturbance
        self.mapping_disturbances = {k: v for k, v in self.mapping_disturbances.items() if not k.startswith('Q_RadSol_or_')}
        self.mapping_disturbances["Q_RadSol"] = "Q_RadSol"

        while True:
            now = self.env.now
            # print(now)
            j = self.disturbances.index.get_indexer([now], method='nearest').item()
            i = self.disturbances.index.get_indexer([now + k * ts], method='nearest').item() + 1
            for key in self.mapping_disturbances.keys():
                single_dist = self.disturbances[key].iloc[j:i]
                #if self.get("rt"):
                    #single_dist.index = single_dist.index + self.env.t_start
                self.set(key + "_scalar", single_dist.iloc[0])
                # self.set(key, single_dist.iloc[0])
            yield self.env.timeout(sample_time)

    # read pkl is for old disturbance mapping (Disturbances_ASHRAE_365_d)
    # def read_pickle(self, filename: str):
    #     with open(filename, 'rb') as f:
    #         return pickle.load(f)

    from datetime import datetime
    def read_csv(self, filename: str):
        """
        Read disturbance
        Args:
            filename: defines where the disturbance.csv is located
        Returns:
            df: DataFrame with all important disturbance data
        """
        column_names = []
        data = {}

        with open(filename, 'r') as file:
            lines = file.readlines()
        lines[0] = 'Time' + lines[0]

        # Extract column names from the first line
        column_names = lines[0].strip().split(',')

        for name in column_names:
            data[name] = []

        # Extract time zero from the first value in the CSV
        time_zero = datetime.strptime(lines[1].split(',')[0], '%Y-%m-%d %H:%M:%S')

        for line in lines[1:]:
            values = line.strip().split(',')
            timestamp = datetime.strptime(values[0], '%Y-%m-%d %H:%M:%S')
            time_difference = (timestamp - time_zero).total_seconds()

            for i, value in enumerate(values[1:]):
                data[column_names[i + 1]].append(float(value))

            # Convert time values to relative seconds
            data['Time'].append(time_difference)

        df = pd.DataFrame(data)
        df.set_index('Time', inplace=True)  # Set 'Time' column as index

        return df

def impuls_signal(current):
    """Returns the ambient temperature in K, given a time in seconds, modeled as a 50% wide impulse signal with constant amplitude."""
    value = np.zeros(shape=current.shape)
    width = 0.5
    impulse_interval = 86400
    impulse_value = 10
    for i in range(current.size):
        # Introduce a 50% wide impulse at intervals with constant amplitude
        if (current[i] % impulse_interval) <= (impulse_interval * width):
             value[i] = 318.15  # Constant amplitude during impulse
        else:
            value[i] = 318.15 + impulse_value  # Maintain ambient base temperature
    return value


