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
            name="T_amb", description="Ambient temperature"
        ),
        al.AgentVariable(
            name="Q_RadSol", description="Radiative solar heat"
        ),
        al.AgentVariable(
            name="T_preTemWin", description="Outdoor surface temperature of window"
        ),
        al.AgentVariable(
            name="T_preTemWall", description="Outdoor surface temperature of wall"
        ),
        al.AgentVariable(
            name="T_preTemRoof", description="Outdoor surface temperature of roof"
        ),
        al.AgentVariable(
            name="schedule_human", description="Internal gains caused by humans"
        ),
        al.AgentVariable(
            name="schedule_dev", description="Internal gains caused by devices"
        ),
        al.AgentVariable(
            name="schedule_light", description="Internal gains caused by light"
        ),
        al.AgentVariable(
            name="T_upper", description="Upper temperature boundary"
        ),
        al.AgentVariable(
            name="T_lower", description="Lower temperature boundary"
        ),
        al.AgentVariable(
            name="r_pel", unit="ct/kWh", description="Weight for P_el in objective function"
        ),
    ]

    parameters: al.AgentVariables = [
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
            name="peakprice_interval_start",
            value=8
        ),
        al.AgentVariable(
            name="peakprice_interval_end",
            value=20
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
            name="const_price",
            value=22.27,
            description="Base price for electricity signal.",
        ),
        al.AgentVariable(
            name="power_price_base",
            value=23.0,
            description="Base price for electricity signal"
        ),
        al.AgentVariable(
            name="power_price_peak",
            value=28.1,
        ),
        al.AgentVariable(
            name="varying_price_signal",
            value=0,
            description="0, for const signal. 1, for varying signal",
        ),
        al.AgentVariable(
            name="path_data_disturbances",
            value=r"predictor\Disturbances",
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

    shared_variable_fields: List[str] = ["outputs"]


class PredictorModule(al.BaseModule):
    """Module that outputs a prediction of the heat load at a specified
    interval."""

    config: PredictorModuleConfig

    def register_callbacks(self):
        pass

    def __init__(self, *, config: dict, agent: Agent):
        super().__init__(config=config, agent=agent)
        # disturbance data
        # new disturbance.csv considered that the TRY 2015 starts at a monday,
        #  This is important for the internal gain profile

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
        self.env.process(self.send_comfort_trajectory())
        self.env.process(self.send_modelica_parse_data())

        while True:
            sample_time = self.get("sampling_time").value
            ts = self.get("time_step").value
            k = self.get("prediction_horizon").value
            now = self.env.now
            grid = np.arange(now, now + k * ts + 1, sample_time)
            # power price
            p_values = power_price_func(self, grid, uncertainty=0)
            p_traj = pd.Series(p_values, index=list(grid))
            self.set("r_pel", p_traj)

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
            Function especially build for year 2015 (should be updated, if TRY changes).

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
        self.mapping_disturbances = {k: v for k, v in self.mapping_disturbances.items() if
                                     not k.startswith('Q_RadSol_or_')}
        self.mapping_disturbances["Q_RadSol"] = "Q_RadSol"

        while True:
            now = self.env.now
            # print(now)
            j = self.disturbances.index.get_indexer([now], method='nearest').item()
            i = self.disturbances.index.get_indexer([now + k * ts], method='nearest').item() + 1
            for key in self.mapping_disturbances.keys():
                single_dist = self.disturbances[key].iloc[j:i]
                # if key == "T_amb":
                #     print(1)
                self.set(key, single_dist)
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
            df: DataFrame with all important disturbance data, index is time in seconds.
        """
        column_names = []
        data = {}

        with open(filename, 'r') as file:
            # Reads all lines from the file into a list called lines,
            # where each element is a line from the file.
            lines = file.readlines()
        # only for ASHRAE disturbance file, lines[0] includes all disturbances' name, add label to the first column.
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


def power_price_func(self, current, uncertainty):
    """
    Generates information for power price signal.
    If varying_price_signal is 0 (defined in config.json), the power price is assigned to a constant const_price.
    If varying_price_signal is 1, the electricity price data is in base and peak mode.
    If varying_price_signal is 2, the electricity price data is read from a csv file and assigned to the simulation time.

    Args:
        current: index grid that represents the simulation time
        uncertainty: set to zero, if there is no uncertainty considered
    Returns:
        value: series with power price assigned to simulation time
    """
    file_path = 'predictor/price_ele_2015_day_ahead.csv'
    random_factor = 1 + uncertainty * (np.random.random() - 0.5)
    varying_price_signal = self.get("varying_price_signal").value
    const_price = self.get("const_price").value
    peakprice_interval_start = self.get("peakprice_interval_start").value
    peakprice_interval_end = self.get("peakprice_interval_end").value

    def peak_and_base():
        day_of_week = int((t / 3600 / 24) % 7)  # 2015 Starts at Thursday
        # day0=Thursday day1=Frei: 2,3 on weekend
        if day_of_week == 2 or day_of_week == 3:
            return self.get("power_price_base").value
        else:
            if round(peakprice_interval_start / 24, 5) <= round((t / 3600 / 24) % 1, 5) < round(
                    peakprice_interval_end / 24, 5):
                return self.get("power_price_peak").value
            else:
                return self.get("power_price_base").value

    if varying_price_signal == 0:
        value = const_price
        return value
    elif varying_price_signal == 1:
        price = []
        for t in current:
            power_price = peak_and_base()
            price.append(power_price)
        return price
    elif varying_price_signal == 2:
        with open(file_path) as file:
            csv_reader = csv.reader(file, delimiter=',')
            next(csv_reader)
            data = list(csv_reader)
            data = interpolate_missing_values(data)

            power_prices = []
            for row in data:
                #power_price = float(row[1])*0.1*1.1889 + 21.914
                #why?
                if float(row[1]) < 0:
                    row[1] = 0
                power_price = float(row[1]) * 0.1 * 1.1889 + 21.914  # *0.1 EUR/MWh to ct/KWh
                power_prices.append(power_price)
        value = []
        for i, time in enumerate(current):
            index = int(time // 3600)
            if index < len(power_prices):
                power_price = power_prices[index] * random_factor
                value.append(power_price)
            else:
                value.append(None)
        return value
    else:
        raise ValueError("Invalid price signal mode")


def interpolate_missing_values(data):
    """
    Used to interpolate values, if there are missing positions in DataFrame
    Returns:
        data: Complete DataFrame without missing values
    """
    non_empty_rows = []
    for i, row in enumerate(data):
        if row[1] != '':
            non_empty_rows.append(i)

    for i, row in enumerate(data):
        if row[1] == '':
            previous = None
            next_value = None
            # Search for the previous non-missing value
            for j in range(i - 1, -1, -1):
                if j in non_empty_rows:
                    previous = float(data[j][1])
                    break
            # Search for the next non-missing value
            for j in range(i + 1, len(data)):
                if j in non_empty_rows:
                    next_value = float(data[j][1])
                    break
            # If no previous value exists but there is a next value, fill with the next value
            if previous is None and next_value is not None:
                data[i][1] = str(next_value)
            if previous is not None and next_value is not None:
                interpolated_value = (previous + next_value) / 2
                data[i][1] = str(interpolated_value)

    return data
