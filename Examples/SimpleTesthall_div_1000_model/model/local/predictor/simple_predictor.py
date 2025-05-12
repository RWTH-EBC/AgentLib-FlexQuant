import agentlib as al
import numpy as np
import pandas as pd
from agentlib.core import Agent
import json
import csv
from datetime import datetime
from typing import List

storeDataFile: bool = True


class PredictorModuleConfig(al.BaseModuleConfig):
    """Module that outputs a prediction of the heat load at a specified
    interval."""
    outputs: al.AgentVariables = [
        al.AgentVariable(
            name="T_amb_pred", description="Ambient temperature"
        ),
        al.AgentVariable(
            name="Q_RadSol_pred", description="Radiative solar heat"
        ),
        al.AgentVariable(
            name="T_preTemWin_pred", description="Outdoor surface temperature of window"
        ),
        al.AgentVariable(
            name="T_preTemWall_pred", description="Outdoor surface temperature of wall"
        ),
        al.AgentVariable(
            name="T_preTemRoof_pred", description="Outdoor surface temperature of roof"
        ),
        al.AgentVariable(
            name="m_flow_ahu_pred", description="Mass flow of ahu"
        ),
        al.AgentVariable(
            name="schedule_human_pred", description="Internal gains caused by humans"
        ),
        al.AgentVariable(
            name="schedule_dev_pred", description="Internal gains caused by devices"
        ),
        al.AgentVariable(
            name="schedule_light_pred", description="Internal gains caused by light"
        ),
        al.AgentVariable(
            name="T_upper_pred", description="Upper temperature boundary"
        ),
        al.AgentVariable(
            name="T_lower_pred", description="Lower temperature boundary"
        ),
        al.AgentVariable(
            name="r_pel", unit="ct/kWh", description="Weight for P_el in objective function"
        ),

        # variables to be sent to the simulator as scalars

        al.AgentVariable(
            name="T_amb_sim", description="Ambient temperature"
        ),
        al.AgentVariable(
            name="Q_RadSol_sim", description="Radiative solar heat"
        ),
        al.AgentVariable(
            name="T_preTemWin_sim", description="Outdoor surface temperature of window"
        ),
        al.AgentVariable(
            name="T_preTemWall_sim", description="Outdoor surface temperature of wall"
        ),
        al.AgentVariable(
            name="T_preTemRoof_sim", description="Outdoor surface temperature of roof"
        ),
        al.AgentVariable(
            name="schedule_human_sim", description="Internal gains caused by humans"
        ),
        al.AgentVariable(
            name="schedule_dev_sim", description="Internal gains caused by devices"
        ),
        al.AgentVariable(
            name="schedule_light_sim", description="Internal gains caused by light"
        ),
        al.AgentVariable(
            name="T_upper_sim", description="Upper temperature boundary"
        ),
        al.AgentVariable(
            name="T_lower_sim", description="Lower temperature boundary"
        ),
    ]

    parameters: al.AgentVariables = [
        al.AgentVariable(
            name="sampling_time", value=300, description="Time between prediction updates.",
        ),
        al.AgentVariable(
            name="time_step", value=1800, description="Sampling time for prediction."
        ),
        al.AgentVariable(
            name="prediction_horizon", value=48, description="Number of sampling points for prediction.",
        ),
        al.AgentVariable(
            name="varying_price_signal", value=2, description="const, for const signal. pandb, for base and peak signal, dyn, for varying signal",
        ),
        al.AgentVariable(
            name="const_price", value=27.25, description="Base price for electricity signal.",
        ),
        al.AgentVariable(
            name="power_price_base", value=24.5
        ),
        al.AgentVariable(
            name="power_price_peak", value=30
        ),
        al.AgentVariable(
            name="comfort_interval_start", value=7, description="Hour of day when stricter comfort starts",
        ),
        al.AgentVariable(
            name="comfort_interval_end", value=19, description="Hour of day when stricter comfort ends",
        ),
        al.AgentVariable(
            name="peakprice_interval_start", value=8
        ),
        al.AgentVariable(
            name="peakprice_interval_end", value=20
        ),
        al.AgentVariable(
            name="upper_comfort_high", value=298.15, description="High value in the comfort set point trajectory.",
        ),
        al.AgentVariable(
            name="upper_comfort_low", value=296.15, description="Low value in the comfort set point trajectory.",
        ),
        al.AgentVariable(
            name="lower_comfort_high", value=292.15, description="High value in the comfort set point trajectory.",
        ),
        al.AgentVariable(
            name="lower_comfort_low", value=290.15, description="Low value in the comfort set point trajectory.",
        ),
        al.AgentVariable(
            name="path_data_disturbances",
            value=r"model\local\predictor\Disturbances_ASHRAE_365_d",
            description="Path to disturbance data file",
        ),
        al.AgentVariable(
            name="path_mapping_disturbances",
            value=r"model\local\predictor\disturbance_mapping.json",
            description="Path to disturbance mapping file",
        ),
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
        # new disturbance.csv considered that the TRY 2010 does not start at a monday,
        # but at a friday. This is important for the internal gain profile
        # self.disturbances1 = self.read_csv('predictor/disturbance.csv')
        self.disturbances = self.read_csv('model/local/predictor/disturbance.csv')
        self.disturbances.index = self.disturbances['SimTime']
        # disturbance mapping
        with open(self.get("path_mapping_disturbances").value, 'r') as f:
            self.mapping_disturbances = json.load(f)

        self.df_Comfort = None
        self.df_Disturbance_sim = None
        self.df_Disturbance_pred = None
        self.df_Price = None

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
            p_values = power_price_func(grid, self.get("varying_price_signal").value, self.get("const_price").value, 
                                        self.get("power_price_base").value, self.get("power_price_peak").value, 
                                        self.get("peakprice_interval_start").value, self.get("peakprice_interval_end").value)
            p_traj = pd.Series(p_values, index=list(grid))
            self.set("r_pel", p_traj)

            if storeDataFile:
                new_df = pd.DataFrame({'r_pel': p_traj})
                new_df.index.type = "time"
                new_df['time_step'] = now
                new_df.set_index(['time_step', new_df.index], inplace=True)
                self.df_Price = pd.concat([self.df_Price, new_df])
                # set the indices once again as concat cant handle indices properly
                indices = pd.MultiIndex.from_tuples(self.df_Price.index, names=["time_step", "time"])
                self.df_Price.set_index(indices, inplace=True)
                self.df_Price.to_csv("results/prediction_price_neg.csv")

            yield self.env.timeout(sample_time)

    def send_comfort_trajectory(self):
        """Sends the series for the comfort condition."""
        sample_time = self.get("sampling_time").value
        comfort_interval_start = self.get("comfort_interval_start").value
        comfort_interval_end = self.get("comfort_interval_end").value
        ts = self.get("time_step").value
        k = self.get("prediction_horizon").value

        def calc_comfort_boundary(t):
            """
            Calculates the comfort boundary based on a time value.
            Function especially build for TRY 2010 (should be updated, if TRY changes).

            Args:
                t: actual time
            """

            day_of_week = int((t / 3600 / 24) % 7)  # Starts at Friday
            # day1=Saturday day2=Sunday: Other Boundarys on weekend
            if day_of_week == 1 or day_of_week == 2:
                return self.get("upper_comfort_high").value, self.get("lower_comfort_low").value
            else:
                if round(comfort_interval_start / 24, 5) <= round((t / 3600 / 24) % 1, 5) < round(comfort_interval_end / 24, 5):
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
            self.set("T_upper_pred", traj_upper)
            self.set("T_lower_pred", traj_lower)
            self.set("T_upper_sim", traj_upper.iloc[0])
            self.set("T_lower_sim", traj_lower.iloc[0])

            if storeDataFile:
                new_df = pd.DataFrame({'T_lower_pred': traj_lower, 'T_upper_pred': traj_upper})
                new_df.index.type = "time"
                new_df['time_step'] = now
                new_df.set_index(['time_step', new_df.index], inplace=True)
                self.df_Comfort = pd.concat([self.df_Comfort, new_df])
                # set the indices once again as concat cant handle indices properly
                indices = pd.MultiIndex.from_tuples(self.df_Comfort.index, names=["time_step", "time"])
                self.df_Comfort.set_index(indices, inplace=True)
                self.df_Comfort.to_csv("results/prediction_comfort_neg.csv")

            yield self.env.timeout(sample_time)

    def send_modelica_parse_data(self):
        """Sends the disturbance calculated by the modelica model"""
        ts = self.get("time_step").value
        k = self.get("prediction_horizon").value
        sample_time = self.get("sampling_time").value
        self.disturbances["Q_RadSol"] = (self.disturbances["Q_RadSol_or_1"] +
                                         self.disturbances["Q_RadSol_or_2"] +
                                         self.disturbances["Q_RadSol_or_3"] +
                                         self.disturbances["Q_RadSol_or_4"])
        self.mapping_disturbances["Q_RadSol"] = "Q_RadSol"  # TODO: tidy up
        del self.mapping_disturbances["Q_RadSol_or_1"]
        del self.mapping_disturbances["Q_RadSol_or_2"]
        del self.mapping_disturbances["Q_RadSol_or_3"]
        del self.mapping_disturbances["Q_RadSol_or_4"]

        while True:
            now = self.env.now
            # print(now)
            j = self.disturbances.index.get_indexer([now], method='nearest').item()
            i = self.disturbances.index.get_indexer([now + k * ts], method='nearest').item() + 1
            grid = np.arange(now, now + k * ts + 1, sample_time)
            results_sim = []
            results_pred = []
            names_sim = []
            names_pred = []
            for key in self.mapping_disturbances.keys():
                single_dist = self.disturbances[key].iloc[j:i]

                self.set(f"{key}_pred", single_dist)
                self.set(f"{key}_sim", single_dist.iloc[0])

                results_pred.append(single_dist)
                results_sim.append(single_dist.iloc[0])
                names_pred.append(f"{key}_pred")
                names_sim.append(f"{key}_sim")

            if storeDataFile:
                new_df_sim = pd.DataFrame(results_sim).T
                new_df_sim.columns = names_sim
                new_df_sim.index.type = "time"
                new_df_sim['time_step'] = now
                new_df_sim.set_index(['time_step', new_df_sim.index], inplace=True)
                self.df_Disturbance_sim = pd.concat([self.df_Disturbance_sim, new_df_sim])
                # set the indices once again as concat cant handle indices properly
                indices = pd.MultiIndex.from_tuples(self.df_Disturbance_sim.index, names=["time_step", "time"])
                self.df_Disturbance_sim.set_index(indices, inplace=True)
                self.df_Disturbance_sim.to_csv("results/prediction_disturb_sim_neg.csv")

                new_df_pred = pd.DataFrame(results_pred).T
                new_df_pred.columns = names_pred
                new_df_pred.index.type = "time"
                new_df_pred['time_step'] = now
                new_df_pred.set_index(['time_step', new_df_pred.index], inplace=True)
                self.df_Disturbance_pred = pd.concat([self.df_Disturbance_pred, new_df_pred])
                # set the indices once again as concat cant handle indices properly
                indices = pd.MultiIndex.from_tuples(self.df_Disturbance_pred.index, names=["time_step", "time"])
                self.df_Disturbance_pred.set_index(indices, inplace=True)
                self.df_Disturbance_pred.to_csv("results/prediction_disturb_pred_neg.csv")

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
        # only for ASHRAE disturbance file
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


def power_price_func(current, varying_price_signal, const_price=None, power_price_base=None, power_price_peak=None,  peakprice_interval_start=None, peakprice_interval_end=None, uncertainty=0):
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
    file_path = 'model/local/predictor/power_price_2010.csv'
    random_factor = 1 + uncertainty * (np.random.random() - 0.5)
    def peak_and_base():
        day_of_week = int((t / 3600 / 24) % 7)  # Starts at Friday
        # day1=Saturday day2=Sunday: Other Boundarys on weekend
        if day_of_week == 1 or day_of_week == 2:
            return power_price_base
        else:
            if round(peakprice_interval_start / 24, 5) <= round((t / 3600 / 24) % 1, 5) < round(peakprice_interval_end / 24, 5):
                return power_price_peak
            else:
                return power_price_base

    if varying_price_signal == 0:
        value = const_price
        if value is None:
            raise ValueError(f"Not enough arguments for varying_price_signal = {varying_price_signal}")
        return value
    elif varying_price_signal == 1:
        price=[]
        for t in current:
            power_price = peak_and_base()
            price.append(power_price)
            if price[-1] is None:
                raise ValueError(f"Not enough arguments for varying_price_signal = {varying_price_signal}")
        return price
    elif varying_price_signal == 2:
        with open(file_path) as file:
            csv_reader = csv.reader(file, delimiter=';')
            next(csv_reader)
            data = list(csv_reader)
            data = interpolate_missing_values(data)

            power_prices = []
            for row in data:
                power_price = float(row[1])*0.1*1.1889 + 21.914
                power_prices.append(power_price)
        value = []
        for i, time in enumerate(current):
            index = int(time // 3600)
            if index < len(power_prices):
                power_price = power_prices[index]*random_factor
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
            for j in range(i - 1, -1, -1):
                if j in non_empty_rows:
                    previous = float(data[j][1])
                    break
            for j in range(i + 1, len(data)):
                if j in non_empty_rows:
                    next_value = float(data[j][1])
                    break
            if previous is not None and next_value is not None:
                interpolated_value = (previous + next_value) / 2
                data[i][1] = str(interpolated_value)

    return data
