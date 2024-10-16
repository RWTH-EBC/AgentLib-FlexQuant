import agentlib as al
from typing import Optional, Tuple, List
import numpy as np
import pandas as pd
import os.path
from math import inf


class FlexibilityModuleConfig(al.BaseModuleConfig):
    outputs: al.AgentVariables = [
        al.AgentVariable(
            name="powerflex_flex_neg", unit='W', description="Negative Powerflexibility"
        ),
        al.AgentVariable(
            name="powerflex_flex_pos", unit='W', description="Positive Powerflexibility"
        ),
        al.AgentVariable(
            name="powerflex_avg_neg", unit='kW', description="Negative Average Powerflexibility"
        ),
        al.AgentVariable(
            name="powerflex_avg_pos", unit='kW', description="Positive Average Powerflexibility"
        ),
        al.AgentVariable(
            name="energyflex_neg", unit='kWh', description="Negative Energyflexibility"
        ),
        al.AgentVariable(
            name="energyflex_pos", unit='kWh', description="Positive Energyflexibility"
        ),
        al.AgentVariable(
            name="costs_neg", unit='ct', description="Saved costs due to baseline"
        ),
        al.AgentVariable(
            name="costs_pos", unit='ct', description="Saved costs due to baseline"
        ),
        al.AgentVariable(
            name="costs_neg_rel", unit='ct/kWh', description="Saved costs due to baseline"
        ),
        al.AgentVariable(
            name="costs_pos_rel", unit='ct/kWh', description="Saved costs due to baseline"
        ),
    ]

    parameters: al.AgentVariables = [
        al.AgentVariable(
            name="time_step", value=900, description="Sampling time for prediction."
        ),
        al.AgentVariable(
            name="sampling_time", value=10, description="Time between prediction updates."
        ),
        al.AgentVariable(
            name="prediction_horizon",
            value=9,
            description="Number of sampling points for prediction.",
        ),
        al.AgentVariable(
            name="flexibility_event",
            value=7200,
            description="Duration of flexibility event",
        ),
        al.AgentVariable(
            name="preperation_horizon",
            value=1800,
            description="Duration of preperation phase before flexibility event",
        ),
    ]

    inputs: al.AgentVariables = [
        al.AgentVariable(
            name="P_el_alg",
            description="Baseline"
        ),
        al.AgentVariable(
            name="P_el_max_alg",
            value=7000,
            description="max neg Flex"
        ),
        al.AgentVariable(
            name="P_el_min_alg",
            value=7000,
            description="max pos Flex"
        ),
        al.AgentVariable(
            name="r_pel",
            value=20
        ),
    ]

    shared_variable_fields: List[str] = ["outputs"]


class FlexibilityModule(al.BaseModule):
    config: FlexibilityModuleConfig
    agent: al.Agent
    df: pd.DataFrame

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.results_file: Optional[str] = kwargs.get("results_file") or "results\\flexibility_results.csv"
        self.var_list = []
        for variable in self.variables:
            self.var_list.append(variable.name)
        self.time = []

    def register_callbacks(self):
        pass

    def get_results(self) -> Optional[pd.DataFrame]:
        """
        Opens results file of flexibilityindicators.py
        results_file defined in __init__
        Use function read_results_file()
        """
        results_file = self.results_file
        try:
            results = self.read_results_file(results_file)
            return results
        except FileNotFoundError:
            self.logger.error("Results file %s was not found.", results_file)
            return None

    def read_results_file(self, results_file: str):
        """
        Read results from csv and creates a Multiindex DataFrame

        Args:
            results_file: defined in __init__
        Returns:
            results: Multiindex DataFrame with calculated Flexibility Indicators and Inputs of class
        """
        # after all calculte, read resukts will be used at the end by results = mas.get_results()
        # the function overwrite the one from agent.py.

        ts = self.get("time_step").value
        n = self.get("prediction_horizon").value
        tvor = self.get("preperation_horizon").value
        x = int(tvor/ts)

        df = pd.read_csv(results_file)
        df = df.drop(columns=df.columns[0:1])  # Drops the first column of df

        # restructure df, start after tvor
        df_flexibility = df[["powerflex_flex_neg","powerflex_flex_pos",
                             "powerflex_avg_neg","powerflex_avg_pos",
                             "energyflex_neg","energyflex_pos",
                             "costs_neg","costs_pos","costs_neg_rel","costs_pos_rel"]].copy()
        nan_rows = pd.DataFrame(np.nan, index=range(x), columns=df_flexibility.columns)
        df_flexibility = pd.concat([nan_rows, df_flexibility], ignore_index=True)
        df_flexibility = df_flexibility[:-x]

        df = df.drop(columns=df_flexibility.columns)
        df_updated = pd.concat([df_flexibility,df], axis=1)

        num_rows = len(df)  # number of rows
        index_tuples = []
        for i in range(num_rows):
            for j in range(n + 1):
                index_tuples.append((i * ts, j * ts))
        multiindex = pd.MultiIndex.from_tuples(index_tuples)
        multiindex = multiindex[0:num_rows]

        results = df_updated.set_index(multiindex)
        # folgend done.
        # Resultfile wird im Prinzip aktuell noch nicht richtig gelesen,
        # weil die Werte bei 0/0 nicht 0/1800 starten -> Muss richtig interpretiert werden
        return results

    def check_initialisation(self, index_grid):
        """
        Checks by using a 'test_variable' whether the current process step is an initialisation step.

        Args:
            index_grid: Grid for pd.Series of the test_variable
        Returns:
            initialisation: Boolean which is True if the current process step is an initialisation step
        """
        initialisation = False
        test_variable = "P_el_alg"  # Should be adjust by use case (probably better as an input)
        value = self.get(test_variable).value
        #traj = pd.Series(values, index=index_grid)

        # if pd.Series(traj).notna().sum() == 1 and traj.iloc[0] is not np.nan:
        #     initialisation = True
        if isinstance(value, int):
            initialisation = True
        # if it is n initialisation step, the test_variable should be a int, which is given as a initial value
        # otherwise the test_variable is a series after one time_step, containing the collection points in ts*ph
        return initialisation

    def write_results(self, df, ts, n):
        """
        Write every data of variables in self.var_list in an DataFrame
        DataFrame will be updated every time step

        Args:
            df: DataFrame which is initialised as an empty DataFrame with columns according to self.var_list
            ts: time step
            n: number of time steps during prediction horizon
        Returns:
            DataFrame with results of every variable in self.var_list
        """
        results = []
        now = self.env.now
        grid = list(range(int(now), int(now) + (n + 1) * ts, ts))
        for name in self.var_list:
            values = self.get(name).value
            if isinstance(values, pd.Series):  # if a seires, align the index by using current time grid
                start = list(range(int(now), int(now) + (len(values)) * ts, ts))
                #start = [grid[0]]
                values = values.values
                traj = pd.Series(values, index=start).bfill().reindex(index=grid)
                results.append(traj)
            else:  # if a floot, add it as the first postion of the current time grid
                traj = pd.Series(np.nan, index=grid)
                if len(grid) > 0:
                    traj.iloc[0] = values
                results.append(traj)

        if not now % ts:
            self.time.append(now)
            new_df = pd.DataFrame(results).T  #.T transposes the DataFrame, switching rows and columns.
            new_df.columns = self.var_list
            df = pd.concat([df, new_df], ignore_index=True)  #concatenates new_df to the existing DataFrame df
        return df

    def cleanup_results(self):
        results_file = self.results_file
        if not results_file:
            return
        os.remove(results_file)

    def mean_value(self, arr, m):
        """
        General Function which calculates the mean value of an array
        Args:
            arr: array with NaN positions, which should get added by mean value of following positions
            m: number of collocation points
        Returns:
            results: Array with original values and the mean value added to the nan positions
        """
        result = []
        values = arr.values[:-1]  # ignore the last one, which should not be including in this 'now' time

        for i in range(0, len(values), m + 1):
            if np.isnan(values[i]):
                data = values[i:i + m + 1]  # slice one collection points (here 3 indexes)
                non_nan_values = np.nan_to_num(data, nan=0)  # change NaN to 0
                mean_value = np.sum(non_nan_values) / m
                result.append(mean_value)
                result.extend(data[1:])  # calculate mean value and add use it in NaN position
            else:
                result.extend(arr[i:i + m + 1])
        # length: original output from mpc agent, ts*ph including all collection points
        return result

    def process(self):

        df = pd.DataFrame(columns=self.var_list)
        ts = self.get("time_step").value
        n = self.get("prediction_horizon").value
        sampling_time = self.get("sampling_time").value
        initialisation_list = []
        index_grid = np.arange(0, ts * (n + 1), ts)

        while True:

            initialisation = self.check_initialisation(index_grid=index_grid)
            initialisation_list.append(initialisation)
            print("Initialisation:", initialisation_list)

            while initialisation == False:
                # calculate flexibility
                self.calc_flex(ts=ts, n=n)

                # write results
                df = self.write_results(df=df, ts=ts, n=n)
                df.to_csv(self.results_file)

                yield self.env.timeout(sampling_time)

            yield self.env.timeout(sampling_time)  # if initialisation step, dont do calc_flex

    def calc_flex(self, ts, n):
        """
        Calculates different flexibility indicators

        Args:
            ts: time step
            n: number of time steps during prediction horizon
        Sets:
            Power Flexibility
            Maximum and Minimum Power Flexibility
            Energy Flexibility
            Time Flexibility
            Flexibility Costs (absolut and relativ)
        """

        now = self.env.now

        t_KPI = self.get("flexibility_event").value
        t_prep = self.get("preperation_horizon").value
        j = int((t_KPI + t_prep) / ts)  # number of time_steps till end of flex.event
        x = int(t_KPI / ts)  # number of time_steps during flex.event

        index_grid_kpi = np.arange(t_prep, j * ts, ts)  #time grid in one prediction event tFE after tVor
        index_grid = np.arange(0, (n + 1) * ts, ts)  #time grid within one th, time step*pre.horizon
        #If now % ts is 0 , not 0 evaluates to True.
        if not now % ts:  # if now is evenly divisible by ts(filter out collocation point)

            pel = self.get("P_el_alg").value
            pel_max = self.get("P_el_max_alg").value
            pel_min = self.get("P_el_min_alg").value

            # index of pel represents a tuple in string form like "(a, b)"
            # splitting on ',' gives ['a', ' b'], and [1] gets the substring ' b' which, after removing spaces,
            # is 'b'
            time = pel.index.map(lambda x: float(x.strip('()').replace(' ', '').split(',')[1])).tolist()

            def count_false_after_true(lst):
                """
                Function counts how many positions in list has digits after a NaN
                Count indicates with how many collocation points the list was optimized

                Args:
                    lst: list with data which should be analysed
                Returns:
                    count: number of collocation points
                """
                count = 0
                found_true = False
                for item in lst:
                    if item:
                        if found_true:
                            break
                        found_true = True
                    elif found_true:
                        count += 1
                return count

            missing_indices = np.isnan(
                pel)  #if values in pel(series) is NaN, change it to 'True', if not, change the values to 'False'
            m = count_false_after_true(
                missing_indices)  #count how many collection points in one time step by counting False after True
            pel = self.mean_value(arr=pel, m=m)  # pel is a list, get mean value of pel for every time step (because of collocation points)
            pel_max = self.mean_value(arr=pel_max, m=m)
            pel_min = self.mean_value(arr=pel_min, m=m)
            # If the new index(index_grid) has entries that are not in the time[:-1], those entries will have NaN values in the resulting Series. If the original index has entries not in the new index, those entries are dropped.
            pel = pd.Series(pel, index=time[:-1]).reindex(
                index=index_grid)  #Reindexing aligns the Series to the new index.
            pel_max = pd.Series(pel_max, index=time[:-1]).reindex(index=index_grid)
            pel_min = pd.Series(pel_min, index=time[:-1]).reindex(index=index_grid)

            self.set("P_el_alg", pel)
            self.set("P_el_max_alg", pel_max)
            self.set("P_el_min_alg", pel_min)

            # Calculate difference between power of mpcs
            # If the difference is below zero a case distinction is made:
            # If the percentage deviation compared to the baseline performance is less than 1 %,
            # the difference is not included, as numerical effects are assumed.
            # If the deviation is higher, the effect should be considered

            powerflex_flex_neg = []
            for i in range(len(pel_max)):
                diff = pel_max.values[i] - pel.values[i]

                if diff < 0:
                    percentage_diff = (abs(diff) / pel.values[i]) * 100

                    if percentage_diff < 1:
                        powerflex_flex_neg.append(0)
                    else:
                        powerflex_flex_neg.append(diff)
                else:
                    powerflex_flex_neg.append(diff)

            powerflex_flex_neg = pd.Series(powerflex_flex_neg, index=index_grid)
            powerflex_flex_neg_kpi = powerflex_flex_neg.reindex(index=index_grid_kpi)

            powerflex_flex_pos = []
            for i in range(len(pel_min)):
                diff = pel.values[i] - pel_min.values[i]

                if diff < 0:
                    percentage_diff = (abs(diff) / pel.values[i]) * 100

                    if percentage_diff < 1:
                        powerflex_flex_pos.append(0)
                    else:
                        powerflex_flex_pos.append(diff)
                else:
                    powerflex_flex_pos.append(diff)

            powerflex_flex_pos = pd.Series(powerflex_flex_pos, index=index_grid)
            powerflex_flex_pos_kpi = powerflex_flex_pos.reindex(index=index_grid_kpi)


            self.set("powerflex_flex_neg", powerflex_flex_neg_kpi)  # in Series
            self.set("powerflex_flex_pos", powerflex_flex_pos_kpi)  # in Series

            # Energy flexibility and Average Power Flexibility
            energyflex_neg = np.sum(powerflex_flex_neg_kpi * ts) / 3.6e6  # 3.6e6 [J] to [MWh]

            powerflex_avg_neg = np.sum(powerflex_flex_neg_kpi) / (x * 1e3)  # [W] to [kW]

            self.set("energyflex_neg", energyflex_neg)
            self.set("powerflex_avg_neg", powerflex_avg_neg)

            # Positive Flexibility
            energyflex_pos = np.sum(powerflex_flex_pos_kpi * ts) / (3.6e6)

            powerflex_avg_pos = np.sum(powerflex_flex_pos_kpi) / (x * 1e3)

            self.set("energyflex_pos", energyflex_pos)
            self.set("powerflex_avg_pos", powerflex_avg_pos)

            #Calc Costs

            # Flexibility Costs as difference between costs of shadow mpc and baseline mpc
            # Costs of mpcs as the product of power and power price during the whole prediction horizon

            Pel_Price = self.get("r_pel").value
            Pel_Price.index = Pel_Price.index - Pel_Price.index[0]
            Pel_Price = Pel_Price.reindex(index=index_grid)

            s = int(len(pel) - 1)

            cost_neg = powerflex_flex_neg[:s] * Pel_Price[:s] / 1000 / 3600  # ct/s
            costs_neg = np.sum(cost_neg * ts)  #ct

            cost_pos = powerflex_flex_pos[:s] * Pel_Price[:s] * (-1) / 1000 / 3600  # ct/s
            costs_pos = np.sum(cost_pos * ts)  #ct

            self.set("costs_neg", costs_neg)
            self.set("costs_pos", costs_pos)

            # Relative Flexibility Costs as deviation of absolute costs for whole prediction horizon
            # and energy flexibility during flexibility event

            if energyflex_neg == 0:
                costs_neg_rel = inf
            else:
                costs_neg_rel = costs_neg / energyflex_neg

            if energyflex_pos == 0:
                costs_pos_rel = inf
            else:
                costs_pos_rel = costs_pos / energyflex_pos

            self.set("costs_neg_rel", costs_neg_rel)
            self.set("costs_pos_rel", costs_pos_rel)
