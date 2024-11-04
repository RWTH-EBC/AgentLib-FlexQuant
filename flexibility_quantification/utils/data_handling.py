from pathlib import Path
from typing import Union

import pandas as pd
import numpy as np

from agentlib_mpc.utils.analysis import load_sim, load_mpc
from flexibility_quantification.data_structures.mpcs import BaselineMPCData, PFMPCData, NFMPCData


def strip_multi_index(series: pd.Series):
    vals = pd.Series(_set_mean_values(series), index=series.index[:-1])
    # Convert the index (communicated as string) into a MultiIndex
    vals.index = vals.index.map(lambda x: eval(x))
    vals.index = pd.MultiIndex.from_tuples(vals.index)
    # vals is multicolumn so get rid of first value (start time of predictions)
    vals.index = vals.index.get_level_values(1).astype(float)
    return vals


def _set_mean_values(series: pd.Series):
    """ Helper function to set the mean values for the collocation points
    """
    # TODO: find a better solution, like using the simulator for the values
    # TODO: clean up
    def count_false_after_true(lst):
        """ Counts the nans, effectively the collocation order """
        # TODO: add collocation order in config. Generator should set this value
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
    missing_indices = np.isnan(series)
    m = count_false_after_true(missing_indices)
    result = []
    values = series.values[:-1]

    for i in range(0, len(values), m + 1):
        if np.isnan(values[i]):
            data = values[i:i + m + 1]
            non_nan_values = np.nan_to_num(data, nan=0)
            mean_value = np.sum(non_nan_values)/m
            result.append(mean_value)
            result.extend(data[1:])
        else:
            result.extend(series[i:i + m + 1])

    return result


def load_indicator(file_path: Path) -> pd.DataFrame:
    """Load the flexibility indicator results from the given file path
    """
    df = pd.read_csv(file_path, header=0, index_col=[0, 1])
    return df


def load_market(file_path: Path) -> pd.DataFrame:
    """Load the market results from the given file path
    """
    df = pd.read_csv(file_path, header=0, index_col=[0, 1])
    return df


RES_TYPE = dict[str, dict[str, pd.DataFrame]]

baselineID = BaselineMPCData().module_id
posFlexID = PFMPCData().module_id
negFlexID = NFMPCData().module_id


def load_results(res_path: Union[str, Path]) -> RES_TYPE:
    """
    Load the results from the given path in the same format as the results from the agentlib

    Keyword arguments:
    res_path -- The path to the results folder as string or pathlib.Path object
    """
    results = {
        "SimAgent": {
            "room": load_sim(Path(res_path, "sim_room.csv"))
        },
        "FlexModel": {
            baselineID: load_mpc(Path(res_path, f"mpc{BaselineMPCData().results_suffix}"))
        },
        posFlexID: {
            posFlexID: load_mpc(Path(res_path, f"mpc{PFMPCData().results_suffix}"))
        },
        negFlexID: {
            negFlexID: load_mpc(Path(res_path, f"mpc{NFMPCData().results_suffix}"))
        },
        "FlexibilityIndicator": {
            "FlexibilityIndicator": load_indicator(Path(res_path, "flexibility_indicator.csv"))
        },
        "FlexibilityMarket": {
            "FlexibilityMarket": load_market(Path(res_path, "flexibility_market.csv"))
        },
    }
    return results


TIME_CONV_FACTOR = {
    "s": 1,
    "min": 60,
    "h": 3600,
    "d": 86400,
}


def convert_timescale_index(results: RES_TYPE, time_unit: str = "h") -> RES_TYPE:
    """
    Convert the timescale of a dataframe index (from seconds) to the given time unit

    Keyword arguments:
    results -- The dictionary of the results with the dataframes
    time_unit -- The time unit to convert the index to (default "h"; options: "s", "min", "h", "d"; assumption: index is in seconds)
    """
    for key, value in results.items():
        for sub_key, sub_value in value.items():
            if isinstance(sub_value.index, pd.MultiIndex):
                sub_value.index = pd.MultiIndex.from_arrays([sub_value.index.get_level_values(level) / TIME_CONV_FACTOR[time_unit] for level in range(sub_value.index.nlevels)])
            else:
                sub_value.index = sub_value.index / TIME_CONV_FACTOR[time_unit]
    return results
