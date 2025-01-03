import pandas as pd
import numpy as np


def strip_multi_index(series: pd.Series):
    vals = pd.Series(_set_mean_values(series), index=series.index[:-1])
    # Convert the index (communicated as string) into a MultiIndex
    vals.index = vals.index.map(lambda x: eval(x))
    vals.index = pd.MultiIndex.from_tuples(vals.index)
    # vals is multicolumn so get rid of first value (start time of predictions)
    vals.index = vals.index.get_level_values(1).astype(float)
    return vals


def fill_nans(series: pd.Series):
    return pd.Series(_set_mean_values(series), index=series.index[:-1])


def _set_mean_values(series: pd.Series):
    """Helper function to set the mean values for the collocation points

    """
    # TODO: find a better solution, like using the simulator for the values
    # TODO: clean up -> check issue from peter
    def count_false_after_true(lst):
        """ Counts the nans, effectively the collocation order """
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

