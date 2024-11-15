import pandas as pd

from agentlib_mpc.utils import TimeConversionTypes, TIME_CONVERSION


def convert_timescale_index(results: dict[str, dict[str, pd.DataFrame]], timescale: TimeConversionTypes) -> dict[str, dict[str, pd.DataFrame]]:
    """ Convert the timescale of a dataframe index (from seconds) to the given time unit

    Keyword arguments:
    results -- The dictionary of the results with the dataframes
    time_unit -- The time unit to convert the index to
    """
    for key, value in results.items():
        for sub_key, sub_value in value.items():
            if isinstance(sub_value.index, pd.MultiIndex):
                sub_value.index = pd.MultiIndex.from_arrays([sub_value.index.get_level_values(level) / TIME_CONVERSION[timescale] for level in range(sub_value.index.nlevels)])
            else:
                sub_value.index = sub_value.index / TIME_CONVERSION[timescale]
    return results
