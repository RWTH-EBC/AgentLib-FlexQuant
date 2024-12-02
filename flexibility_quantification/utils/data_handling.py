from typing import Literal

import pandas as pd

MEAN = "mean"
INTERPOLATE = "interpolate"
FillNaMethods = Literal["mean", "interpolate"]


def strip_multi_index(series: pd.Series) -> pd.Series:
    # Convert the index (communicated as string) into a MultiIndex
    series.index = series.index.map(lambda x: eval(x))
    series.index = pd.MultiIndex.from_tuples(series.index)
    # vals is multicolumn so get rid of first value (start time of predictions)
    series.index = series.index.get_level_values(1).astype(float)
    return series


def fill_nans(series: pd.Series, method: FillNaMethods) -> pd.Series:
    if method == MEAN:
        series = _set_mean_values(series=series)
    elif method == INTERPOLATE:
        # Interpolate missing values
        series = series.interpolate(method="index", limit_direction="both")
    return series


def _set_mean_values(series: pd.Series) -> pd.Series:
    """ Fills intervals including the nan with the mean of the following values. """
    def _get_intervals_for_mean(s: pd.Series) -> list[pd.Interval]:
        intervals = []
        start: int = None
        end: int
        for index, value in s.items():
            if pd.isna(value):
                if pd.isna(start):
                    start = index
                else:
                    end = index
                    intervals.append(pd.Interval(left=start, right=end, closed="right"))
                    start = end
        return intervals

    for interval in _get_intervals_for_mean(series):
        interval_index = (interval.left <= series.index) & (series.index < interval.right)
        series[interval_index] = series[interval_index].mean(skipna=True)

    return series

