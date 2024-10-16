import pandas as pd
import ast
import numpy as np


# Different functions for reading the results files
def create_mpc_series(results_mpc: pd.DataFrame, variable_mpc: str, value_type: str):
    res_mpc = results_mpc.copy()
    value = res_mpc[value_type][variable_mpc]
    # Select only the entries where the second level of the index is 0
    series = value[value.index.get_level_values(1) == 0]
    # dropping the second level
    series = series.reset_index(level=1, drop=True)
    return series


def create_pel_series(results_mpc: pd.DataFrame, variable_mpc: str, initial_time):
    res_mpc = results_mpc.copy()
    value = res_mpc[variable_mpc]
    series = value[value.index.get_level_values(1) == 0]
    series = series.reset_index(level=1, drop=True)
    series.index = series.index + initial_time
    return series


def get_first_val_from_logger(logger_series: pd.Series):
    time_list = []
    value_list = []
    for time_dict in logger_series:
        time_dict = ast.literal_eval(time_dict)  # convert str to dict
        keys = [float(x) for x in time_dict.keys()]
        values = [float(x) for x in time_dict.values()]
        time_dict = dict(zip(keys, values))
        time_list.append(list(time_dict.keys())[0])
        value_list.append(list(time_dict.values())[0])
    return pd.Series(value_list, index=time_list, dtype="float32")


def create_flex_series(results: pd.DataFrame, time_step: float):
    outer_index = results.index.get_level_values(0)
    idx = np.searchsorted(outer_index, time_step, side="left")
    if idx > 0 and (
            idx == len(outer_index)
            or np.fabs(time_step - outer_index[idx - 1])
            < np.fabs(time_step - outer_index[idx])
    ):
        closest = outer_index[idx - 1]
    else:
        closest = outer_index[idx]

    data_at_ts = results.loc[closest]
    data_at_ts = data_at_ts.copy()
    data_at_ts.index = data_at_ts.index + closest

    return data_at_ts


def create_flex_firststep(results: pd.DataFrame, var: str):
    res_mpc = results.copy()
    series = res_mpc[var]
    res = series[series.index.get_level_values(1) == 0]
    index = res.index.get_level_values(0)
    df = pd.DataFrame(res.values, index=index, columns=[var])
    return df


def create_flex_after_pre(results: pd.DataFrame, var: str):
    res_mpc = results.copy()
    series = res_mpc[var]
    #TODO: add right location of tvor/t_sample to find value
    res = series[series.index.get_level_values(1) == 1800]
    index = res.index.get_level_values(0)
    df = pd.DataFrame(res.values, index=index, columns=[var])
    return df


def extract_min_max_flex(series):
    min_result = {}
    max_result = {}
    grouped_index = series.groupby(level=0)
    for key, group in grouped_index:
        non_na_values = group.dropna()
        if not non_na_values.empty:
            # Find the maximum and minimum values and their corresponding second-level indices
            max_value = non_na_values.max()
            min_value = non_na_values.min()
            max_value_index = non_na_values.idxmax()
            min_value_index = non_na_values.idxmin()
            # Use the first-level index of the maximum and minimum values as the new index
            max_result[max_value_index[0]] = max_value
            min_result[min_value_index[0]] = min_value

    min_series = pd.Series(min_result, name="min_values")/1000
    max_series = pd.Series(max_result, name="max_values")/1000

    return min_series, max_series


def process_collocation_points(results_mpc: pd.DataFrame, variable_mpc, value_type, ts) -> pd.Series:
    """
    处理多层索引的Series，计算每个第一层索引下，第二层索引为0和ts之间的数据平均值，
    并将平均值放入第二层索引为0的行中，最后只保留第二层索引为0的行。

    参数：
    value : pd.Series
        多层索引的Series，其中每个第一层索引下包含多行数据。
    ts : int
        第二层索引的时间步长，用于确定计算平均值的范围。

    返回：
    pd.Series
        处理后的Series，只保留第二层索引为0的行。
    """
    res_mpc = results_mpc.copy()
    value = res_mpc[value_type][variable_mpc]
    # 创建一个value的副本来进行编辑
    edited_value = value.copy()

    # 获取第一层索引的唯一值，逐层处理
    for level1 in value.index.get_level_values(0).unique():
        # 筛选出该第一层索引下的所有数据
        subset = value.loc[level1]

        # 找到第二层索引中为0和ts的行
        idx_0 = subset.index[subset.index == 0]
        idx_ts = subset.index[subset.index == ts]

        if len(idx_0) == 0 or len(idx_ts) == 0:
            # 如果没有0或者ts的索引，跳过该层处理
            continue

        # 对于每个相邻的 0 和 ts 进行处理
        for i in range(len(idx_0)):
            start_idx = idx_0[i]
            end_idx = idx_ts[i]

            # 获取 start_idx 和 end_idx 之间的数据
            between_values = subset[(subset.index > start_idx) & (subset.index < end_idx)]

            if len(between_values) > 0:
                # 计算平均值，忽略 NaN
                avg_value = between_values.mean()

                # 将平均值放在 start_idx 处
                edited_value.loc[(level1, start_idx)] = avg_value

    # 只保留第二层索引为 0 的行
    edited_value = edited_value[edited_value.index.get_level_values(1) == 0]
    edited_value = edited_value.reset_index(level=1, drop=True)

    return edited_value