import pandas as pd
import numpy as np

def temp_trajectory_adapter(grid_or_series, values=None, slope=3/3600, flex_event=None):
    if isinstance(grid_or_series, pd.Series):
        traj_ub = grid_or_series.copy()
        traj_ub_copy = traj_ub.copy()  # Create a copy of the original series
        grid = traj_ub.index
    else:
        assert values is not None, "If not passing a Series, you must provide values."
        traj_ub = pd.Series(values, index=grid_or_series)
        traj_ub_copy = traj_ub.copy()  # Create a copy of the original series
        grid = grid_or_series

    for i in grid[1:]:
        next_idx = traj_ub.index.get_loc(i) + 1
        previous_idx = traj_ub.index.get_loc(i) - 1
        previous_time = traj_ub.index[previous_idx]
        time_delta = abs(traj_ub[i] - traj_ub.iloc[previous_idx]) / slope
        new_time = i + time_delta

        if i != grid[-1]:
            try:
                next_time = traj_ub.index[next_idx]
                if new_time < next_time:
                    traj_ub_copy.at[new_time] = traj_ub[i]
                    traj_ub_copy.at[i] = traj_ub[previous_time]
                    traj_ub_copy = traj_ub_copy.sort_index()
                if new_time == next_time:
                    traj_ub_copy.at[i] = traj_ub[previous_time]
                    traj_ub_copy.at[next_time] = traj_ub[i]
                if new_time > next_time:
                    raise ValueError(f"new_time {new_time} is not less than next_time {next_time}")
            except Exception:
                traj_ub_copy.at[next_time] = traj_ub[i]
                traj_ub_copy.at[i] = traj_ub[previous_time]
            finally:
                traj_ub_copy = traj_ub_copy.sort_index()
        else:
            traj_ub_copy.at[new_time] = traj_ub[i]
            traj_ub_copy.at[i] = traj_ub[previous_time]
            traj_ub_copy = traj_ub_copy.sort_index()
    traj_ub_copy = traj_ub_copy._append(pd.Series([traj_ub.iloc[-1]], index=[traj_ub.index[-1]]))
    traj_ub_copy = traj_ub_copy[~traj_ub_copy.index.duplicated(keep='first')]
    traj_ub_copy = traj_ub_copy.sort_index()
    
    return traj_ub_copy



def build_comfort_trajectory(grid, values_ub, slope, flex_event=None):
    """
    grid: list of time points (seconds)
    values_ub: list of comfort values
    slope: float, K/s
    flex_event: dict with keys 'start', 'end', 'upper_boundary_shift' (optional)
    Returns: pd.Series with shifted values if flex_event is given
    """
    grid = np.array(grid, dtype=float)
    values_ub = np.array(values_ub, dtype=float)

    # Optionally apply flex event shift
    if flex_event:
        start = flex_event.get("start")
        end = flex_event.get("end")
        shift = flex_event.get("upper_boundary_shift", 0)
        if start is not None and end is not None and shift:
            mask = (grid >= start) & (grid <= end)
            values_ub[mask] += shift

    # Optionally, interpolate for slope (if needed)
    # Example: linear interpolation between points
    grid = [0, 7200, 10400, 28800]
    values_ub = [292, 295, 292, 295]
    traj_ub = pd.Series(values_ub, index=grid)
    for i in grid[1:-1]:
        time_delta = abs(traj_ub[i] - traj_ub[i + 1]) / slope
        new_time = i + time_delta
        traj_ub.at[new_time] = traj_ub[i]
        traj_ub.at[i] = traj_ub.iloc[traj_ub.index.get_loc(i) - 1]


    return traj_ub

test_traj = pd.Series(
    [292, 295, 292, 295],
    index=[0, 7200, 10400, 28800]
)

def temp_shift_traj_adapter(temp_traj: pd.Series, current_time: float) -> float:

    start_idx = temp_traj.index.get_indexer([current_time], method='ffill')[0]
    start_value = temp_traj.iloc[start_idx]
    start_time = temp_traj.index[start_idx] if start_idx < len(temp_traj) else current_time
    end_idx = start_idx + 1 if start_idx + 1 < len(temp_traj) else start_idx
    end_value = temp_traj.iloc[end_idx] if end_idx < len(temp_traj) else start_value
    end_time = temp_traj.index[end_idx] if end_idx < len(temp_traj) else current_time
    print(f"current time: {current_time}, start time and value: {start_time, start_value}, end time and value: {end_time, end_value}")
    

    if start_value == end_value:
        temp = start_value
        print(f"Start and End values are equal: {start_value}. Returning start value.")
        return temp
    
    else:
        interpolated_value = np.interp(current_time, [start_time, end_time], [start_value, end_value])
        print(f"Values not equal, Interpolated Value: {interpolated_value}")
        return interpolated_value


# if __name__ == "__main__":
#     adapted_traj = temp_trajectory_adapter(test_traj, slope=3/3600)
#     print(f"Adapted Trajectory: {adapted_traj}")

#     for i in range(8):
#         temp_shift_traj_adapter(adapted_traj, i * 3600)
    