import numpy as np
import pandas as pd



def generate_comfort_trajectory(grid, values_ub, values_lb, slope, t_sample=60, heating=True):
    """
    Generates gradual comfort trajectories (upper and lower bounds) based on comfort setpoints.

    Args:
        grid (list or np.array): Time points (in seconds) marking start of each segment.
        values_ub (list): Upper boundary comfort values for each segment.
        values_lb (list): Lower boundary comfort values for each segment.
        slope (float): Rate of change in K/h.
        t_sample (int): Time step in seconds (default is 60s).
        heating (bool): Currently unused. Reserved for future logic if needed.

    Returns:
        Tuple[pd.Series, pd.Series]: Upper and lower boundary comfort trajectories.
    """
    def interpolate_values(start, end, slope_per_sec):
        values = []
        current = start
        while (current < end) if end > start else (current > end):
            values.append(current)
            delta = t_sample * slope_per_sec
            current = np.round(current + delta if end > start else current - delta, 3)
        values.append(end)  # Ensure the last value is exactly the endpoint
        return values

    slope_per_sec = slope / 3600

    def generate_trajectory(values):
        full_values, full_times = [], []
        for i in range(len(values) - 1):
            v_start, v_end = values[i], values[i + 1]
            time_start = grid[i]
            segment_values = interpolate_values(v_start, v_end, slope_per_sec)
            segment_times = [time_start + j * t_sample for j in range(len(segment_values))]

            full_values.extend(segment_values)
            full_times.extend(segment_times)
        return pd.Series(full_values, index=full_times)

    traj_ub = generate_trajectory(values_ub)
    traj_lb = generate_trajectory(values_lb)

    return traj_ub, traj_lb

