import pandas as pd
import matplotlib.pyplot as plt

from Examples.SimpleBuilding.predictor.simple_predictor import PredictorModule, PredictorModuleConfig, FlexEvent, amb_temp_func
from agentlib.utils.multi_agent_system import LocalMASAgency
from agentlib.core.environment import Environment

# Example grids and corresponding values for testing
# test_cases = [
#     ([0, 7200, 10800, 28800], [292, 295, 292, 295]),  # Original case
#     ([0, 3600, 7200, 10800], [290, 292, 295, 293]),
#     ([0, 1800, 3600, 5400, 7200], [288, 290, 292, 294, 296]),
#     ([0, 900, 1800, 2700, 3600], [285, 287, 289, 291, 293]),
# ]


# def temp_trajectory_adapter(grid, values_ub, slope ):
#     traj_ub = pd.Series(values_ub, index=grid)
#     traj_ub_copy = traj_ub.copy()  # Create a copy of the original series

#     for i in grid[1:]:
#         print(f"\n---\ni = {i}")
#         next_idx = traj_ub.index.get_loc(i) + 1
#         previous_idx = traj_ub.index.get_loc(i) - 1
#         previous_time = traj_ub.index[previous_idx]
#          # Handle the case where next_idx is out of bounds
#         time_delta = abs(traj_ub[i] - traj_ub.iloc[previous_idx]) / slope

#         new_time = i + time_delta

#         if i != grid[-1]:  # Skip the last element in the grid
#             try:
#                 next_time = traj_ub.index[next_idx] 
#                 if new_time < next_time:
#                     print(f"time_delta: {time_delta}")
#                     print(f"new_time: {new_time}")

#                     traj_ub_copy.at[new_time] = traj_ub[i]
#                     traj_ub_copy.at[i] = traj_ub[previous_time]
#                     traj_ub_copy = traj_ub_copy.sort_index()
#                     print(traj_ub_copy)
#                 # Check if new_time is less than the next_time
#                 if new_time == next_time:
#                     print(f"new_time {new_time} is equal to next_time {next_time}, skipping update.")
#                     traj_ub_copy.at[i] = traj_ub[previous_time]
#                     traj_ub_copy.at[next_time] = traj_ub[i]  # Keep the next time value unchanged
                    
#                 if new_time > next_time:
#                     raise ValueError(f"new_time {new_time} is not less than next_time {next_time}")
                

            
#             except Exception as e:
#                 print(f"Error: {e}")
#                 traj_ub_copy.at[next_time] = traj_ub[i]
#                 traj_ub_copy.at[i] = traj_ub[previous_time]
#             finally:
#                 # Ensure the series is sorted after each iteration

#                 traj_ub_copy = traj_ub_copy.sort_index()
#         else:
            
#             traj_ub_copy.at[new_time] = traj_ub[i]
#             # Remove duplicates where the value appears twice and is the same in traj_ub
#             traj_ub_copy.at[i] = traj_ub[previous_time]
#             traj_ub_copy = traj_ub_copy.sort_index()

    
#     # Duplicate the last value and set it equal to the second last value
#     print("\n---\nFinalizing trajectory...")
#     traj_ub_copy = traj_ub_copy._append(pd.Series([traj_ub.iloc[-1]], index=[traj_ub.index[-1]]))
#     traj_ub_copy = traj_ub_copy[~traj_ub_copy.index.duplicated(keep='first')]
#     traj_ub_copy = traj_ub_copy.sort_index()
    
#     print("\nFinal Series:")
#     print(traj_ub_copy)
#     return traj_ub_copy

# temp_trajectory_adapter(grid=[0, 3600, 7200, 10800], values_ub=[290, 292, 295, 293], slope = 3 / 3600)

    
# for idx, (grid, values_ub) in enumerate(test_cases):
#     print(f"\nTesting with grid: {grid} and values: {values_ub}")
#     traj_ub = pd.Series(values_ub, index=grid)
#     slope = 3 / 3600  # Slope for testing
#     result = temp_trajectory_adapter(grid, values_ub, slope)
    
#     plt.figure(figsize=(8, 4))
#     plt.plot(result.index, result.values, marker='o')
#     plt.xlabel(f"Time (grid: {grid})")
#     plt.ylabel(f"Value (values: {values_ub})")
#     plt.title(f"Temperature Trajectory for Test Case {idx+1}")
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()

grid = [0, 7200, 10800, 28800]  # Example grid in seconds
values_ub = [292, 295, 292, 295]  # Example upper boundary values
values_lb = [290, 293, 290, 293]  # Example lower boundary values
traj_ub = pd.Series(grid , values_ub )
traj_lb = pd.Series(grid  , values_lb )


env_config= {"rt": False, "factor": 10, "t_sample": 60} 

env = Environment(config=env_config)



   
    

class FlexEventSetter:
    def __init__(self, flex_event: FlexEvent, env):
        self.flex_event = flex_event.as_dict()
        self.env = env
    def shift_temperature_for_flex_event(self, temp_series: pd.Series, bound: str ) -> pd.Series:
        shifted_series = temp_series.copy()

        if self.flex_event["start"] is not None and self.flex_event["end"] is not None:
            start = self.flex_event["start"]
            end = self.flex_event["end"]
            now = self.env.now

            new_index = sorted(set(temp_series.index).union({now + start, now + end}))
            shifted_series = temp_series.reindex(new_index).interpolate(method='index')

            start_idx = shifted_series.index.searchsorted(now + start)
            end_idx = shifted_series.index.searchsorted(now + end)

            if bound == "upper" and self.flex_event.get("upper_boundary_shift") is not None:
                shifted_series.iloc[start_idx:end_idx + 1] += self.flex_event["upper_boundary_shift"]
                # Set the shadow variable
                self.set("T_upper_shadow", shifted_series)
            elif bound == "lower" and self.flex_event.get("lower_boundary_shift") is not None:
                shifted_series.iloc[start_idx:end_idx + 1] -= self.flex_event["lower_boundary_shift"]
                # Set the shadow variable
                self.set("T_lower_shadow", shifted_series)

        return shifted_series

# Usage:
def run_example():
    flex_event = FlexEvent(
        upper_boundary_shift=2,
        lower_boundary_shift=2,
        gradient=0.1,
        start=3600,
        end=10000
    )
    
   
    flex_event_setter = FlexEventSetter(flex_event, env)

    # Shift temperature series for upper boundary
    shifted_traj_ub = flex_event_setter.shift_temperature_for_flex_event(traj_ub, "upper")
    shifted_traj_lb = flex_event_setter.shift_temperature_for_flex_event(traj_lb, "lower")
    print(f"Shifted Upper Boundary Trajectory: {shifted_traj_ub}")
    print(f"Shifted Lower Boundary Trajectory: {shifted_traj_lb}")


if "__main__" == __name__:
    run_example()