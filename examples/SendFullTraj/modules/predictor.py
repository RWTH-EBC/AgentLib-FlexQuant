import time
from pathlib import Path

import agentlib as al
import filelock
import pandas as pd
from agentlib.core import Agent
import numpy as np
from pydantic import Field
from typing import Optional, List


def read_csv_with_retry(filepath: str, max_retries: int = 10,
                        retry_delay: float = 0.1, lock_timeout: float = 1.0,
                        **read_csv_kwargs) -> pd.DataFrame:
    """
    Read a CSV file with file locking for concurrent access.

    Works cross-platform using the filelock library.
    """
    filepath = Path(filepath)
    lock_path = filepath.with_suffix(filepath.suffix + ".lock")
    lock = filelock.FileLock(lock_path, timeout=lock_timeout)

    last_exception = None

    for attempt in range(max_retries):
        try:
            with lock:
                return pd.read_csv(filepath, **read_csv_kwargs)
        except filelock.Timeout:
            last_exception = TimeoutError(f"Could not acquire lock for {filepath}")
        except (IOError, OSError, pd.errors.EmptyDataError) as e:
            last_exception = e

        sleep_time = retry_delay * (2 ** attempt) + np.random.uniform(0, 0.05)
        time.sleep(sleep_time)

    raise IOError(
        f"Failed to read {filepath} after {max_retries} attempts. "
        f"Last error: {last_exception}"
    )


def generate_outputs_from_disturbances(disturbances_path: str, prefix:str) -> List[dict]:
    """Generate output variable dicts from disturbance CSV columns."""
    try:
        df = read_csv_with_retry(disturbances_path, index_col=0, nrows=1)
        columns = df.columns.tolist()
    except FileNotFoundError:
        raise ValueError(f"Disturbances file not found: {disturbances_path}")
    except Exception as e:
        raise ValueError(f"Error reading disturbances file: {e}")

    dynamic_outputs = []
    for col in columns:
        dynamic_outputs.append({
            "name": f"{col}_prediction",
            "alias": f"{prefix}_{col}_prediction",
            "description": f"Prediction of {col}",
            "value": float(df[col].values[0])
        })
        dynamic_outputs.append({
            "name": f"{col}_measurement",
            "alias": f"{prefix}_{col}_measurement",
            "description": f"Measurement of {col}",
            "value": float(df[col].values[0])
        })

    return dynamic_outputs


class PredictorModuleConfig(al.BaseModuleConfig):
    """Module that outputs a prediction of the heat load at a specified interval.

    """

    outputs: al.AgentVariables = []

    disturbances_path: Optional[str] = Field(default="disturbance.csv")
    prefix: Optional[str] = Field(default="")
    sampling_time_predictor: Optional[int] = Field(default=60)
    prediction_horizon: Optional[int] = Field(default=97)
    time_step_mpc: Optional[int] = Field(default=900)

    def __init__(self, _agent_id, *args, **kwargs):
        # Modify kwargs before it gets copied to _user_config in super().__init__
        disturbances_path = kwargs.get("disturbances_path", "disturbance.csv")
        current_outputs = kwargs.get("outputs", [])
        prefix = kwargs.get("prefix", "")

        # Only generate if outputs is empty or not provided
        if disturbances_path and not current_outputs:
            dynamic_outputs = generate_outputs_from_disturbances(disturbances_path, prefix)
            kwargs["outputs"] = dynamic_outputs
        elif disturbances_path and current_outputs:
            # Merge: add dynamic outputs that aren't already defined
            dynamic_outputs = generate_outputs_from_disturbances(disturbances_path, prefix)
            existing_names = set()
            for out in current_outputs:
                if isinstance(out, dict):
                    existing_names.add(out.get("name"))
                elif hasattr(out, "name"):
                    existing_names.add(out.name)

            for dyn_out in dynamic_outputs:
                if dyn_out["name"] not in existing_names:
                    current_outputs.append(dyn_out)

            kwargs["outputs"] = current_outputs

        super().__init__(_agent_id, *args, **kwargs)


class PredictorModule(al.BaseModule):
    """Module that outputs a prediction of the heat load at a specified
    interval.

    """

    config: PredictorModuleConfig

    def register_callbacks(self):
        pass

    def __init__(self, *, config: dict, agent: Agent):
        super().__init__(config=config, agent=agent)
        # disturbance data
        self.disturbances = read_csv_with_retry(self.config.disturbances_path,
                                                index_col=0, dtype=float)
        # fill NaNs (for simulation studies most likely not the case)
        self.disturbances.interpolate(method="index", inplace=True)

    def process(self):
        """Sets a new prediction at each time step."""
        sample_time = self.config.sampling_time_predictor
        while True:
            self.send_disturbance_csv_profiles()
            yield self.env.timeout(sample_time)

    def send_disturbance_csv_profiles(self):
        """Sends all data given in the disturbances.csv file.
        Also handles non-equidistant files (e.g. from change-of-value systems)

        """
        now = self.env.time
        horizon_end = now + self.config.prediction_horizon * self.config.time_step_mpc + 1

        # Find position in index where 'now' would be inserted
        idx_array = self.disturbances.index.values
        start_pos = np.searchsorted(idx_array, now, side='right') - 1
        start_pos = max(0, start_pos)  # Ensure we don't go negative

        # If position is in the future, go one step back
        if idx_array[start_pos] > now and start_pos > 0:
            start_pos -= 1

        # Find the end position
        end_pos = np.searchsorted(idx_array, horizon_end, side='right')
        # Ensure we have at least one point
        end_pos = max(end_pos, start_pos + 1)

        for label in self.disturbances.columns.to_list():
            # Measurement: last active value
            single_dist_measurement = self.disturbances[label].iloc[start_pos]
            # Prediction: slice preserves the original index
            single_dist_prediction = self.disturbances[label].iloc[start_pos:end_pos]

            self.set(label + "_measurement", single_dist_measurement)
            self.set(label + "_prediction", single_dist_prediction)
