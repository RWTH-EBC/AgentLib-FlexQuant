import pytest
import pandas as pd
import os
import sys
from pathlib import Path
import importlib.util

# Add the project root to the Python path to allow for absolute imports
# This helps in locating the flexibility_quantification package if needed
root_path = Path(__file__).parent.parent
sys.path.insert(0, str(root_path))


def run_example_from_path(example_path: Path):
    """
    Dynamically imports and runs the 'run_example' function from a script
    in the specified directory.

    This function robustly handles changing the working directory AND the
    Python import path, ensuring the script can find both its local files
    and its local modules.
    """
    run_script_path = example_path / 'main_one_room_flex.py'
    if not run_script_path.is_file():
        raise FileNotFoundError(
            f"Could not find the run script at {run_script_path}. "
            "Please ensure it is named 'run.py' or adjust the test code."
        )

    # --- SETUP: Store original paths before changing them ---
    original_cwd = Path.cwd()
    original_sys_path = sys.path[:]  # Create a copy of the sys.path list

    try:
        # --- STEP 1: Change CWD for file access (e.g., config.json) ---
        os.chdir(example_path)

        # --- STEP 2: Add example dir to sys.path for module imports ---
        sys.path.insert(0, str(example_path))

        # Dynamically import the run_example function from the script
        spec = importlib.util.spec_from_file_location("run_module", run_script_path)
        run_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(run_module)

        if not hasattr(run_module, 'run_example'):
            raise AttributeError(
                "The 'run.py' script must contain a 'run_example' function.")

        # Execute the function and get the results
        results = run_module.run_example()
        return results

    finally:
        # --- TEARDOWN: Always restore original paths to avoid side-effects ---
        os.chdir(original_cwd)
        sys.path[:] = original_sys_path  # Restore the original sys.path


def test_oneroom_simple_mpc(snapshot):
    """
    Unit test for the oneroom_simpleMPC example using snapshot testing.

    This test runs the example via its own run script and compares the
    full resulting dataframes against stored snapshots.
    """
    # Define the path to the example directory
    example_path = root_path / 'Examples' / 'oneroom_simpleMPC'

    # Run the example and get the results object
    res = run_example_from_path(example_path)

    # Extract the full resulting dataframes as requested
    df_neg_flex_res = res["NegFlexMPC"]["NegFlexMPC"]
    df_pos_flex_res = res["PosFlexMPC"]["PosFlexMPC"]
    df_baseline_res = res["FlexModel"]["Baseline"]
    df_indicator_res = res["FlexibilityIndicator"]["FlexibilityIndicator"]

    # Assert that the entire DataFrame matches the snapshot.
    # We convert to JSON because it's a stable, human-readable format.
    snapshot.assert_match(
        df_neg_flex_res.to_json(orient='split', indent=2),
        'oneroom_simpleMPC_neg_flex.json'
    )
    snapshot.assert_match(
        df_pos_flex_res.to_json(orient='split', indent=2),
        'oneroom_simpleMPC_pos_flex.json'
    )
    snapshot.assert_match(
        df_baseline_res.to_json(orient='split', indent=2),
        'oneroom_simpleMPC_baseline.json'
    )
    snapshot.assert_match(
        df_indicator_res.to_json(orient='split', indent=2),
        'oneroom_simpleMPC_indicator.json'
    )