import pytest
import sys


@pytest.fixture
def module_cleanup():
    """
    A pytest fixture that cleans up any modules imported during a test.

    This works by taking a snapshot of sys.modules before the test runs
    and removing any new additions after the test completes. This is useful
    for ensuring test isolation when dynamically importing code.
    """
    # --- SETUP: Take a snapshot of currently loaded modules ---
    initial_modules = set(sys.modules.keys())

    # --- The test runs at this point ---
    yield

    # --- TEARDOWN: Clean up newly added modules ---
    final_modules = set(sys.modules.keys())
    newly_added_modules = final_modules - initial_modules

    for module_name in newly_added_modules:
        # We check again because some modules might be part of a package
        # and could have been removed when the parent was removed.
        if module_name in sys.modules:
            del sys.modules[module_name]


def round_floats_in_structure(obj, precision: int):
    """
    Recursively traverses a data structure and rounds any float values.
    Handles nested dictionaries and lists.
    """
    if isinstance(obj, float):
        # set the value to 0.0 if it is either 0.0 or -0.0
        if obj == 0:
            obj = 0.0
        return round(obj, precision)
    if isinstance(obj, dict):
        return {k: round_floats_in_structure(v, precision) for k, v in obj.items()}
    if isinstance(obj, list):
        return [round_floats_in_structure(elem, precision) for elem in obj]
    return obj
