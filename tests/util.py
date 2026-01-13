import pytest
import sys
import json
from pathlib import Path
from typing import Union, List


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


def convert_paths_to_absolute_in_json(
        json_paths: List[Union[str, Path]],
        base_path: Union[str, Path, None] = None,
) -> None:
    """
    Load JSON config files, convert all relative paths to absolute paths,
    and save them back to the same files.

    Args:
        json_paths: List of paths to JSON config files to process.
        base_path: Base directory to resolve relative paths against.
                   If None, uses the parent of the 'tests' directory.
    """
    path_keys = {
        "file", "results_file", "result_filename", "market_config",
        "flex_files_directory", "results_directory", "flex_base_directory_path",
    }
    path_extensions = {".py", ".csv", ".json"}

    def is_path_like(key: str, value: str) -> bool:
        """Determine if a string value looks like a file path."""
        if not isinstance(value, str) or not value:
            return False
        if key.lower() in path_keys:
            return True
        if "\\" in value or "/" in value:
            for ext in path_extensions:
                if value.lower().endswith(ext):
                    return True
            if "sample_files" in value or "tests" in value:
                return True
        return False

    def resolve_path(path_str: str, base: Path) -> str:
        """Convert a relative path string to an absolute path."""
        if not path_str:
            return path_str
        normalized = path_str.replace("\\", "/")
        path = Path(normalized)
        if path.is_absolute():
            return str(path)
        return str((base / path).resolve())

    def process_value(key: str, value, base: Path):
        """Recursively process a value, converting paths where found."""
        if isinstance(value, dict):
            return {k: process_value(k, v, base) for k, v in value.items()}
        elif isinstance(value, list):
            return [process_value(key, item, base) for item in value]
        elif isinstance(value, str) and is_path_like(key, value):
            return resolve_path(value, base)
        return value

    for json_path in json_paths:
        json_path = Path(json_path).resolve()

        # Determine base path
        if base_path is None:
            # Find project root by looking for 'tests' directory
            current = json_path
            while current.name != "tests" and current.parent != current:
                current = current.parent
            if current.name == "tests":
                resolved_base = current.parent
            else:
                resolved_base = json_path.parent
        else:
            resolved_base = Path(base_path).resolve()

        # Load, process, and save
        with open(json_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        config = process_value("", config, resolved_base)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
