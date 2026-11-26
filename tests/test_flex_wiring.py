"""Tests of the communication wiring that the generator creates for the indicator.

The generated configs of the OneRoom_SimpleMPC example are checked, without running
the multi-agent system.
"""
import json
import shutil
import sys
from pathlib import Path

import pytest
from util import module_cleanup  # noqa: F401 (fixture)

root_path = Path(__file__).parent.parent
sys.path.insert(0, str(root_path))

import agentlib_flexquant.data_structures.globals as glbs  # noqa: E402
from agentlib_flexquant.generate_flex_agents import FlexAgentGenerator  # noqa: E402
from agentlib_flexquant.modules.flexibility_indicator import (  # noqa: E402
    FlexibilityIndicatorModuleConfig,
)
from agentlib.core.errors import ConfigurationError  # noqa: E402

EXAMPLE = root_path / "examples" / "OneRoom_SimpleMPC"
FLEX_CONFIG = Path("flex_configs") / "flexibility_agent_config.json"
MPC_CONFIG = Path("mpc_and_sim") / "simple_model.json"


@pytest.fixture
def example_copy(tmp_path, monkeypatch):
    """Copy the configs and models of the example and work inside the copy."""
    for folder in ("flex_configs", "mpc_and_sim"):
        shutil.copytree(EXAMPLE / folder, tmp_path / folder)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _module(path: str, module_type_part: str) -> dict:
    with open(path, encoding="utf-8") as file:
        agent = json.load(file)
    return next(m for m in agent["modules"] if module_type_part in str(m.get("type")))


def _aliases(variables: list) -> dict:
    return {v["name"]: v.get("alias", v["name"]) for v in variables}


def _generate() -> dict:
    paths = FlexAgentGenerator(
        flex_config=str(FLEX_CONFIG), mpc_agent_config=str(MPC_CONFIG)
    ).generate_flex_agents()
    return dict(zip(["baseline", "pos", "neg", "indicator"], paths))


def test_efficiency_comes_from_baseline_only(example_copy, module_cleanup):
    """The indicator listens only to the efficiency sent by the Baseline MPC."""
    paths = _generate()

    baseline_outputs = _aliases(_module(paths["baseline"], "mpc")["outputs"])
    assert baseline_outputs["eta_heater"] == glbs.ETA_THERMAL_ALIAS_BASE
    assert baseline_outputs["P_el"] == glbs.POWER_ALIAS_BASE
    assert baseline_outputs["E_out"] == glbs.STORED_ENERGY_ALIAS_BASE

    # the Shadow MPCs keep sharing their efficiency under its own name,
    # which the indicator no longer listens to
    for direction in ("pos", "neg"):
        shadow_outputs = _aliases(_module(paths[direction], "mpc")["outputs"])
        assert shadow_outputs["eta_heater"] == "eta_heater"

    indicator = _module(paths["indicator"], "indicator")
    indicator_inputs = _aliases(indicator["inputs"])
    # the input keeps its name (column in the results file) but not its alias
    assert indicator_inputs["eta_heater"] == glbs.ETA_THERMAL_ALIAS_BASE
    assert "eta_heater" not in indicator_inputs.values()

    # loading the generated indicator config keeps the alias
    config = FlexibilityIndicatorModuleConfig(**indicator, _agent_id="indicator")
    loaded = {v.name: v.alias for v in config.inputs}
    assert loaded["eta_heater"] == glbs.ETA_THERMAL_ALIAS_BASE
    assert None not in loaded


def test_efficiency_must_be_baseline_output(example_copy, module_cleanup):
    """An efficiency variable that the Baseline MPC does not output is rejected."""
    with open(FLEX_CONFIG, encoding="utf-8") as file:
        flex_config = json.load(file)
    indicator_module = next(
        m for m in flex_config["indicator_config"]["agent_config"]["modules"]
        if "indicator" in str(m.get("type"))
    )
    indicator_module["correct_costs"]["eta_thermal_base"] = "not_an_output"
    with open(FLEX_CONFIG, "w", encoding="utf-8") as file:
        json.dump(flex_config, file)

    with pytest.raises(ConfigurationError, match="not_an_output"):
        _generate()


def test_no_efficiency_input_without_efficiency():
    """Without eta_thermal_base, the indicator gets no input named None."""
    config = FlexibilityIndicatorModuleConfig(
        module_id="indicator", type="agentlib_flexquant.flexibility_indicator",
        _agent_id="indicator",
    )
    assert all(v.name is not None for v in config.inputs)
