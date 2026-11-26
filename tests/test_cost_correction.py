"""Tests of the correction of the costs by the stored energy.

Synthetic profiles on the MPC time grid are fed to the KPI calculation, and the corrected
costs are compared with
    costs - mean(c_el) * (dE_stored / mean(eta) + dE_stored_el),
where dE is the difference shadow - baseline at the end of the horizon.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

root_path = Path(__file__).parent.parent
sys.path.insert(0, str(root_path))

import agentlib_flexquant.data_structures.globals as glbs  # noqa: E402
from agentlib_flexquant.data_structures.flex_kpis import FlexibilityData  # noqa: E402
from agentlib_flexquant.modules.flexibility_indicator import (  # noqa: E402
    CallBackHandler,
    FlexibilityIndicatorModuleConfig,
)

PARAMS = dict(prep_time=900, market_time=900, flex_event_duration=1800, time_step=900,
              prediction_horizon=8)
GRID = np.arange(0, 9 * 900, 900)
PRICE = [30, 30, 35, 40, 40, 35, 30, 25, 25]
COP = [3.0, 3.1, 3.2, 3.0, 2.9, 3.1, 3.3, 3.4, 3.2]
POWER = {
    "base": [2, 2, 2, 2, 2, 2, 2, 2, 2],
    "pos": [2, 2, 2, 0.5, 0.5, 2.6, 2.4, 2, 2],
    "neg": [2, 2, 2, 4, 4, 1.5, 1.6, 2, 2],
}
# stored thermal and electrical energy; only the end values enter the correction
E_TH = {"base": [10] * 9, "pos": [10] * 8 + [9.2], "neg": [10] * 8 + [11.5]}
E_EL = {"base": [5] * 9, "pos": [5] * 8 + [4.0], "neg": [5] * 8 + [6.5]}
SHADOW = {"positive": "pos", "negative": "neg"}
SUFFIX = {"base": "base", "pos": "flex_pos", "neg": "flex_neg"}


def _series(values) -> pd.Series:
    return pd.Series(np.asarray(values, dtype=float), index=GRID)


def _kpis(with_el_storage: bool) -> FlexibilityData:
    data = FlexibilityData(**PARAMS)
    for mpc in ("base", "pos", "neg"):
        data.update_profile(f"power_profile_{SUFFIX[mpc]}", _series(POWER[mpc]), mpc=True)
        data.update_profile(f"stored_energy_profile_{SUFFIX[mpc]}", _series(E_TH[mpc]),
                            mpc=True)
        if with_el_storage:
            data.update_profile(f"stored_el_energy_profile_{SUFFIX[mpc]}",
                                _series(E_EL[mpc]), mpc=True)
    data.update_profile("eta_thermal_base", _series(COP), mpc=True)
    data.update_profile("electricity_price_series", _series(PRICE), mpc=False)
    data.update_profile("feed_in_price_series", _series([0] * 9), mpc=False)
    data.calculate(enable_energy_costs_correction=True, calculate_flex_cost=True,
                   integration_method=glbs.CONSTANT, collocation_time_grid=[])
    return data


@pytest.mark.parametrize("direction", ["positive", "negative"])
def test_corrected_costs_with_electrical_storage(direction):
    """Thermal energy is divided by the mean efficiency, electrical energy is not."""
    kpis = getattr(_kpis(with_el_storage=True), f"kpis_{direction[:3]}")
    shadow = SHADOW[direction]
    d_th = E_TH[shadow][-1] - E_TH["base"][-1]
    d_el = E_EL[shadow][-1] - E_EL["base"][-1]
    expected = kpis.costs.value - np.mean(PRICE) * (d_th / np.mean(COP) + d_el)
    assert kpis.corrected_costs.value == pytest.approx(expected, rel=1e-12)
    assert kpis.corrected_costs_rel.value == pytest.approx(
        expected / kpis.energy_flex.value, rel=1e-12)


@pytest.mark.parametrize("direction", ["positive", "negative"])
def test_corrected_costs_without_electrical_storage_unchanged(direction):
    """Without stored electrical energy, the result equals the formula of 0.4.1 exactly."""
    data_th = _kpis(with_el_storage=False)
    kpis = getattr(data_th, f"kpis_{direction[:3]}")
    shadow = SHADOW[direction]
    d_th = E_TH[shadow][-1] - E_TH["base"][-1]
    old_formula = kpis.costs.value - d_th * np.mean(PRICE) / np.mean(COP)
    assert kpis.corrected_costs.value == old_formula

    # the stored electrical energy changes the corrected costs only
    kpis_el = getattr(_kpis(with_el_storage=True), f"kpis_{direction[:3]}")
    assert kpis_el.costs.value == kpis.costs.value
    assert kpis_el.energy_flex.value == kpis.energy_flex.value
    assert kpis_el.corrected_costs.value != kpis.corrected_costs.value


def _indicator_config(**correct_costs) -> FlexibilityIndicatorModuleConfig:
    return FlexibilityIndicatorModuleConfig(
        module_id="indicator", type="agentlib_flexquant.flexibility_indicator",
        _agent_id="indicator", correct_costs=correct_costs,
    )


EL_ALIASES = {glbs.STORED_EL_ENERGY_ALIAS_BASE, glbs.STORED_EL_ENERGY_ALIAS_NEG,
              glbs.STORED_EL_ENERGY_ALIAS_POS}


def test_indicator_waits_for_electrical_storage():
    """With the variable set, the indicator has the inputs and waits for them."""
    config = _indicator_config(enable_energy_costs_correction=True,
                               stored_energy_variable="E_th",
                               stored_electrical_energy_variable="E_bat")
    assert EL_ALIASES <= {v.name for v in config.inputs}
    handler = CallBackHandler(config=config)
    handler.initialize_callback_variables(data=FlexibilityData(**PARAMS), config=config)
    assert EL_ALIASES <= set(handler.necessary_callback_variables)


@pytest.mark.parametrize("correct_costs", [
    dict(enable_energy_costs_correction=True, stored_energy_variable="E_th"),
    dict(enable_energy_costs_correction=False, stored_electrical_energy_variable="E_bat"),
])
def test_no_electrical_storage_inputs_unless_used(correct_costs):
    """Without the variable or without the correction, inputs stay as in 0.4.1."""
    config = _indicator_config(**correct_costs)
    assert not EL_ALIASES & {v.name for v in config.inputs}
    handler = CallBackHandler(config=config)
    handler.initialize_callback_variables(data=FlexibilityData(**PARAMS), config=config)
    assert not EL_ALIASES & set(handler.necessary_callback_variables)
