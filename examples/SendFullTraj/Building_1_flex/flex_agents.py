import casadi as ca
import pandas as pd
import logging
from typing import List
from pydantic import model_validator, PrivateAttr
from agentlib_mpc.models.casadi_model import (
    CasadiModel,
    CasadiInput,
    CasadiState,
    CasadiParameter,
    CasadiOutput,
    CasadiModelConfig,
)
from flexmod.models.utils.model_parameters import get_model_parameters
import flexmod.modifier as modi
from flexmod.data_structures.modifier import BaseComponentModifier

logger = logging.getLogger(__name__)
MODIFIER_REGISTRY = {
    "generation": modi.GenerationModifier,
    "distribution": modi.DistributionModifier,
    "transfer": modi.TransferModifier,
    "building": modi.BuildingModifier,
    "electrical": modi.ElectricalModifier,
    "user": modi.boundary_conditions.UserModifier,
    "weather": modi.boundary_conditions.WeatherModifier,
    "price": modi.boundary_conditions.PriceModifier,
}


def load_modifier(data: dict) -> BaseComponentModifier:
    modifier_cls = MODIFIER_REGISTRY[data["modifier_type"]]
    return modifier_cls(**data)


class BaselineMPCModelConfig(CasadiModelConfig):
    bes_parameters_path: str
    bes_parameters: dict = {}
    prediction_horizon: int
    time_step: int
    inputs: List[CasadiInput] = [
        CasadiInput(
            name="_P_external",
            value=0,
            unit="W",
            type="pd.Series",
            description="External power profile to be provided",
        ),
        CasadiInput(
            name="in_provision",
            value=False,
            unit="-",
            type="bool",
            description="Flag signaling if the flexibility is in provision",
        ),
        CasadiInput(
            name="rel_start",
            value=0,
            unit="s",
            type="int",
            description="relative start time of the flexibility event",
        ),
        CasadiInput(
            name="rel_end",
            value=0,
            unit="s",
            type="int",
            description="relative end time of the flexibility event",
        ),
    ]
    states: List[CasadiState] = []
    parameters: List[CasadiParameter] = [
        CasadiParameter(
            name="profile_deviation_weight",
            value=0,
            unit="-",
            description="Weight of soft constraint for deviation from accepted flexible profile",
        )
    ]
    outputs: List[CasadiOutput] = []
    mpc_modifiers: List[dict]
    _modifier_dict = PrivateAttr(default={})

    @model_validator(mode="before")
    @classmethod
    def load_parameters_and_configure(cls, data):
        if not isinstance(data, dict):
            return data
        bes_parameters = data.get("bes_parameters", {})
        bes_parameters_path = data.get("bes_parameters_path")
        modifiers = data.get("mpc_modifiers", [])
        if not bes_parameters and bes_parameters_path:
            bes_parameters = get_model_parameters(bes_parameters_path)
            data["bes_parameters"] = bes_parameters
        if "user_config" not in data:
            data["user_config"] = {}

        def merge_generated_into_config(field_name, generated_objects):
            """
            Merges generated objects into the model (not module) user_config.
            This is needed to for the model validator include_default_model_variables
            in AgentLib, which merges default and user given variables. However,
            defaults for this model are empty (e.g. parameters = []) and user_config
            does not contain information about the optimization variables, causing the
            loop to break early and returning empty lists for the optimization
            variables. To bypass this, the user_config is filled here, with the
            optimization variables given by the modifiers.
            """
            user_defined_list = data["user_config"].get(field_name, [])
            user_defined_names = {item["name"] for item in user_defined_list}
            generated_dicts = [obj.dict() for obj in generated_objects]
            items_to_add = [
                d for d in generated_dicts if d["name"] not in user_defined_names
            ]
            data["user_config"][field_name] = items_to_add + user_defined_list

        all_inputs = []
        all_states = []
        all_parameters = []
        all_outputs = []
        for modifier in modifiers:
            mpc_modi = load_modifier(modifier).get_mpc_model_modifier()
            all_inputs.extend(mpc_modi.get_inputs(bes_parameters))
            all_states.extend(mpc_modi.get_states(bes_parameters))
            all_parameters.extend(mpc_modi.get_parameters(bes_parameters))
            all_outputs.extend(mpc_modi.get_outputs(bes_parameters))
        merge_generated_into_config("inputs", all_inputs)
        merge_generated_into_config("states", all_states)
        merge_generated_into_config("parameters", all_parameters)
        merge_generated_into_config("outputs", all_outputs)
        if "inputs" not in data or not data["inputs"]:
            data["inputs"] = all_inputs
        if "states" not in data or not data["states"]:
            data["states"] = all_states
        if "parameters" not in data or not data["parameters"]:
            data["parameters"] = all_parameters
        if "outputs" not in data or not data["outputs"]:
            data["outputs"] = all_outputs
        return data

    @model_validator(mode="after")
    def populate_modifier_dict(self):
        """Set the private _modifier_dict attribute after validation."""
        self._modifier_dict = {
            modifier["modifier_type"]: load_modifier(modifier)
            for modifier in self.mpc_modifiers
        }
        return self


class BaselineMPCModel(CasadiModel):
    config: BaselineMPCModelConfig
    building: modi.BuildingMPCModelModifier
    transfer: modi.TransferMPCModelModifier
    distribution: modi.DistributionMPCModelModifier
    generation: modi.GenerationMPCModelModifier
    electrical: modi.ElectricalMPCModelModifier

    def setup_system(self):
        self.building = self.config._modifier_dict["building"].get_mpc_model_modifier()
        self.transfer = self.config._modifier_dict["transfer"].get_mpc_model_modifier()
        self.distribution = self.config._modifier_dict[
            "distribution"
        ].get_mpc_model_modifier()
        self.generation = self.config._modifier_dict[
            "generation"
        ].get_mpc_model_modifier()
        self.electrical = self.config._modifier_dict[
            "electrical"
        ].get_mpc_model_modifier()
        bes_parameters = self.config.bes_parameters
        m_dot_transfer_nominal = self.m_dot_transfer_nominal
        m_dot_generation_const = self.m_dot_generation
        T_ret_transfer = self.transfer.T_return(self)
        T_sup_generation = self.generation.T_supply(self)
        T_ret_generation = self.distribution.T_return(self, T_ret_transfer)
        m_dot_transfer = self.transfer.m_dot(self, m_dot_transfer_nominal)
        m_dot_generation = self.distribution.m_dot(
            self, m_dot_generation_const, m_dot_transfer
        )
        T_sup_transfer = self.distribution.T_supply(self, T_sup_generation)
        self.T_Air.ode, T_Rad, E_Zone = self.building.eq(
            self, bes_parameters, self.Q_dot_transfer_rad, self.Q_dot_transfer_con
        )
        self.Q_dot_transfer_rad.alg, self.Q_dot_transfer_con.alg, E_transfer = (
            self.transfer.eq(self, T_sup_transfer, T_Rad, m_dot_transfer)
        )
        self.T_sup_generation.alg, P_el_generation = self.generation.eq(
            self, T_ret_generation, m_dot_generation
        )
        E_TES = self.distribution.eq(
            self, T_sup_generation, T_ret_transfer, m_dot_generation, m_dot_transfer
        )
        E_bat, P_bat_charge, P_bat_discharge, P_pv, P_load = self.electrical.eq(self)
        self.E_stored.alg = E_Zone + E_TES + E_transfer + E_bat
        self.P_el.alg = P_el_generation + P_bat_charge - P_bat_discharge - P_pv + P_load
        self.constraints = [
            *self.building.constraints(self),
            *self.transfer.constraints(self),
            *self.distribution.constraints(self),
            *self.generation.constraints(self),
            *self.electrical.constraints(self),
        ]
        only_gen = not (self.electrical.parent.use_bat or self.electrical.parent.use_pv)
        objective = (
            self.building.objective(self)
            + self.transfer.objective(self)
            + self.distribution.objective(self)
            + self.generation.objective(self, only_gen=only_gen)
            + self.electrical.objective(self)
        ) / self.config.prediction_horizon
        obj_std = objective
        return ca.if_else(
            self.in_provision.sym,
            ca.if_else(
                self.time < self.rel_start.sym,
                obj_std,
                ca.if_else(
                    self.time >= self.rel_end.sym,
                    obj_std,
                    sum(
                        [
                            self.profile_deviation_weight
                            * (self.P_el - self._P_external) ** 2
                        ]
                    ),
                ),
            ),
            obj_std,
        )


logger = logging.getLogger(__name__)
MODIFIER_REGISTRY = {
    "generation": modi.GenerationModifier,
    "distribution": modi.DistributionModifier,
    "transfer": modi.TransferModifier,
    "building": modi.BuildingModifier,
    "electrical": modi.ElectricalModifier,
    "user": modi.boundary_conditions.UserModifier,
    "weather": modi.boundary_conditions.WeatherModifier,
    "price": modi.boundary_conditions.PriceModifier,
}


def load_modifier(data: dict) -> BaseComponentModifier:
    modifier_cls = MODIFIER_REGISTRY[data["modifier_type"]]
    return modifier_cls(**data)


class PosFlexModelConfig(CasadiModelConfig):
    bes_parameters_path: str
    bes_parameters: dict = {}
    prediction_horizon: int
    time_step: int
    inputs: List[CasadiInput] = [
        CasadiInput(
            name="valve_opening_full",
            value=None,
            unit="Not defined",
            type="pd.Series",
            description="full control trajectory output of baseline mpc",
        ),
        CasadiInput(
            name="nSet_HP_full",
            value=None,
            unit="Not defined",
            type="pd.Series",
            description="full control trajectory output of baseline mpc",
        ),
        CasadiInput(
            name="in_provision",
            value=False,
            unit="Not defined",
            type="bool",
            description="Flag indicating whether flexibility should be provisioned",
        ),
    ]
    states: List[CasadiState] = []
    parameters: List[CasadiParameter] = [
        CasadiParameter(
            name="prep_time",
            value=3600,
            unit="s",
            description="Preparation time before switching objective",
        ),
        CasadiParameter(
            name="flex_event_duration",
            value=14400,
            unit="s",
            description="Duration of the flexibility event",
        ),
        CasadiParameter(
            name="market_time",
            value=900,
            unit="s",
            description="Market time associated with the objective switch",
        ),
        CasadiParameter(
            name="s_P_pos", value=10, unit="Not defined", description="Not defined"
        ),
        CasadiParameter(
            name="s_P_neg", value=1, unit="Not defined", description="Not defined"
        ),
        CasadiParameter(
            name="s_T", value=250, unit="Not defined", description="Not defined"
        ),
    ]
    outputs: List[CasadiOutput] = []
    mpc_modifiers: List[dict]
    _modifier_dict = PrivateAttr(default={})

    @model_validator(mode="before")
    @classmethod
    def load_parameters_and_configure(cls, data):
        if not isinstance(data, dict):
            return data
        bes_parameters = data.get("bes_parameters", {})
        bes_parameters_path = data.get("bes_parameters_path")
        modifiers = data.get("mpc_modifiers", [])
        if not bes_parameters and bes_parameters_path:
            bes_parameters = get_model_parameters(bes_parameters_path)
            data["bes_parameters"] = bes_parameters
        if "user_config" not in data:
            data["user_config"] = {}

        def merge_generated_into_config(field_name, generated_objects):
            """
            Merges generated objects into the model (not module) user_config.
            This is needed to for the model validator include_default_model_variables
            in AgentLib, which merges default and user given variables. However,
            defaults for this model are empty (e.g. parameters = []) and user_config
            does not contain information about the optimization variables, causing the
            loop to break early and returning empty lists for the optimization
            variables. To bypass this, the user_config is filled here, with the
            optimization variables given by the modifiers.
            """
            user_defined_list = data["user_config"].get(field_name, [])
            user_defined_names = {item["name"] for item in user_defined_list}
            generated_dicts = [obj.dict() for obj in generated_objects]
            items_to_add = [
                d for d in generated_dicts if d["name"] not in user_defined_names
            ]
            data["user_config"][field_name] = items_to_add + user_defined_list

        all_inputs = []
        all_states = []
        all_parameters = []
        all_outputs = []
        for modifier in modifiers:
            mpc_modi = load_modifier(modifier).get_mpc_model_modifier()
            all_inputs.extend(mpc_modi.get_inputs(bes_parameters))
            all_states.extend(mpc_modi.get_states(bes_parameters))
            all_parameters.extend(mpc_modi.get_parameters(bes_parameters))
            all_outputs.extend(mpc_modi.get_outputs(bes_parameters))
        merge_generated_into_config("inputs", all_inputs)
        merge_generated_into_config("states", all_states)
        merge_generated_into_config("parameters", all_parameters)
        merge_generated_into_config("outputs", all_outputs)
        if "inputs" not in data or not data["inputs"]:
            data["inputs"] = all_inputs
        if "states" not in data or not data["states"]:
            data["states"] = all_states
        if "parameters" not in data or not data["parameters"]:
            data["parameters"] = all_parameters
        if "outputs" not in data or not data["outputs"]:
            data["outputs"] = all_outputs
        return data

    @model_validator(mode="after")
    def populate_modifier_dict(self):
        """Set the private _modifier_dict attribute after validation."""
        self._modifier_dict = {
            modifier["modifier_type"]: load_modifier(modifier)
            for modifier in self.mpc_modifiers
        }
        return self


class PosFlexModel(CasadiModel):
    config: PosFlexModelConfig
    building: modi.BuildingMPCModelModifier
    transfer: modi.TransferMPCModelModifier
    distribution: modi.DistributionMPCModelModifier
    generation: modi.GenerationMPCModelModifier
    electrical: modi.ElectricalMPCModelModifier

    def setup_system(self):
        nSet_HP_lower = ca.if_else(
            self.time < self.market_time.sym, self.nSet_HP_full.sym, self.nSet_HP.lb
        )
        nSet_HP_upper = ca.if_else(
            self.time < self.market_time.sym, self.nSet_HP_full.sym, self.nSet_HP.ub
        )
        valve_opening_lower = ca.if_else(
            self.time < self.market_time.sym,
            self.valve_opening_full.sym,
            self.valve_opening.lb,
        )
        valve_opening_upper = ca.if_else(
            self.time < self.market_time.sym,
            self.valve_opening_full.sym,
            self.valve_opening.ub,
        )
        self.building = self.config._modifier_dict["building"].get_mpc_model_modifier()
        self.transfer = self.config._modifier_dict["transfer"].get_mpc_model_modifier()
        self.distribution = self.config._modifier_dict[
            "distribution"
        ].get_mpc_model_modifier()
        self.generation = self.config._modifier_dict[
            "generation"
        ].get_mpc_model_modifier()
        self.electrical = self.config._modifier_dict[
            "electrical"
        ].get_mpc_model_modifier()
        bes_parameters = self.config.bes_parameters
        m_dot_transfer_nominal = self.m_dot_transfer_nominal
        m_dot_generation_const = self.m_dot_generation
        T_ret_transfer = self.transfer.T_return(self)
        T_sup_generation = self.generation.T_supply(self)
        T_ret_generation = self.distribution.T_return(self, T_ret_transfer)
        m_dot_transfer = self.transfer.m_dot(self, m_dot_transfer_nominal)
        m_dot_generation = self.distribution.m_dot(
            self, m_dot_generation_const, m_dot_transfer
        )
        T_sup_transfer = self.distribution.T_supply(self, T_sup_generation)
        self.T_Air.ode, T_Rad, E_Zone = self.building.eq(
            self, bes_parameters, self.Q_dot_transfer_rad, self.Q_dot_transfer_con
        )
        self.Q_dot_transfer_rad.alg, self.Q_dot_transfer_con.alg, E_transfer = (
            self.transfer.eq(self, T_sup_transfer, T_Rad, m_dot_transfer)
        )
        self.T_sup_generation.alg, P_el_generation = self.generation.eq(
            self, T_ret_generation, m_dot_generation
        )
        E_TES = self.distribution.eq(
            self, T_sup_generation, T_ret_transfer, m_dot_generation, m_dot_transfer
        )
        E_bat, P_bat_charge, P_bat_discharge, P_pv, P_load = self.electrical.eq(self)
        self.E_stored.alg = E_Zone + E_TES + E_transfer + E_bat
        self.P_el.alg = P_el_generation + P_bat_charge - P_bat_discharge - P_pv + P_load
        self.constraints = [
            *self.building.constraints(self),
            *self.transfer.constraints(self),
            *self.distribution.constraints(self),
            *self.generation.constraints(self),
            *self.electrical.constraints(self),
            (valve_opening_lower, self.valve_opening, valve_opening_upper),
            (nSet_HP_lower, self.nSet_HP, nSet_HP_upper),
        ]
        only_gen = not (self.electrical.parent.use_bat or self.electrical.parent.use_pv)
        objective = (
            self.building.objective(self)
            + self.transfer.objective(self)
            + self.distribution.objective(self)
            + self.generation.objective(self, only_gen=only_gen)
            + self.electrical.objective(self)
        ) / self.config.prediction_horizon
        obj_std = objective
        obj_flex = (
            sum([self.s_T * self.T_slack**2, self.s_P_pos * self.P_el])
            / self.config.prediction_horizon
            + 0
        )
        return ca.if_else(
            self.time < self.prep_time.sym + self.market_time.sym,
            obj_std,
            ca.if_else(
                self.time
                < self.prep_time.sym
                + self.flex_event_duration.sym
                + self.market_time.sym,
                obj_flex,
                obj_std,
            ),
        )


logger = logging.getLogger(__name__)
MODIFIER_REGISTRY = {
    "generation": modi.GenerationModifier,
    "distribution": modi.DistributionModifier,
    "transfer": modi.TransferModifier,
    "building": modi.BuildingModifier,
    "electrical": modi.ElectricalModifier,
    "user": modi.boundary_conditions.UserModifier,
    "weather": modi.boundary_conditions.WeatherModifier,
    "price": modi.boundary_conditions.PriceModifier,
}


def load_modifier(data: dict) -> BaseComponentModifier:
    modifier_cls = MODIFIER_REGISTRY[data["modifier_type"]]
    return modifier_cls(**data)


class NegFlexModelConfig(CasadiModelConfig):
    bes_parameters_path: str
    bes_parameters: dict = {}
    prediction_horizon: int
    time_step: int
    inputs: List[CasadiInput] = [
        CasadiInput(
            name="valve_opening_full",
            value=None,
            unit="Not defined",
            type="pd.Series",
            description="full control trajectory output of baseline mpc",
        ),
        CasadiInput(
            name="nSet_HP_full",
            value=None,
            unit="Not defined",
            type="pd.Series",
            description="full control trajectory output of baseline mpc",
        ),
        CasadiInput(
            name="in_provision",
            value=False,
            unit="Not defined",
            type="bool",
            description="Flag indicating whether flexibility should be provisioned",
        ),
    ]
    states: List[CasadiState] = []
    parameters: List[CasadiParameter] = [
        CasadiParameter(
            name="prep_time",
            value=3600,
            unit="s",
            description="Preparation time before switching objective",
        ),
        CasadiParameter(
            name="flex_event_duration",
            value=14400,
            unit="s",
            description="Duration of the flexibility event",
        ),
        CasadiParameter(
            name="market_time",
            value=900,
            unit="s",
            description="Market time associated with the objective switch",
        ),
        CasadiParameter(
            name="s_P_pos", value=10, unit="Not defined", description="Not defined"
        ),
        CasadiParameter(
            name="s_P_neg", value=1, unit="Not defined", description="Not defined"
        ),
        CasadiParameter(
            name="s_T", value=250, unit="Not defined", description="Not defined"
        ),
    ]
    outputs: List[CasadiOutput] = []
    mpc_modifiers: List[dict]
    _modifier_dict = PrivateAttr(default={})

    @model_validator(mode="before")
    @classmethod
    def load_parameters_and_configure(cls, data):
        if not isinstance(data, dict):
            return data
        bes_parameters = data.get("bes_parameters", {})
        bes_parameters_path = data.get("bes_parameters_path")
        modifiers = data.get("mpc_modifiers", [])
        if not bes_parameters and bes_parameters_path:
            bes_parameters = get_model_parameters(bes_parameters_path)
            data["bes_parameters"] = bes_parameters
        if "user_config" not in data:
            data["user_config"] = {}

        def merge_generated_into_config(field_name, generated_objects):
            """
            Merges generated objects into the model (not module) user_config.
            This is needed to for the model validator include_default_model_variables
            in AgentLib, which merges default and user given variables. However,
            defaults for this model are empty (e.g. parameters = []) and user_config
            does not contain information about the optimization variables, causing the
            loop to break early and returning empty lists for the optimization
            variables. To bypass this, the user_config is filled here, with the
            optimization variables given by the modifiers.
            """
            user_defined_list = data["user_config"].get(field_name, [])
            user_defined_names = {item["name"] for item in user_defined_list}
            generated_dicts = [obj.dict() for obj in generated_objects]
            items_to_add = [
                d for d in generated_dicts if d["name"] not in user_defined_names
            ]
            data["user_config"][field_name] = items_to_add + user_defined_list

        all_inputs = []
        all_states = []
        all_parameters = []
        all_outputs = []
        for modifier in modifiers:
            mpc_modi = load_modifier(modifier).get_mpc_model_modifier()
            all_inputs.extend(mpc_modi.get_inputs(bes_parameters))
            all_states.extend(mpc_modi.get_states(bes_parameters))
            all_parameters.extend(mpc_modi.get_parameters(bes_parameters))
            all_outputs.extend(mpc_modi.get_outputs(bes_parameters))
        merge_generated_into_config("inputs", all_inputs)
        merge_generated_into_config("states", all_states)
        merge_generated_into_config("parameters", all_parameters)
        merge_generated_into_config("outputs", all_outputs)
        if "inputs" not in data or not data["inputs"]:
            data["inputs"] = all_inputs
        if "states" not in data or not data["states"]:
            data["states"] = all_states
        if "parameters" not in data or not data["parameters"]:
            data["parameters"] = all_parameters
        if "outputs" not in data or not data["outputs"]:
            data["outputs"] = all_outputs
        return data

    @model_validator(mode="after")
    def populate_modifier_dict(self):
        """Set the private _modifier_dict attribute after validation."""
        self._modifier_dict = {
            modifier["modifier_type"]: load_modifier(modifier)
            for modifier in self.mpc_modifiers
        }
        return self


class NegFlexModel(CasadiModel):
    config: NegFlexModelConfig
    building: modi.BuildingMPCModelModifier
    transfer: modi.TransferMPCModelModifier
    distribution: modi.DistributionMPCModelModifier
    generation: modi.GenerationMPCModelModifier
    electrical: modi.ElectricalMPCModelModifier

    def setup_system(self):
        nSet_HP_lower = ca.if_else(
            self.time < self.market_time.sym, self.nSet_HP_full.sym, self.nSet_HP.lb
        )
        nSet_HP_upper = ca.if_else(
            self.time < self.market_time.sym, self.nSet_HP_full.sym, self.nSet_HP.ub
        )
        valve_opening_lower = ca.if_else(
            self.time < self.market_time.sym,
            self.valve_opening_full.sym,
            self.valve_opening.lb,
        )
        valve_opening_upper = ca.if_else(
            self.time < self.market_time.sym,
            self.valve_opening_full.sym,
            self.valve_opening.ub,
        )
        self.building = self.config._modifier_dict["building"].get_mpc_model_modifier()
        self.transfer = self.config._modifier_dict["transfer"].get_mpc_model_modifier()
        self.distribution = self.config._modifier_dict[
            "distribution"
        ].get_mpc_model_modifier()
        self.generation = self.config._modifier_dict[
            "generation"
        ].get_mpc_model_modifier()
        self.electrical = self.config._modifier_dict[
            "electrical"
        ].get_mpc_model_modifier()
        bes_parameters = self.config.bes_parameters
        m_dot_transfer_nominal = self.m_dot_transfer_nominal
        m_dot_generation_const = self.m_dot_generation
        T_ret_transfer = self.transfer.T_return(self)
        T_sup_generation = self.generation.T_supply(self)
        T_ret_generation = self.distribution.T_return(self, T_ret_transfer)
        m_dot_transfer = self.transfer.m_dot(self, m_dot_transfer_nominal)
        m_dot_generation = self.distribution.m_dot(
            self, m_dot_generation_const, m_dot_transfer
        )
        T_sup_transfer = self.distribution.T_supply(self, T_sup_generation)
        self.T_Air.ode, T_Rad, E_Zone = self.building.eq(
            self, bes_parameters, self.Q_dot_transfer_rad, self.Q_dot_transfer_con
        )
        self.Q_dot_transfer_rad.alg, self.Q_dot_transfer_con.alg, E_transfer = (
            self.transfer.eq(self, T_sup_transfer, T_Rad, m_dot_transfer)
        )
        self.T_sup_generation.alg, P_el_generation = self.generation.eq(
            self, T_ret_generation, m_dot_generation
        )
        E_TES = self.distribution.eq(
            self, T_sup_generation, T_ret_transfer, m_dot_generation, m_dot_transfer
        )
        E_bat, P_bat_charge, P_bat_discharge, P_pv, P_load = self.electrical.eq(self)
        self.E_stored.alg = E_Zone + E_TES + E_transfer + E_bat
        self.P_el.alg = P_el_generation + P_bat_charge - P_bat_discharge - P_pv + P_load
        self.constraints = [
            *self.building.constraints(self),
            *self.transfer.constraints(self),
            *self.distribution.constraints(self),
            *self.generation.constraints(self),
            *self.electrical.constraints(self),
            (valve_opening_lower, self.valve_opening, valve_opening_upper),
            (nSet_HP_lower, self.nSet_HP, nSet_HP_upper),
        ]
        only_gen = not (self.electrical.parent.use_bat or self.electrical.parent.use_pv)
        objective = (
            self.building.objective(self)
            + self.transfer.objective(self)
            + self.distribution.objective(self)
            + self.generation.objective(self, only_gen=only_gen)
            + self.electrical.objective(self)
        ) / self.config.prediction_horizon
        obj_std = objective
        obj_flex = (
            sum([self.s_T * self.T_slack**2, -self.s_P_neg * self.P_el])
            / self.config.prediction_horizon
            + 0
            + 0
        )
        return ca.if_else(
            self.time < self.prep_time.sym + self.market_time.sym,
            obj_std,
            ca.if_else(
                self.time
                < self.prep_time.sym
                + self.flex_event_duration.sym
                + self.market_time.sym,
                obj_flex,
                obj_std,
            ),
        )
