from agentlib_mpc.models.casadi_model import CasadiModel, CasadiInput, CasadiState, CasadiParameter, CasadiOutput, CasadiModelConfig
import casadi as ca
from typing import List
import numpy as np
import agentlib
import pandas as pd

class BaselineMPCModelConfig(CasadiModelConfig):
    inputs: List[CasadiInput] = [CasadiInput(name='mDot', value=0.0225, unit='K', description='Air mass flow into zone'), CasadiInput(name='load', value=150, unit='W', description='Heat load into zone'), CasadiInput(name='T_in', value=280.15, unit='K', description='Inflow air temperature'), CasadiInput(name='T_upper', value=294.15, unit='K', description='Upper boundary (soft) for T.'), CasadiInput(name='T_lower', value=292.15, unit='K', description='Upper boundary (soft) for T.'), CasadiInput(name='Time', value=0, unit='s', description='time trajectory'), CasadiInput(name='_P_external', value=0, unit='W', description='External power profile to be provised'), CasadiInput(name='in_provision', value=False, unit='-', description='Flag signaling if the flexibility is in provision'), CasadiInput(name='rel_start', value=0, unit='s', description='relative start time of the flexibility event'), CasadiInput(name='rel_end', value=0, unit='s', description='relative end time of the flexibility event')]
    states: List[CasadiState] = [CasadiState(name='t_sim', value=0, unit='sec', description='simulation time'), CasadiState(name='T', value=293.15, unit='K', description='Temperature of zone'), CasadiState(name='T_slack', value=0, unit='K', description='Slack variable of temperature of zone')]
    parameters: List[CasadiParameter] = [CasadiParameter(name='cp', value=1000, unit='J/kg*K', description='thermal capacity of the air'), CasadiParameter(name='C', value=100000, unit='J/K', description='thermal capacity of zone'), CasadiParameter(name='s_T', value=1, unit='-', description='Weight for T in constraint function'), CasadiParameter(name='r_mDot', value=1, unit='-', description='Weight for mDot in objective function'), CasadiParameter(name='s_P', value=0, unit='-', description='Weight for P in objective function')]
    outputs: List[CasadiOutput] = [CasadiOutput(name='T_out', unit='K', description='Temperature of zone'), CasadiOutput(name='P_el', unit='W', description='The power input to the system'), CasadiOutput(name='mDot_full', unit='W', type='pd.Series', value=pd.Series([0]), description='full control output')]

class BaselineMPC(CasadiModel):
    config: BaselineMPCModelConfig

    def setup_system(self):
        self.T.ode = self.cp * self.mDot / self.C * (self.T_in - self.T) + self.load / self.C
        self.P_el.alg = self.cp * self.mDot * (self.T_in - self.T)
        self.T_out.alg = self.T
        self.constraints = [(self.T_lower, self.T + self.T_slack, self.T_upper)]
        objective = sum([self.r_mDot * self.mDot, self.s_T * self.T_slack ** 2])
        self.mDot_full.alg = self.mDot
        obj_std = sum([self.r_mDot * self.mDot, self.s_T * self.T_slack ** 2])
        return obj_std + ca.if_else(self.in_provision.sym, ca.if_else(self.Time.sym < self.rel_start.sym, 0, ca.if_else(self.Time.sym > self.rel_end.sym, 0, sum([1.0 * (self.P_el - self._P_external) ** 2]))), 0)

class FlexShadowMPCConfig(CasadiModelConfig):
    inputs: List[CasadiInput] = [CasadiInput(name='mDot', value=0.0225, unit='K', description='Air mass flow into zone'), CasadiInput(name='load', value=150, unit='W', description='Heat load into zone'), CasadiInput(name='T_in', value=280.15, unit='K', description='Inflow air temperature'), CasadiInput(name='T_upper', value=294.15, unit='K', description='Upper boundary (soft) for T.'), CasadiInput(name='T_lower', value=292.15, unit='K', description='Upper boundary (soft) for T.'), CasadiInput(name='Time', value=0, unit='s', description='time trajectory'), CasadiInput(name='_mDot', unit='W', type='pd.Series', value=pd.Series([0])), CasadiInput(name='in_provision', unit='-', value=False)]
    states: List[CasadiState] = [CasadiState(name='t_sim', value=0, unit='sec', description='simulation time'), CasadiState(name='T', value=293.15, unit='K', description='Temperature of zone'), CasadiState(name='T_slack', value=0, unit='K', description='Slack variable of temperature of zone')]
    parameters: List[CasadiParameter] = [CasadiParameter(name='cp', value=1000, unit='J/kg*K', description='thermal capacity of the air'), CasadiParameter(name='C', value=100000, unit='J/K', description='thermal capacity of zone'), CasadiParameter(name='s_T', value=1, unit='-', description='Weight for T in constraint function'), CasadiParameter(name='r_mDot', value=1, unit='-', description='Weight for mDot in objective function'), CasadiParameter(name='prep_time', value=0, unit='s', description='time to switch objective'), CasadiParameter(name='flex_event_duration', value=0, unit='s', description='time to switch objective'), CasadiParameter(name='market_time', value=0, unit='s', description='time to switch objective'), CasadiParameter(name='s_P', value=0, unit='-', description='Weight for P in objective function')]
    outputs: List[CasadiOutput] = [CasadiOutput(name='T_out', unit='K', description='Temperature of zone'), CasadiOutput(name='P_el', unit='W', description='The power input to the system')]

class PosFlexModel(CasadiModel):
    config: FlexShadowMPCConfig

    def setup_system(self):
        self.T.ode = self.cp * self.mDot / self.C * (self.T_in - self.T) + self.load / self.C
        self.P_el.alg = self.cp * self.mDot * (self.T_in - self.T)
        self.T_out.alg = self.T
        mDot_neg = ca.if_else(self.Time.sym < self.market_time.sym, self._mDot.sym, self.mDot.lb)
        mDot_pos = ca.if_else(self.Time.sym < self.market_time.sym, self._mDot.sym, self.mDot.ub)
        self.constraints = [(self.T_lower, self.T + self.T_slack, self.T_upper), 
        (mDot_neg, self.mDot, mDot_pos)]
        objective = sum([self.r_mDot * self.mDot, self.s_T * self.T_slack ** 2])
        obj_flex = sum([self.s_T * self.T_slack ** 2, self.s_P * self.P_el])
        obj_std = sum([self.r_mDot * self.mDot, self.s_T * self.T_slack ** 2])
        return ca.if_else(self.Time.sym < self.prep_time.sym + self.market_time.sym, obj_std, ca.if_else(self.Time.sym < self.prep_time.sym + self.flex_event_duration.sym + self.market_time.sym, obj_flex, obj_std))

class NegFlexModel(CasadiModel):
    config: FlexShadowMPCConfig

    def setup_system(self):
        self.T.ode = self.cp * self.mDot / self.C * (self.T_in - self.T) + self.load / self.C
        self.P_el.alg = self.cp * self.mDot * (self.T_in - self.T)
        self.T_out.alg = self.T
        mDot_neg = ca.if_else(self.Time.sym < self.market_time.sym, self._mDot.sym, self.mDot.lb)
        mDot_pos = ca.if_else(self.Time.sym < self.market_time.sym, self._mDot.sym, self.mDot.ub)
        self.constraints = [(self.T_lower, self.T + self.T_slack, self.T_upper), 
        (mDot_neg, self.mDot, mDot_pos)]
        objective = sum([self.r_mDot * self.mDot, self.s_T * self.T_slack ** 2])
        obj_flex = sum([self.s_T * self.T_slack ** 2, -self.s_P * self.P_el])
        obj_std = sum([self.r_mDot * self.mDot, self.s_T * self.T_slack ** 2])
        return ca.if_else(self.Time.sym < self.prep_time.sym + self.market_time.sym, obj_std, ca.if_else(self.Time.sym < self.prep_time.sym + self.flex_event_duration.sym + self.market_time.sym, obj_flex, obj_std))