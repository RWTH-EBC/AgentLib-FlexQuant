import math
import logging
import json
from itertools import combinations
from typing import List
from utilities.parse_radiator_record import parse_rad_record, parse_modelica_record
from agentlib_mpc.models.casadi_model import (
    CasadiModel,
    CasadiInput,
    CasadiState,
    CasadiParameter,
    CasadiOutput,
    CasadiModelConfig,
)
from math import inf
import casadi as ca
from local.mpc.utils.calc_resistances import calc_resistances

logger = logging.getLogger(__name__)

# update record
with open(r"predictor/setup_disturbances.json", 'r') as f:
    setup = json.load(f)
path_to_mos = setup["path_zone_record"]
path_to_radiator = setup["path_radiator_record"]
tz_par = parse_modelica_record(path_to_mos)  #tz_par is dict containing input values for building model
rad_par = parse_rad_record(path_to_radiator)


class SimpleTestHallModelConfig(CasadiModelConfig):
    inputs: List[CasadiInput] = [
        # controls
        CasadiInput(name="T_flow_in_max", value=328.15, unit="K", description="flow temperature of radiator"),
        # disturbances
        CasadiInput(name="Q_RadSol", value=0, unit="W",
                    description="Radiative solar heat for all orientations"),
        CasadiInput(name="T_amb", value=273.15, unit="K",
                    description="Ambient temperature on the outside"),
        CasadiInput(name="T_preTemWin", value=294.15, unit="K",
                    description="Outdoor surface temperature of window"),
        CasadiInput(name="T_preTemWall", value=294.15, unit="K",
                    description="Outdoor surface temperature of wall"),
        CasadiInput(name="T_preTemRoof", value=294.15, unit="K",
                    description="Outdoor surface temperature of roof"),
        CasadiInput(name="schedule_human", value=0.1, unit="", description="Radiative solar heat"),
        CasadiInput(name="schedule_dev", value=0, unit="", description="Radiative solar heat"),
        CasadiInput(name="schedule_light", value=0, unit="", description="Radiative solar heat"),
        CasadiInput(name="r_pel", value=0.4, unit="-", description="Power Price"),

        # settings
        CasadiInput(name="T_upper", value=300, unit="K",
                    description="Upper boundary (soft) for T.", ),
        CasadiInput(name="T_lower", value=288, unit="K",
                    description="Lower boundary (soft) for T.", ),
    ]

    states: List[CasadiState] = [
        # differential
        CasadiState(name="T_Air", value=290.15, unit="K", description="Temperature of zone"),
        CasadiState(name="T_Roof", value=290.15, unit="K",
                    description="Temperature of roof in middle"),
        CasadiState(name="T_Floor", value=290.15, unit="K",
                    description="Temperature of floor in middle"),
        CasadiState(name="T_ExtWall", value=290.15, unit="K",
                    description="Outer wall temperature in middle"),
        CasadiState(name="T_IntWall", value=290.15, unit="K",
                    description="Inner wall temperature in middle"),
        # differential for radiator
        CasadiState(name="T_radiator_m_1", value=293.15, unit="K", description="radiator temperature layer i"),
        CasadiState(name="T_radiator_m_2", value=293.15, unit="K", description="radiator temperature layer i"),
        CasadiState(name="T_radiator_m_3", value=293.15, unit="K", description="radiator temperature layer i"),
        #debug
        CasadiState(name="T_preWin", value=280),
        CasadiState(name="T_preWall", value=280),
        CasadiState(name="T_preRoof", value=280),
        # slack variables
        CasadiState(name="T_slack", value=0, unit="K",
                    description="Slack variable for (soft) constraint of T."),
        CasadiState(name="Q_flow_slack1", value=0, unit="W",
                    description="Slack variable for abs of Q Flow"),
        CasadiState(name="Q_flow_slack2", value=0, unit="W",
                    description="Slack variable for abs of Q Flow")
    ]
    # corresponding to:
    # 1. parameter in mpc cost function: COP, time_step
    # 2. given values in record.mo for building model.
    parameters: List[CasadiParameter] = [
        CasadiParameter(name="time_step", value=900),
        CasadiParameter(name="T_soil", value=tz_par['TSoil'],
                        unit="K", description="Temperature of soil"),
        CasadiParameter(name="COP", value=5,
                        unit="-", description="COP of heat pump"),
        CasadiParameter(name="activityDegree", value=tz_par['activityDegree'],
                        unit="met", description="activity Degree of people in met"),
        CasadiParameter(name="specificPeople", value=tz_par['specificPeople'],
                        unit="1/m^2", description="people per squaremeter"),
        CasadiParameter(name="VAir", value=tz_par['VAir'], unit="m^3",
                        description="Air volume of thermal zone"),
        CasadiParameter(name="AZone", value=tz_par['AZone'],
                        unit="m^2", description="zone area"),
        CasadiParameter(name="ratioConvectiveHeatPeople", value=tz_par['ratioConvectiveHeatPeople'],
                        unit="-",
                        description="Ratio of convective heat from overall heat output for people"),
        CasadiParameter(name="internalGainsMachinesSpecific",
                        value=tz_par['internalGainsMachinesSpecific'],
                        unit="W", description="Heat Flux of machines"),
        CasadiParameter(name="ratioConvectiveHeatMachines",
                        value=tz_par['ratioConvectiveHeatMachines'],
                        unit="-",
                        description="Ratio of convective heat from overall heat output for machines"),
        CasadiParameter(name="lightingPowerSpecific", value=tz_par['lightingPowerSpecific'],
                        unit="W/m^2", description="Heat flux of lighting"),
        CasadiParameter(name="ratioConvectiveHeatLighting",
                        value=tz_par['ratioConvectiveHeatLighting'],
                        unit="-",
                        description="Ratio of convective heat from overall heat output for lights"),
        CasadiParameter(name="useConstantACHrate",
                        value=tz_par['useConstantACHrate'],
                        unit="-",
                        description="if using a constant infiltration rate is used"),
        CasadiParameter(name="baseACH",
                        value=tz_par['baseACH'],
                        unit="-",
                        description="base ACH rate for ventilation controller"),
        CasadiParameter(name="air_rho", value=1.2,
                        unit="kg/m**3", description="density of air"),  #mo中暂无
        CasadiParameter(name="air_cp", value=1006,
                        unit="J/kg*K", description="thermal capacity of air"),  #mo中暂无
        CasadiParameter(name="CRoof", value=tz_par['CRoof'],
                        unit="J/K", description="Heat capacities of roof"),
        CasadiParameter(name="CExt", value=tz_par['CExt'],
                        unit="J/K", description="Heat capacities of exterior walls"),
        CasadiParameter(name="CInt", value=tz_par['CInt'],
                        unit="J/K", description="Heat capacities of interior walls"),
        CasadiParameter(name="CFloor", value=tz_par['CFloor'],
                        unit="J/K", description="Heat capacities of floor"),

        CasadiParameter(name="hConRoofOut", value=tz_par['hConRoofOut'],
                        unit="W/(m^2*K)",
                        description="Roof's convective coefficient of heat transfer (outdoor)"),
        CasadiParameter(name="hConRoof", value=tz_par['hConRoof'],
                        unit="W/(m^2*K)",
                        description="Roof's convective coefficient of heat transfer (indoor)"),
        CasadiParameter(name="RRoof", value=tz_par['RRoof'],
                        unit="K/W", description="Resistances of roof, from inside to outside"),
        CasadiParameter(name="RRoofRem", value=tz_par['RRoofRem'], unit="K/W",
                        description="Resistance of remaining resistor between capacity n and outside"),
        CasadiParameter(name="hConExt", value=tz_par['hConExt'],
                        unit="W/(m^2*K)",
                        description="External walls convective coefficient of heat transfer (indoor)"),
        CasadiParameter(name="RExt", value=tz_par['RExt'],
                        unit="K/W",
                        description="Resistances of external walls, from inside to middle of wall"),
        CasadiParameter(name="hConWallOut", value=tz_par['hConWallOut'],
                        unit="W/(m^2*K)",
                        description="External walls convective coefficient of heat transfer (outdoor)"),
        CasadiParameter(name="RExtRem", value=tz_par['RExtRem'],
                        unit="K/W",
                        description="Resistances of external walls, from middle of wall to outside"),
        CasadiParameter(name="hConInt", value=tz_par['hConInt'],
                        unit="W/(m^2*K)",
                        description="Internal walls convective coefficient of heat transfer (indoor)"),
        CasadiParameter(name="RInt", value=tz_par['RInt'],
                        unit="K/W",
                        description="Resistances of internal walls, from inside to outside"),
        CasadiParameter(name="hConWin", value=tz_par['hConWin'],
                        unit="W/(m^2*K)",
                        description="Windows convective coefficient of heat transfer (indoor)"),
        CasadiParameter(name="hConWinOut", value=tz_par['hConWinOut'],
                        unit="W/(m^2*K)",
                        description="Windows convective coefficient of heat transfer (outdoor)"),
        CasadiParameter(name="RWin", value=tz_par['RWin'],
                        unit="K/W", description="Resistances of windows, from inside to outside"),
        CasadiParameter(name="hConFloor", value=tz_par['hConFloor'],
                        unit="W/(m^2*K)",
                        description="Floor convective coefficient of heat transfer (indoor)"),
        CasadiParameter(name="RFloor", value=tz_par['RFloor'],
                        unit="K/W", description="Resistances of floor, from inside to outside"),
        CasadiParameter(name="RFloorRem", value=tz_par['RFloorRem'], unit="K/W",
                        description="Resistance of remaining resistor between capacity n and outside"),
        CasadiParameter(name="hRad", value=tz_par['hRad'],
                        unit="W/(m^2*K)",
                        description="Coefficient of heat transfer for linearized radiation exchange between walls"),
        CasadiParameter(name="hRadRoof", value=tz_par['hRadRoof'],
                        unit="W/(m^2*K)",
                        description="Coefficient of heat transfer for linearized radiation for roof"),
        CasadiParameter(name="hRadWall", value=tz_par['hRadWall'],
                        unit="W/(m^2*K)",
                        description="Coefficient of heat transfer for linearized radiation for walls"),
        CasadiParameter(name="gWin", value=tz_par['gWin'],
                        unit="-", description="Total energy transmittance of windows"),
        CasadiParameter(name="ratioWinConRad", value=tz_par['ratioWinConRad'],
                        unit="-",
                        description="Ratio for windows between convective and radiation emission"),
        CasadiParameter(name="AExttot",
                        value=sum(tz_par['AExt']) if type(tz_par['AExt']) == list else tz_par[
                            'AExt'],
                        unit="m^2", description="total external walls area"),
        CasadiParameter(name="AInttot",
                        value=sum(tz_par['AInt']) if type(tz_par['AInt']) == list else tz_par[
                            'AInt'],
                        unit="m^2", description="total internal walls area"),
        CasadiParameter(name="AWintot",
                        value=sum(tz_par['AWin']) if type(tz_par['AWin']) == list else tz_par[
                            'AWin'],
                        unit="m^2", description="total window area"),
        CasadiParameter(name="AFloortot",
                        value=sum(tz_par['AFloor']) if type(tz_par['AFloor']) == list else tz_par[
                            'AFloor'],
                        unit="m^2", description="total floor area"),
        CasadiParameter(name="ARooftot",
                        value=sum(tz_par['ARoof']) if type(tz_par['ARoof']) == list else tz_par[
                            'ARoof'],
                        unit="m^2", description="total roof area"),
        CasadiParameter(name="ATransparent", value=sum(tz_par['ATransparent']) if type(
            tz_par['ATransparent']) == list else tz_par['ATransparent'],
                        unit="m^2", description="total transparent area"),

        # Parameters of radiator:
        CasadiParameter(name="N",value=3,unit="-", description="radiator layer"),
        CasadiParameter(name="m_flow", value=0.058, unit="kg/s", description="mass flow for radiator"),
        CasadiParameter(name="T_ref", value=289.15,
                        unit="K",
                        description="reference temperature"),
        CasadiParameter(name="NominalPower", value=rad_par['NominalPower'],
                        unit="W/m",
                        description="nominal power of radiator per m at nominal temp."),
        CasadiParameter(name="T_in_nom", value=rad_par['RT_nom'][0],
                        unit="K",
                        description="nominal temperatures Tin,Tout,Tair DIN-EN 442"),
        CasadiParameter(name="T_out_nom", value=rad_par['RT_nom'][1],
                        unit="K",
                        description="nominal temperatures Tin,Tout,Tair DIN-EN 442"),
        CasadiParameter(name="T_air_nom", value=rad_par['RT_nom'][2],
                        unit="K",
                        description="nominal temperatures Tin,Tout,Tair DIN-EN 442"),
        CasadiParameter(name="dT_V_nom", value=(rad_par['RT_nom'][0] - rad_par['RT_nom'][2]),
                        unit="K",
                        description="nominal flow temperatures DIN-EN 442"),
        CasadiParameter(name="dT_R_nom", value=(rad_par['RT_nom'][1] - rad_par['RT_nom'][2]),
                        unit="K",
                        description="nominal returen temperatures DIN-EN 442"),
        CasadiParameter(name="PressureDrop", value=rad_par['PressureDrop'],
                        unit="Pa",
                        description="PressureDrop"),
        CasadiParameter(name="n", value=rad_par['Exponent'],
                        unit="",
                        description="radiator exponent"),
        CasadiParameter(name="VolumeWater", value=rad_par['VolumeWater'],
                        unit="l/m",
                        description="water volume inside radiator per m"),
        CasadiParameter(name="MassSteel", value=rad_par['MassSteel'],
                        unit="kg/m",
                        description="material mass of radiator per m"),
        CasadiParameter(name="length", value=rad_par['length'],
                        unit="m",
                        description="length of radiator"),
        CasadiParameter(name="height", value=rad_par['height'],
                        unit="m",
                        description="height of radiator"),
        CasadiParameter(name="s_eff", value=rad_par['rad_fac'],
                        unit="-",
                        description="Radiative coefficient"),
        # material paras of raditar
        CasadiParameter(name="d_wall", value=0.025,
                        unit="m",
                        description="Thickness of radiator wall"),
        CasadiParameter(name="DensitySteel", value=rad_par['DensitySteel'],
                        unit="kg/m3",
                        description=""),
        CasadiParameter(name="CapacitySteel", value=rad_par['CapacitySteel'],
                        unit="J/(kg*K)",
                        description=""),
        CasadiParameter(name="LambdaSteel", value=rad_par['LambdaSteel'],
                        unit="W/(m*K)",
                        description="thermal conductivity of steel"),
        # physical parameters
        CasadiParameter(name="eps", value=0.95,
                        unit="",
                        description="emissivity"),
        CasadiParameter(name="sigma", value=5.6703744191844314E-08,
                        unit="W*m^−2*K^−4",
                        description="Stefan-Boltzmann constant"),
        CasadiParameter(name='cp_water', value=4184, unit="J/kg*K",
                        description="spezifische Wärmekapazität von Wasser "),
        CasadiParameter(name='rho_water', value=995.586, unit="kg/m3",
                        description="density of water"),

        # Below: parameter for cost function and optimizing functions
        # parameter for cost function
        CasadiParameter(name="s_T", value=0.5,
                        unit="-", description="Weight for T_slack"),
        CasadiParameter(name="s_Pel", value=0.5,
                        unit="-", description="Weight for P_el"),

        # β in building model: der den Anteil der gesamten Sonneneinstrahlung auf die Komponente angibt.
        CasadiParameter(name="fac_IG", value=1,
                        unit="-", description="factor for internal gains"),
        CasadiParameter(name="fac_IG_air", value=1,
                        unit="-", description="factor for internal gains"),
        CasadiParameter(name="fac_IG_floor", value=1,
                        unit="-", description="factor for internal gains"),
        CasadiParameter(name="fac_sol_air", value=1,
                        unit="-", description="factor for internal gains"),
        CasadiParameter(name="fac_sol_floor", value=1,
                        unit="-", description="factor for internal gains"),
    ]

    outputs: List[CasadiOutput] = [
        CasadiOutput(name="P_el_max_alg", unit='W', description="Output for controll variable"),
        CasadiOutput(name="T_rad_nf", unit='K', description="radiative temperature"),
        CasadiOutput(name="Q_flow_total_max", unit='W', description="Output for controll variable"),
        CasadiOutput(name="T_flow_out_NF", unit='K', description="Output temperature radiator")
    ]


class SimpleTestHall(CasadiModel):
    config: SimpleTestHallModelConfig

    def setup_system(self):
        # Get areas as python variable (not casadi)
        #todo: calculate automatically from thermal zone
        awintot = sum(tz_par['AWin']) if type(tz_par['AWin']) == list else tz_par['AWin']
        aexttot = sum(tz_par['AExt']) if type(tz_par['AExt']) == list else tz_par['AExt']
        ainttot = sum(tz_par['AInt']) if type(tz_par['AInt']) == list else tz_par['AInt']
        arooftot = sum(tz_par['ARoof']) if type(tz_par['ARoof']) == list else tz_par['ARoof']
        afloortot = sum(tz_par['AFloor']) if type(tz_par['AFloor']) == list else tz_par['AFloor']
        tz_par['AWintot'] = awintot
        tz_par['AInttot'] = ainttot
        tz_par['AExttot'] = aexttot
        tz_par['ARooftot'] = arooftot
        tz_par['AFloortot'] = afloortot

        awinmean = awintot / len(tz_par['AWin']) if isinstance(tz_par['AWin'], list) else awintot
        aextmean = aexttot / len(tz_par['AExt']) if isinstance(tz_par['AExt'], list) else aexttot
        aintmean = ainttot / len(tz_par['AInt']) if isinstance(tz_par['AInt'], list) else ainttot
        aroofmean = arooftot / len(tz_par['ARoof']) if isinstance(tz_par['ARoof'], list) else arooftot
        afloormean = afloortot / len(tz_par['AFloor']) if isinstance(tz_par['AFloor'], list) else afloortot

        tz_par['Awinmean'] = awinmean
        tz_par['Aextmean'] = aextmean
        tz_par['Aintmean'] = aintmean
        tz_par['Aroofmean'] = aroofmean
        tz_par['AFloormean'] = afloormean

        area_tot = awintot + aexttot + ainttot + arooftot + afloortot

        # Split Factor for interal gains
        split_int_ig = ainttot / area_tot
        split_roof_ig = arooftot / area_tot
        split_ext_ig = aexttot / area_tot
        split_win_ig = awintot / area_tot
        split_floor_ig = afloortot / area_tot
        split_ig_dict = {'int': split_int_ig,
                         'roof': split_roof_ig,
                         'ext': split_ext_ig,
                         'floor': split_floor_ig,
                         'win': split_win_ig}
        # Evaluate Components
        has_floor = self.AFloortot.value > 0
        has_roof = self.ARooftot.value > 0

        # Calculate Splitfactors for Solar Radiation
        # (Compared to the Modelica Model we don't calculate n_orientation*n_components split
        # factors, instead we average over all orientations
        # to keep the model simple)

        split_int_sol = ainttot / (area_tot - aextmean - awinmean)
        split_roof_sol = arooftot / (area_tot - aextmean - awinmean)
        split_ext_sol = (aexttot - aextmean) / (area_tot - aextmean - awinmean)
        split_win_sol = (awintot - awinmean) / (area_tot - aextmean - awinmean)
        split_floor_sol = afloortot / (area_tot - aextmean - awinmean)

        split_sol_dict = {'int': split_int_sol,
                          'roof': split_roof_sol,
                          'ext': split_ext_sol,
                          'floor': split_floor_sol,
                          'win': split_win_sol}

        ###### Radiator
        # paramters of radiator
        # steel resistance per layer
        G_radiator = self.LambdaSteel * (2 * self.height * self.length) / self.d_wall / self.N  # W/k
        # steel heat capacity per layer
        c_radiator = self.CapacitySteel * (
                self.length * self.MassSteel) / self.N  # (self.length * self.MassSteel)/Nlayer
        # water volume per layer
        volume_rad = self.VolumeWater * self.length / 1000 / self.N
        # water heat capacity per layer
        c_water = volume_rad * self.rho_water * self.cp_water
        # sum of heat capacity per layer
        c_sum = c_water + c_radiator
        # nominal power per layer
        q_dot_nom = (self.length * self.NominalPower) / self.N
        # heater excess temp
        dT_norm = (((self.n - 1) * (self.dT_V_nom - self.dT_R_nom)) / (
                self.dT_R_nom ** (1 - self.n) - self.dT_V_nom ** (1 - self.n))) ** (
                          1 / self.n)
        kA_conv = ((1 - self.s_eff) * q_dot_nom) / (dT_norm ** self.n)

        # for radiation
        # heater excess temp(radiation)
        delta_nom = (dT_norm + self.T_air_nom) ** 4 - self.T_air_nom ** 4
        A_in_internal = (self.s_eff * q_dot_nom) / (self.sigma * delta_nom * self.eps)
        eps_rad_twostar = 1

        # Initialize
        T_flow_out = []
        Q_flow = []
        T_radiator_sur = []
        q_radiator_conv = []
        q_radiator_rad = []
        q_radiator_rad_total = 0
        q_radiator_conv_total = 0

        layer = self.N.value

        #####layer 1
        T_flow_in_1 = self.T_flow_in_max
        # Calculate return temperature at layer 1
        T_flow_out_1 = (self.m_flow * self.cp_water * T_flow_in_1 + G_radiator * 0.5 * self.T_radiator_m_1) / (
                G_radiator * 0.5 + self.m_flow * self.cp_water)
        T_flow_out.append(T_flow_out_1)

        # Calculate transferred heat flow at layer 1
        Q_flow_1 = self.m_flow * self.cp_water * (T_flow_in_1 - T_flow_out_1)
        Q_flow.append(Q_flow_1)

        # Calculate surface temperature of the layer 1
        T_radiator_sur_1 = 2 * self.T_radiator_m_1 - T_flow_out_1
        T_radiator_sur.append(T_radiator_sur_1)

        # Convection at layer 1
        alpha_conv = kA_conv * ((ca.fabs(self.T_Air - T_radiator_sur_1)) ** (self.n - 1))
        q_radiator_conv_1 = alpha_conv * (T_radiator_sur_1 - self.T_Air)
        q_radiator_conv.append(q_radiator_conv_1)
        # Update current total convective heat
        q_radiator_conv_total += q_radiator_conv_1

        # input flow temperature for next layer
        T_flow_in_2 = T_flow_out_1

        #####layer 2
        T_flow_out_2 = (self.m_flow * self.cp_water * T_flow_in_2 + G_radiator * 0.5 *
                        self.T_radiator_m_2) / (G_radiator * 0.5 + self.m_flow * self.cp_water)
        T_flow_out.append(T_flow_out_2)

        # Calculate transferred heat flow at layer 2
        Q_flow_2 = self.m_flow * self.cp_water * (T_flow_in_2 - T_flow_out_2)
        Q_flow.append(Q_flow_2)

        # Calculate surface temperature of the layer 2
        T_radiator_sur_2 = 2 * self.T_radiator_m_2 - T_flow_out_2
        T_radiator_sur.append(T_radiator_sur_2)

        # Convection at layer 2
        alpha_conv = kA_conv * ((ca.fabs(self.T_Air - T_radiator_sur_2)) ** (self.n - 1))
        q_radiator_conv_2 = alpha_conv * (T_radiator_sur_2 - self.T_Air)
        q_radiator_conv.append(q_radiator_conv_2)
        # Update current total convective heat
        q_radiator_conv_total += q_radiator_conv_2

        #####layer 3
        T_flow_in_3 = T_flow_out_2
        # Calculate return temperature at layer 3
        T_flow_out_3 = (self.m_flow * self.cp_water * T_flow_in_3 + G_radiator * 0.5 *
                        self.T_radiator_m_3) / (G_radiator * 0.5 + self.m_flow * self.cp_water)
        T_flow_out.append(T_flow_out_3)

        # Calculate transferred heat flow at layer 3
        Q_flow_3 = self.m_flow * self.cp_water * (T_flow_in_3 - T_flow_out_3)
        Q_flow.append(Q_flow_3)

        # Calculate surface temperature of the layer 3
        T_radiator_sur_3 = 2 * self.T_radiator_m_3 - T_flow_out_3
        T_radiator_sur.append(T_radiator_sur_3)

        # Convection at layer 3
        alpha_conv = kA_conv * ((ca.fabs(self.T_Air - T_radiator_sur_3)) ** (self.n - 1))
        q_radiator_conv_3 = alpha_conv * (T_radiator_sur_3 - self.T_Air)
        q_radiator_conv.append(q_radiator_conv_3)
        # Update current total convective heat
        q_radiator_conv_total += q_radiator_conv_3

        # internal heat gain convective & radiative from human/devices/lights
        # --- algebraic eq:
        heat_humans_conv = ((0.865 - (0.025 * (self.T_Air - 273.15))) *
                            (
                                    self.activityDegree * 58 * 1.8) + 35) * self.specificPeople * self.AZone * self.ratioConvectiveHeatPeople * self.schedule_human
        heat_humans_rad = (heat_humans_conv *
                           (1 - self.ratioConvectiveHeatPeople) / self.ratioConvectiveHeatPeople)
        heat_devices_conv = self.AZone * self.internalGainsMachinesSpecific * self.schedule_dev * self.ratioConvectiveHeatMachines  # schedule = int_gains[2]
        heat_devices_rad = (heat_devices_conv *
                            (1 - self.ratioConvectiveHeatMachines) / self.ratioConvectiveHeatMachines)
        heat_lights_conv = self.AZone * self.lightingPowerSpecific * self.schedule_light * self.ratioConvectiveHeatLighting
        heat_lights_rad = (heat_lights_conv *
                           (1 - self.ratioConvectiveHeatLighting) / self.ratioConvectiveHeatLighting)
        # Q_flow[W] =ventRate*VAir*air_cp*air_rho*(port_a.T - port_b.T)/hToS
        heat_ACH = self.baseACH * self.VAir * 1000 * 1.25 * (self.T_amb - self.T_Air) / 3600
        q_ig_rad = heat_humans_rad + heat_devices_rad + heat_lights_rad  # + q_radiator_rad_total
        q_ig_conv = heat_humans_conv + heat_devices_conv + heat_lights_conv + q_radiator_conv_total + heat_ACH

        # parameters
        # heat capacities of air in the room in [J/K]
        c_air = self.VAir * self.air_rho * self.air_cp

        # Thermal transmittance
        # k Heat transfer coefficient in [W/K] = convective coefficient of heat transfer [W/m^2*K] * area[m^2]
        # thermal resistance R = 1/k
        # Roof
        if has_roof:
            k_air_roof = self.hConRoof * self.ARooftot
            k_amb_roof = 1 / (1 / ((self.hConRoofOut + self.hRadRoof) * self.ARooftot) + self.RRoofRem)
            k_roof = 1 / self.RRoof
        else:
            k_air_roof = 0
            k_amb_roof = 0
            k_roof = 0

        # Floor
        if has_floor:
            k_air_floor = self.hConFloor * self.AFloortot
            k_soil_floor = 1 / self.RFloorRem
            k_floor = 1 / self.RFloor
        else:
            k_air_floor = 0
            k_floor = 0

        # Exterior Walls
        k_air_ext = self.hConExt * self.AExttot  # * self.fac_air_ext
        k_amb_ext = 1 / (1 / ((self.hConWallOut + self.hRadWall) * self.AExttot) + self.RExtRem)  # * self.fac_amb_ext
        k_ext = 1 / self.RExt

        # Indoor Air
        k_roof_air = k_air_roof
        k_ext_air = k_air_ext
        k_int_air = self.hConInt * self.AInttot  # * self.fac_int_air
        k_win_air = self.hConWin * self.AWintot  # * self.fac_win_air
        k_floor_air = k_air_floor

        # Interior Walls
        k_int = 1 / self.RInt
        # End of calculation of transition coefficients

        # Solar radiation to walls (approximated)
        Q_RadSol_air = (self.Q_RadSol / (self.gWin * (1 - self.ratioWinConRad) * self.ATransparent)
                             * self.gWin * self.ratioWinConRad * self.ATransparent)
        #self.Q_RadSol_int_sol = self.Q_RadSol * split_int_sol
        #self.Q_RadSol_roof_sol = self.Q_RadSol * split_roof_sol
        #self.Q_RadSol_ext_sol = self.Q_RadSol * split_ext_sol

        # calc resistances
        coeff_dict = calc_resistances(tz_par=tz_par, rad_par=rad_par, split_sol=split_sol_dict, split_ig=split_ig_dict)
        # coeff_dict = calc_resistances0(tz_par=tz_par,split_sol=split_sol_dict, split_ig=split_ig_dict)

        T_radiator_sur_exp_1 = T_radiator_sur_1 ** 4
        T_radiator_sur_exp_2 = T_radiator_sur_2 ** 4
        T_radiator_sur_exp_3 = T_radiator_sur_3 ** 4
        T_rad_exp = self.T_rad_nf ** 4
        # Calculate Surface Temperature of components
        # The surface temperature is a function of the symbolic variable
        # (T_Air, T_ext, T_int, T_roof, Q_RadSol, q_ig_rad, T_preTemWin)
        if has_floor:
            self.T_ExtWall_sur = (coeff_dict['T_ext_sur']['T_Air'] * self.T_Air +
                                  coeff_dict['T_ext_sur']['T_ext'] * self.T_ExtWall +
                                  coeff_dict['T_ext_sur']['T_int'] * self.T_IntWall +
                                  coeff_dict['T_ext_sur']['T_roof'] * self.T_Roof +
                                  coeff_dict['T_ext_sur']['T_floor'] * self.T_Floor +
                                  coeff_dict['T_ext_sur']['Q_RadSol'] * self.Q_RadSol +
                                  coeff_dict['T_ext_sur']['q_ig_rad'] * q_ig_rad +
                                  coeff_dict['T_ext_sur']['T_preTemWin'] * self.T_preTemWin +
                                  coeff_dict['T_ext_sur']['T_radiator_sur_exp_1'] * T_radiator_sur_exp_1 +
                                  coeff_dict['T_ext_sur']['T_radiator_sur_exp_2'] * T_radiator_sur_exp_2 +
                                  coeff_dict['T_ext_sur']['T_radiator_sur_exp_3'] * T_radiator_sur_exp_3 +
                                  coeff_dict['T_ext_sur']['T_rad_exp'] * T_rad_exp)
            self.T_IntWall_sur = (coeff_dict['T_int_sur']['T_Air'] * self.T_Air +
                                  coeff_dict['T_int_sur']['T_ext'] * self.T_ExtWall +
                                  coeff_dict['T_int_sur']['T_int'] * self.T_IntWall +
                                  coeff_dict['T_int_sur']['T_roof'] * self.T_Roof +
                                  coeff_dict['T_int_sur']['T_floor'] * self.T_Floor +
                                  coeff_dict['T_int_sur']['Q_RadSol'] * self.Q_RadSol +
                                  coeff_dict['T_int_sur']['q_ig_rad'] * q_ig_rad +
                                  coeff_dict['T_int_sur']['T_preTemWin'] * self.T_preTemWin +
                                  coeff_dict['T_int_sur']['T_radiator_sur_exp_1'] * T_radiator_sur_exp_1 +
                                  coeff_dict['T_int_sur']['T_radiator_sur_exp_2'] * T_radiator_sur_exp_2 +
                                  coeff_dict['T_int_sur']['T_radiator_sur_exp_3'] * T_radiator_sur_exp_3 +
                                  coeff_dict['T_int_sur']['T_rad_exp'] * T_rad_exp)
            self.T_Roof_sur = (coeff_dict['T_roof_sur']['T_Air'] * self.T_Air +
                               coeff_dict['T_roof_sur']['T_ext'] * self.T_ExtWall +
                               coeff_dict['T_roof_sur']['T_int'] * self.T_IntWall +
                               coeff_dict['T_roof_sur']['T_roof'] * self.T_Roof +
                               coeff_dict['T_roof_sur']['T_floor'] * self.T_Floor +
                               coeff_dict['T_roof_sur']['Q_RadSol'] * self.Q_RadSol +
                               coeff_dict['T_roof_sur']['q_ig_rad'] * q_ig_rad +
                               coeff_dict['T_roof_sur']['T_preTemWin'] * self.T_preTemWin +
                               coeff_dict['T_roof_sur']['T_radiator_sur_exp_1'] * T_radiator_sur_exp_1 +
                               coeff_dict['T_roof_sur']['T_radiator_sur_exp_2'] * T_radiator_sur_exp_2 +
                               coeff_dict['T_roof_sur']['T_radiator_sur_exp_3'] * T_radiator_sur_exp_3 +
                               coeff_dict['T_roof_sur']['T_rad_exp'] * T_rad_exp)
            self.T_Win_sur = (coeff_dict['T_win_sur']['T_Air'] * self.T_Air +
                              coeff_dict['T_win_sur']['T_ext'] * self.T_ExtWall +
                              coeff_dict['T_win_sur']['T_int'] * self.T_IntWall +
                              coeff_dict['T_win_sur']['T_roof'] * self.T_Roof +
                              coeff_dict['T_win_sur']['T_floor'] * self.T_Floor +
                              coeff_dict['T_win_sur']['Q_RadSol'] * self.Q_RadSol +
                              coeff_dict['T_win_sur']['q_ig_rad'] * q_ig_rad +
                              coeff_dict['T_win_sur']['T_preTemWin'] * self.T_preTemWin +
                              coeff_dict['T_win_sur']['T_radiator_sur_exp_1'] * T_radiator_sur_exp_1 +
                              coeff_dict['T_win_sur']['T_radiator_sur_exp_2'] * T_radiator_sur_exp_2 +
                              coeff_dict['T_win_sur']['T_radiator_sur_exp_3'] * T_radiator_sur_exp_3 +
                              coeff_dict['T_win_sur']['T_rad_exp'] * T_rad_exp)
            self.T_Floor_sur = (coeff_dict['T_floor_sur']['T_Air'] * self.T_Air +
                                coeff_dict['T_floor_sur']['T_ext'] * self.T_ExtWall +
                                coeff_dict['T_floor_sur']['T_int'] * self.T_IntWall +
                                coeff_dict['T_floor_sur']['T_roof'] * self.T_Roof +
                                coeff_dict['T_floor_sur']['T_floor'] * self.T_Floor +
                                coeff_dict['T_floor_sur']['Q_RadSol'] * self.Q_RadSol +
                                coeff_dict['T_floor_sur']['q_ig_rad'] * q_ig_rad +
                                coeff_dict['T_floor_sur']['T_preTemWin'] * self.T_preTemWin +
                                coeff_dict['T_floor_sur']['T_radiator_sur_exp_1'] * T_radiator_sur_exp_1 +
                                coeff_dict['T_floor_sur']['T_radiator_sur_exp_2'] * T_radiator_sur_exp_2 +
                                coeff_dict['T_floor_sur']['T_radiator_sur_exp_3'] * T_radiator_sur_exp_3 +
                                coeff_dict['T_floor_sur']['T_rad_exp'] * T_rad_exp)
            self.T_rad_nf.alg = (coeff_dict['T_rad']['T_Air'] * self.T_Air +
                              coeff_dict['T_rad']['T_ext'] * self.T_ExtWall +
                              coeff_dict['T_rad']['T_int'] * self.T_IntWall +
                              coeff_dict['T_rad']['T_roof'] * self.T_Roof +
                              coeff_dict['T_rad']['T_floor'] * self.T_Floor +
                              coeff_dict['T_rad']['Q_RadSol'] * self.Q_RadSol +
                              coeff_dict['T_rad']['q_ig_rad'] * q_ig_rad +
                              coeff_dict['T_rad']['T_preTemWin'] * self.T_preTemWin +
                              coeff_dict['T_rad']['T_radiator_sur_exp_1'] * T_radiator_sur_exp_1 +
                              coeff_dict['T_rad']['T_radiator_sur_exp_2'] * T_radiator_sur_exp_2 +
                              coeff_dict['T_rad']['T_radiator_sur_exp_3'] * T_radiator_sur_exp_3 +
                              coeff_dict['T_rad']['T_rad_exp'] * T_rad_exp)
        else:
            self.T_ExtWall_sur = (coeff_dict['T_ext_sur']['T_Air'] * self.T_Air +
                                  coeff_dict['T_ext_sur']['T_ext'] * self.T_ExtWall +
                                  coeff_dict['T_ext_sur']['T_int'] * self.T_IntWall +
                                  coeff_dict['T_ext_sur']['T_roof'] * self.T_Roof +
                                  coeff_dict['T_ext_sur']['Q_RadSol'] * self.Q_RadSol +
                                  coeff_dict['T_ext_sur']['q_ig_rad'] * q_ig_rad +
                                  coeff_dict['T_ext_sur']['T_preTemWin'] * self.T_preTemWin +
                                  coeff_dict['T_ext_sur']['T_radiator_sur_exp_1'] * T_radiator_sur_exp_1 +
                                  coeff_dict['T_ext_sur']['T_radiator_sur_exp_2'] * T_radiator_sur_exp_2 +
                                  coeff_dict['T_ext_sur']['T_radiator_sur_exp_3'] * T_radiator_sur_exp_3 +
                                  coeff_dict['T_ext_sur']['T_rad_exp'] * T_rad_exp)
            self.T_IntWall_sur = (coeff_dict['T_int_sur']['T_Air'] * self.T_Air +
                                  coeff_dict['T_int_sur']['T_ext'] * self.T_ExtWall +
                                  coeff_dict['T_int_sur']['T_int'] * self.T_IntWall +
                                  coeff_dict['T_int_sur']['T_roof'] * self.T_Roof +
                                  coeff_dict['T_int_sur']['Q_RadSol'] * self.Q_RadSol +
                                  coeff_dict['T_int_sur']['q_ig_rad'] * q_ig_rad +
                                  coeff_dict['T_int_sur']['T_preTemWin'] * self.T_preTemWin +
                                  coeff_dict['T_int_sur']['T_radiator_sur_exp_1'] * T_radiator_sur_exp_1 +
                                  coeff_dict['T_int_sur']['T_radiator_sur_exp_2'] * T_radiator_sur_exp_2 +
                                  coeff_dict['T_int_sur']['T_radiator_sur_exp_3'] * T_radiator_sur_exp_3 +
                                  coeff_dict['T_int_sur']['T_rad_exp'] * T_rad_exp)
            self.T_Roof_sur = (coeff_dict['T_roof_sur']['T_Air'] * self.T_Air +
                               coeff_dict['T_roof_sur']['T_ext'] * self.T_ExtWall +
                               coeff_dict['T_roof_sur']['T_int'] * self.T_IntWall +
                               coeff_dict['T_roof_sur']['T_roof'] * self.T_Roof +
                               coeff_dict['T_roof_sur']['Q_RadSol'] * self.Q_RadSol +
                               coeff_dict['T_roof_sur']['q_ig_rad'] * q_ig_rad +
                               coeff_dict['T_roof_sur']['T_preTemWin'] * self.T_preTemWin +
                               coeff_dict['T_roof_sur']['T_radiator_sur_exp_1'] * T_radiator_sur_exp_1 +
                               coeff_dict['T_roof_sur']['T_radiator_sur_exp_2'] * T_radiator_sur_exp_2 +
                               coeff_dict['T_roof_sur']['T_radiator_sur_exp_3'] * T_radiator_sur_exp_3 +
                               coeff_dict['T_roof_sur']['T_rad_exp'] * T_rad_exp)
            self.T_Win_sur = (coeff_dict['T_win_sur']['T_Air'] * self.T_Air +
                              coeff_dict['T_win_sur']['T_ext'] * self.T_ExtWall +
                              coeff_dict['T_win_sur']['T_int'] * self.T_IntWall +
                              coeff_dict['T_win_sur']['T_roof'] * self.T_Roof +
                              coeff_dict['T_win_sur']['Q_RadSol'] * self.Q_RadSol +
                              coeff_dict['T_win_sur']['q_ig_rad'] * q_ig_rad +
                              coeff_dict['T_win_sur']['T_preTemWin'] * self.T_preTemWin +
                              coeff_dict['T_win_sur']['T_radiator_sur_exp_1'] * T_radiator_sur_exp_1 +
                              coeff_dict['T_win_sur']['T_radiator_sur_exp_2'] * T_radiator_sur_exp_2 +
                              coeff_dict['T_win_sur']['T_radiator_sur_exp_3'] * T_radiator_sur_exp_3 +
                              coeff_dict['T_win_sur']['T_rad_exp'] * T_rad_exp)
            self.T_rad_nf.alg = (coeff_dict['T_rad']['T_Air'] * self.T_Air +
                              coeff_dict['T_rad']['T_ext'] * self.T_ExtWall +
                              coeff_dict['T_rad']['T_int'] * self.T_IntWall +
                              coeff_dict['T_rad']['T_roof'] * self.T_Roof +
                              coeff_dict['T_rad']['Q_RadSol'] * self.Q_RadSol +
                              coeff_dict['T_rad']['q_ig_rad'] * q_ig_rad +
                              coeff_dict['T_rad']['T_preTemWin'] * self.T_preTemWin +
                              coeff_dict['T_rad']['T_radiator_sur_exp_1'] * T_radiator_sur_exp_1 +
                              coeff_dict['T_rad']['T_radiator_sur_exp_2'] * T_radiator_sur_exp_2 +
                              coeff_dict['T_rad']['T_radiator_sur_exp_3'] * T_radiator_sur_exp_3 +
                              coeff_dict['T_rad']['T_rad_exp'] * T_rad_exp)

        # Differential Equations of States
        if has_floor:
            self.T_Air.ode = (1 / c_air) * (
                    (self.T_Roof_sur - self.T_Air) * k_roof_air +
                    (self.T_ExtWall_sur - self.T_Air) * k_ext_air
                    + (self.T_IntWall_sur - self.T_Air) * k_int_air
                    + (self.T_Win_sur - self.T_Air) * k_win_air
                    + (self.T_Floor_sur - self.T_Air) * k_floor_air
                    + q_ig_conv * self.fac_IG_air
                    + Q_RadSol_air * self.fac_sol_air)
        else:
            self.T_Air.ode = (1 / c_air) * (
                    (self.T_Roof_sur - self.T_Air) * k_roof_air +
                    (self.T_ExtWall_sur - self.T_Air) * k_ext_air
                    + (self.T_IntWall_sur - self.T_Air) * k_int_air
                    + (self.T_Win_sur - self.T_Air) * k_win_air
                    + (self.T_Floor - self.T_Air) * k_floor_air
                    + q_ig_conv * self.fac_IG_air
                    + Q_RadSol_air * self.fac_sol_air)

        if has_roof:
            self.T_Roof.ode = (1 / self.CRoof) * (
                    (self.T_Roof_sur - self.T_Roof) * k_roof +
                    (self.T_preTemRoof - self.T_Roof) * k_amb_roof
            )
        else:
            self.T_Roof.ode = 0 * self.T_Air

        if has_floor:
            self.T_Floor.ode = (1 / self.CFloor) * (
                    (self.T_Floor_sur - self.T_Floor) * k_floor +
                    (self.T_soil - self.T_Floor) * k_soil_floor)
        else:
            self.T_Floor.ode = 0 * self.T_Air  # not physical, just here to keep agent config the same, even when there is no floor

        self.T_ExtWall.ode = (1 / self.CExt) * ((self.T_ExtWall_sur - self.T_ExtWall) * k_ext
                                                + (self.T_preTemWall - self.T_ExtWall) * k_amb_ext)

        self.T_IntWall.ode = (1 / self.CInt) * ((self.T_IntWall_sur - self.T_IntWall) * k_int)

        # TODO: Radiation of radiator
        # Radiation at layer 1
        q_radiator_rad_1 = self.sigma * eps_rad_twostar * A_in_internal * (
                T_radiator_sur_1 ** 4 - self.T_rad_nf ** 4)
        q_radiator_rad.append(q_radiator_rad_1)
        # Update current total radiative heat
        q_radiator_rad_total += q_radiator_rad_1
        # Radiation at layer 2
        q_radiator_rad_2 = self.sigma * eps_rad_twostar * A_in_internal * (
                T_radiator_sur_2 ** 4 - self.T_rad_nf ** 4)
        q_radiator_rad.append(q_radiator_rad_2)
        # Update current total radiative heat
        q_radiator_rad_total += q_radiator_rad_2
        # Radiation at layer 3
        q_radiator_rad_3 = self.sigma * eps_rad_twostar * A_in_internal * (
                T_radiator_sur_3 ** 4 - self.T_rad_nf ** 4)
        q_radiator_rad.append(q_radiator_rad_3)
        # Update current total radiative heat
        q_radiator_rad_total += q_radiator_rad_3

        # DGL: radiator temperature at layer i
        self.T_radiator_m_1.ode = 1 / c_sum * (Q_flow_1 - (q_radiator_conv_1 + q_radiator_rad_1))
        self.T_radiator_m_2.ode = 1 / c_sum * (Q_flow_2 - (q_radiator_conv_2 + q_radiator_rad_2))
        self.T_radiator_m_3.ode = 1 / c_sum * (Q_flow_3 - (q_radiator_conv_3 + q_radiator_rad_3))

        # calculate transferred Q_flow
        # here the calculated Q_flow is actually as same as the sum of all Q_flow at each layer
        Q_flow_total = self.m_flow * self.cp_water * (self.T_flow_in_max - T_flow_out[layer - 1])
        #Q_flow_total = Q_flow_1 + Q_flow_2 + Q_flow_3
        self.Q_flow_total_max.alg = Q_flow_total
        self.P_el_max_alg.alg = ca.fabs(Q_flow_total) / self.COP
        self.Q_abs = ca.fabs(Q_flow_total)

        # debug
        self.T_flow_out_NF.alg = T_flow_out[layer - 1]

        # region Constraints: List[(lower bound, function, upper bound)]
        self.constraints = [
            # soft constraints
            # T_air
            (-inf, self.T_Air - self.T_slack, self.T_upper),
            (self.T_lower, self.T_Air + self.T_slack, inf),
            (0, self.T_slack, inf),
            # Q_flow
            (0, self.Q_flow_slack1, inf),
            (0, self.Q_flow_slack2, inf),
            (0, self.Q_flow_total_max + self.Q_flow_slack1 - self.Q_flow_slack2, 0),
            #(273.15, T_flow_out[layer - 1], self.T_flow_in_max)
        ]

        # region Objective function
        objective = sum(
            [
                (self.s_T / (self.s_T + self.s_Pel)) * self.T_slack ** 2,
                -(self.s_Pel / (self.s_T + self.s_Pel)) * (
                        self.Q_abs) / self.COP / 1000,
            ]
        )

        return objective
