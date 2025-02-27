import math
import pandas as pd
import logging
from itertools import combinations
from typing import List
from modelica_parser import parse_modelica_record
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
import sys
sys.path.append("Model/local")
from mpc.utils.calc_resistances import calc_resistances

logger = logging.getLogger(__name__)

# parameters for thermal zone
path_to_mos = r"Model\local\mpc\ASHRAE140_900.mo"
tz_par = parse_modelica_record(path_to_mos)


class SimpleTestHallModelConfig(CasadiModelConfig):
    # mode: Literal[0, 1] = 0
    inputs: List[CasadiInput] = [
        # controls
        CasadiInput(name="Q_Tabs_set", value=0, unit="W", description="Setpoint for TABS"),
        CasadiInput(name="T_ahu_set", value=292, unit="K", description="Setpoint for ahu"),

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
        CasadiInput(name="m_flow_ahu", value=0.1, unit="kg/s", description="Radiative solar heat"),
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
        CasadiState(name="T_Air", value=293.15, unit="K", description="Temperature of zone"),
        CasadiState(name="T_Roof", value=293.15, unit="K",
                    description="Temperature of roof in middle"),
        CasadiState(name="T_Floor", value=293.15, unit="K",
                    description="Temperature of floor in middle"),
        CasadiState(name="T_ExtWall", value=293.15, unit="K",
                    description="Outer wall temperature in middle"),
        CasadiState(name="T_IntWall", value=293.15, unit="K",
                    description="Inner wall temperature in middle"),
        # CasadiState(name="T_Win", value=293.15, unit="K", description="Window temperature on inside"),
        CasadiState(name="T_Tabs", value=293.15, unit="K", description="TABS temperature"),
        CasadiState(name="Q_Tabs_set_del", value=0, unit="W",
                    description="Setpoint for TABS incl. delay"),
        # CasadiState(name="T_ahu_set_del", value=293.15, unit="W", description="Setpoint for ahu incl. delay"),
        # slack variables
        CasadiState(name="T_slack", value=0, unit="K",
                    description="Slack variable for (soft) constraint of T."),
        CasadiState(name="Q_tabs_slack1", value=0, unit="W",
                    description="Slack variable for abs of q_tabs."),
        CasadiState(name="Q_tabs_slack2", value=0, unit="W",
                    description="lack variable for abs of q_tabs."),
        CasadiState(name="Q_ahu_slack1", value=0, unit="W",
                    description="Slack variable for abs of q_ahu."),
        CasadiState(name="Q_ahu_slack2", value=0, unit="W",
                    description="lack variable for abs of q_ahu."),
    ]

    parameters: List[CasadiParameter] = [
        CasadiParameter(name="delay_const", value=280),
        CasadiParameter(name="time_step", value=900),
        CasadiParameter(name="mode", value=0),
        CasadiParameter(name="COP", value=3,
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
        CasadiParameter(name="air_rho", value=1.2,
                        unit="kg/m**3", description="density of air"),
        CasadiParameter(name="air_cp", value=1006,
                        unit="J/kg*K", description="thermal capacity of air"),
        CasadiParameter(name="CRoof", value=tz_par['CRoof'],
                        unit="J/K", description="Heat capacities of roof"),
        CasadiParameter(name="CExt", value=tz_par['CExt'],
                        unit="J/K", description="Heat capacities of exterior walls"),
        CasadiParameter(name="CInt", value=tz_par['CInt'],
                        unit="J/K", description="Heat capacities of interior walls"),
        CasadiParameter(name="CFloor", value=tz_par['CFloor'],
                        unit="J/K", description="Heat capacities of floor"),
        CasadiParameter(name="concrete_rho", value=2100,
                        unit="kg/m**3", description="density of concrete"),
        CasadiParameter(name="concrete_cp", value=1000,
                        unit="J/(kg*K)", description="specific heat capacity of concrete"),
        CasadiParameter(name="d_Tabs", value=0.1,
                        unit="m", description="thickness of activated concrete"),
        CasadiParameter(name="Area_Tabs", value=48,
                        unit="m**2", description="area of activated concrete"),

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
        CasadiParameter(name="hConTabs", value=20,
                        unit="W/(m^2*K)",
                        description="TABS convective coefficient of heat transfer (indoor)"),
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

        CasadiParameter(name="s_T", value=0.5,
                        unit="-", description="Weight for T_slack"),
        CasadiParameter(name="s_Pel", value=0.5,
                        unit="-", description="Weight for P_el"),

        # calibration factor, used for wandb
        CasadiParameter(name="fac_amb_win", value=1,
                        unit="-", description="calibration factor"),
        CasadiParameter(name="fac_air_roof", value=1,
                        unit="-", description="calibration factor"),
        CasadiParameter(name="fac_ext_roof", value=1,
                        unit="-", description="calibration factor"),
        CasadiParameter(name="fac_int_roof", value=1,
                        unit="-", description="calibration factor"),
        CasadiParameter(name="fac_amb_roof", value=1,
                        unit="-", description="calibration factor"),
        CasadiParameter(name="fac_floor_roof", value=1,
                        unit="-", description="calibration factor"),
        CasadiParameter(name="fac_win_roof", value=1,
                        unit="-", description="calibration factor"),
        CasadiParameter(name="fac_air_floor", value=1,
                        unit="-", description="calibration factor"),
        CasadiParameter(name="fac_ext_floor", value=1,
                        unit="-", description="calibration factor"),
        CasadiParameter(name="fac_win_floor", value=1,
                        unit="-", description="calibration factor"),
        CasadiParameter(name="fac_int_floor", value=1,
                        unit="-", description="calibration factor"),
        CasadiParameter(name="fac_air_ext", value=1,
                        unit="-", description="calibration factor"),
        CasadiParameter(name="fac_int_ext", value=1,
                        unit="-", description="calibration factor"),
        CasadiParameter(name="fac_amb_ext", value=1,
                        unit="-", description="calibration factor"),
        CasadiParameter(name="fac_win_ext", value=1,
                        unit="-", description="calibration factor"),
        CasadiParameter(name="fac_int_air", value=1,
                        unit="-", description="calibration factor"),
        CasadiParameter(name="fac_win_air", value=1,
                        unit="-", description="calibration factor"),
        CasadiParameter(name="fac_win_int", value=1,
                        unit="-", description="calibration factor"),
        CasadiParameter(name="CWinScaler", value=1e-5,
                        unit="-",
                        description="Scaler for window capacity. In FMU there is no capacity for window."
                                    "Here, we need ont to calculate state. The initial guess can be scaled with this value."),
        CasadiParameter(name="fac_IG_air", value=1,
                        unit="-", description="factor for internal gains"),
        CasadiParameter(name="fac_IG_roof", value=1,
                        unit="-", description="factor for internal gains"),
        CasadiParameter(name="fac_IG_floor", value=1,
                        unit="-", description="factor for internal gains"),
        CasadiParameter(name="fac_IG_int", value=1,
                        unit="-", description="factor for internal gains"),
        CasadiParameter(name="fac_IG_ext", value=1,
                        unit="-", description="factor for internal gains"),
        CasadiParameter(name="fac_IG_win", value=1,
                        unit="-", description="factor for internal gains"),
        CasadiParameter(name="fac_IG", value=1,
                        unit="-", description="factor for internal gains"),
        CasadiParameter(name="fac_sol_air", value=1,
                        unit="-", description="factor for internal gains"),
        CasadiParameter(name="fac_sol_roof", value=1,
                        unit="-", description="factor for internal gains"),
        CasadiParameter(name="fac_sol_floor", value=1,
                        unit="-", description="factor for internal gains"),
        CasadiParameter(name="fac_sol_ext", value=1,
                        unit="-", description="factor for internal gains"),
        CasadiParameter(name="fac_sol_int", value=1,
                        unit="-", description="factor for internal gains"),
        CasadiParameter(name="fac_sol_win", value=1,
                        unit="-", description="factor for internal gains"),
    ]

    outputs: List[CasadiOutput] = [
        CasadiOutput(name="P_el_c", unit='W'),
        CasadiOutput(name="E_stored", unit='kWh'),
        CasadiOutput(name="q_tabs_del_out", unit='W', value=0),
        CasadiOutput(name="T_out", unit='K'),
        CasadiOutput(name="T_Win_sur", value=293.15, unit="K", description="Surface Temperature on the inside of the window"),
        CasadiOutput(name="T_ExtWall_sur", value=293.15, unit="K", description="Surface Temperature on the inside of the external wall"),
        CasadiOutput(name="T_IntWall_sur", value=293.15, unit="K", description="Surface Temperature on the inside of the internal wall"),
        CasadiOutput(name="T_Roof_sur", value=293.15, unit="K", description="Surface Temperature on the inside of the roof"),
        CasadiOutput(name="heat_humans_conv", value=500, unit="W", description="Convective internal gains heatflow from humans"),
        CasadiOutput(name="heat_humans_rad", value=500, unit="W", description="Radiative internal gains heatflow from humans"),
        CasadiOutput(name="heat_devices_conv", value=500, unit="W", description="Convective internal gains heatflow from devices"),
        CasadiOutput(name="heat_devices_rad", value=500, unit="W", description="Radiative internal gains heatflow from devices"),
        CasadiOutput(name="heat_lights_conv", value=500, unit="W", description="Convective internal gains heatflow from lights"),
        CasadiOutput(name="heat_lights_rad", value=500, unit="W", description="Radiative internal gains heatflow from lights"),
        CasadiOutput(name="Q_RadSol_ext_sol", value=500, unit="W", description="Radiative internal gains heatflow from lights"),
        CasadiOutput(name="Q_RadSol_int_sol", value=500, unit="W", description="Radiative internal gains heatflow from lights"),
        CasadiOutput(name="Q_RadSol_roof_sol", value=500, unit="W", description="Radiative internal gains heatflow from lights"),
        CasadiOutput(name="Q_tabs_abs", value=0, unit="W", description=""),
        CasadiOutput(name="Q_ahu_abs", value=0, unit="W", description=""),
        CasadiOutput(name="Q_Ahu", value=5000, unit="W", description="Heat of ahu"),
        # CasadiOutput(name="P_el", value=7000, unit="W", description="P_el"),
        # CasadiOutput(name="q_ig_conv", value=7000, unit="W",
        #        description="Internal gains (convective)"),
        # CasadiOutput(name="q_Roof", value=7000, unit="W", description="Heat into zone through roof"),
        # CasadiOutput(name="q_ExtWall", value=7000, unit="W",
        #        description="Heat into zone through exterior wall"),
        # CasadiOutput(name="q_IntWall", value=7000, unit="W",
        #        description="Heat into zone through interior wall"),
        # CasadiOutput(name="q_Win", value=7000, unit="W",
        #        description="Heat into zone through window"),
        # CasadiOutput(name="q_Floor", value=7000, unit="W",
        #        description="Heat into zone through floor"),
        # CasadiOutput(name="q_Tabs", value=0, unit="W", description="Heat into zone through TABS"),
        CasadiOutput(name="q_Ahu", value=7000, unit="W", description="Heat into zone through AHU"),
        # CasadiOutput(name="Q_Tabs", value=0, unit="W", description="Heat from water to TABS"),
        CasadiOutput(name="Q_RadSol_air", value=0, unit="W",
               description="Approximated solar radiation for air"),

        CasadiOutput(name="heat_roof", value=500, unit="W", description="Heatflow through roof"),
        # CasadiOutput(name="heat_window", value=500, unit="W", description="Heatflow through window"),
        CasadiOutput(name="heat_extWall", value=500, unit="W",
               description="Heatflow through extWall"),
        # CasadiOutput(name="T_Win_outdoor", value=280, unit="K", description="Test"),
    ]

class SimpleTestHall(CasadiModel):
    config: SimpleTestHallModelConfig

    def setup_system(self):
        self.q_tabs_del_out.alg = self.Q_Tabs_set_del
        self.T_out.alg = self.T_Air
        # Get areas as python variable (not casadi)
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
        area_tot = awintot + aexttot + ainttot + arooftot + afloortot
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

        # # Calculate Total and mean Areas
        # area_tot = self.AWintot + self.AExttot + self.AInttot + self.ARooftot + self.AFloortot
        # # 171.6 = 12 + 63.6 + 48 + 48 + 0
        #
        # # # Calculate Splitfactors for Internal Gains
        # split_int_ig = self.AInttot / area_tot
        # split_roof_ig = self.ARooftot / area_tot
        # split_ext_ig = self.AExttot / area_tot
        # split_win_ig = self.AWintot / area_tot
        # split_floor_ig = self.AFloortot / area_tot

        # Calculate Splitfactors for Solar Radiation
        # (Compared to the Modelica Model we don't calculate n_orientation*n_components split
        # factors, instead we average over all orientations
        # to keep the model simple)
        split_int_sol = 0.32  # (AInttot - AIntMean)/(area_tot - AMeanTot)
        split_roof_sol = 0.32  # (ARooftot - ARoofMean)/(area_tot - AMeanTot)
        split_ext_sol = 0.36  # (AExttot - AExtMean)/(area_tot - AMeanTot)
        split_win_sol = 0  # (AWintot - AWinMean)/(area_tot - AMeanTot)
        split_floor_sol = 0  # (AFloortot - AFloorMean)/(area_tot - AMeanTot)
        split_sol_dict = {'int': split_int_sol,
                          'roof': split_roof_sol,
                          'ext': split_ext_sol,
                          'floor': split_floor_sol,
                          'win': split_win_sol}

        # calc resistances
        coeff_dict = calc_resistances(tz_par=tz_par, split_sol=split_sol_dict, split_ig=split_ig_dict)


        # --- algebraic eq:
        heat_humans_conv = ((0.865 - (0.025 * (self.T_Air - 273.15))) *
                            (
                                    self.activityDegree * 58 * 1.8) + 35) * self.specificPeople * self.AZone * self.ratioConvectiveHeatPeople * self.schedule_human
        heat_humans_rad = heat_humans_conv * (
                1 - self.ratioConvectiveHeatPeople) / self.ratioConvectiveHeatPeople
        heat_devices_conv = self.AZone * self.internalGainsMachinesSpecific * self.schedule_dev * self.ratioConvectiveHeatMachines  # schedule = int_gains[2]
        heat_devices_rad = heat_devices_conv * (
                1 - self.ratioConvectiveHeatMachines) / self.ratioConvectiveHeatMachines
        heat_lights_conv = self.AZone * self.lightingPowerSpecific * self.schedule_light * self.ratioConvectiveHeatLighting
        heat_lights_rad = heat_lights_conv * (
                1 - self.ratioConvectiveHeatLighting) / self.ratioConvectiveHeatLighting
        # debug
        self.heat_humans_conv.alg = ((0.865 - (0.025 * (self.T_Air - 273.15))) * (
                self.activityDegree * 58 * 1.8) + 35) * self.specificPeople * self.AZone * self.ratioConvectiveHeatPeople * self.schedule_human
        self.heat_humans_rad.alg = heat_humans_conv * (
                1 - self.ratioConvectiveHeatPeople) / self.ratioConvectiveHeatPeople
        self.heat_devices_conv.alg = self.AZone * self.internalGainsMachinesSpecific * self.schedule_dev * self.ratioConvectiveHeatMachines  # schedule = int_gains[2]
        self.heat_devices_rad.alg = heat_devices_conv * (
                1 - self.ratioConvectiveHeatMachines) / self.ratioConvectiveHeatMachines
        self.heat_lights_conv.alg = self.AZone * self.lightingPowerSpecific * self.schedule_light * self.ratioConvectiveHeatLighting
        self.heat_lights_rad.alg = heat_lights_conv * (
                1 - self.ratioConvectiveHeatLighting) / self.ratioConvectiveHeatLighting

        q_ig_conv = (heat_humans_conv + heat_devices_conv + heat_lights_conv) * self.fac_IG
        q_ig_rad = heat_humans_rad + heat_devices_rad + heat_lights_rad

        # parameters
        # capacities
        c_air = self.VAir * self.air_rho * self.air_cp
        c_tabs = self.Area_Tabs * self.d_Tabs * self.concrete_rho * self.concrete_cp  # 1008e4

        # Thermal transmittance
        # Roof
        if has_roof:
            k_air_roof = self.hConRoof * self.ARooftot  # * self.fac_air_roof
            k_ext_roof = self.hRad * min(self.AExttot.value,
                                         self.ARooftot.value)  # * self.fac_ext_roof
            k_int_roof = self.hRad * min(self.AInttot.value,
                                         self.ARooftot.value)  # * self.fac_int_roof
            k_amb_roof = 1/(1/((self.hConRoofOut + self.hRadRoof) * self.ARooftot) + self.RRoofRem)  # * self.fac_amb_roof
            if has_floor:
                k_floor_roof = self.hRad * min(self.AFloortot.value,
                                               self.ARooftot.value)  # * self.fac_floor_roof
            else:
                k_floor_roof = 0
            k_win_roof = min(self.AWintot.value,
                             self.ARooftot.value) * self.hRad  # * self.fac_win_roof
            k_roof = 1 / self.RRoof
        else:
            k_air_roof = 0
            k_ext_roof = 0
            k_int_roof = 0
            k_amb_roof = 0
            k_floor_roof = 0
            k_win_roof = 0
            k_roof = 0
        # Floor # TODO
        if has_floor:
            k_air_floor = self.hConFloor * self.AFloortot
            k_ext_floor = self.hRad * min(self.AFloortot.value, self.AExttot.value)
            k_win_floor = min(self.AWintot.value, self.AFloortot.value)
            k_roof_floor = k_floor_roof
            k_int_floor = self.hRad * min(self.AFloortot.value, self.AInttot.value)
            k_floor = 1 / self.RFloor
        else:
            k_roof_floor = 0
            k_ext_floor = 0
            k_int_floor = 0
            k_air_floor = 0
            k_win_floor = 0
            k_floor = 0

        # Exterior Walls
        k_air_ext = self.hConExt * self.AExttot  # * self.fac_air_ext
        k_roof_ext = k_ext_roof
        k_int_ext = self.hRad * min(self.AExttot.value, self.AInttot.value)  # * self.fac_int_ext
        k_amb_ext = 1 / (1 / ((self.hConWallOut + self.hRadWall) * self.AExttot) + self.RExtRem)  # * self.fac_amb_ext
        k_win_ext = min(self.AExttot.value, self.AWintot.value) * self.hRad  # * self.fac_win_ext
        k_floor_ext = k_ext_floor
        k_ext = 1 / self.RExt

        # Indoor Air
        k_roof_air = k_air_roof
        k_ext_air = k_air_ext
        k_int_air = self.hConInt * self.AInttot  # * self.fac_int_air
        k_win_air = self.hConWin * self.AWintot  # * self.fac_win_air
        k_floor_air = k_air_floor

        # Interior Walls
        k_air_int = k_int_air
        k_ext_int = k_int_ext
        k_roof_int = k_int_roof
        k_win_int = min(self.AWintot.value, self.AInttot.value) * self.hRad  # * self.fac_win_int
        k_int = 1 / self.RInt
        k_floor_int = k_int_floor

        # window

        k_roof_win = k_win_roof
        k_ext_win = k_win_ext
        k_int_win = k_win_int
        k_floor_win = k_win_floor
        k_air_win = k_win_air
        k_amb_win = 1 / (1 / ((self.hConWinOut + self.hRadWall) * self.AWintot) + self.RWin) #  * self.fac_amb_win
        k_win_amb = k_amb_win

        # TABS
        k_tabs_air = self.Area_Tabs * self.hConTabs  # 960

        # Solar radiation to walls (approximated)
        self.Q_RadSol_air.alg = (
                self.Q_RadSol / (self.gWin * (1 - self.ratioWinConRad) * self.ATransparent)
                * self.gWin * self.ratioWinConRad * self.ATransparent)
        self.Q_RadSol_int_sol.alg = self.Q_RadSol * split_int_sol
        self.Q_RadSol_roof_sol.alg = self.Q_RadSol * split_roof_sol
        self.Q_RadSol_ext_sol.alg = self.Q_RadSol * split_ext_sol

        # add delay to set points
        self.Q_Tabs_set_del.ode = (self.Q_Tabs_set - self.Q_Tabs_set_del) / self.delay_const
        # self.T_ahu_set_del.ode = (self.T_ahu_set-self.T_ahu_set_del) / 280

        # self.q_Ahu.alg = self.m_flow_ahu * self.air_cp * (self.T_ahu_set_del - self.T_Air)
        self.q_Ahu.alg = self.m_flow_ahu * self.air_cp * (self.T_ahu_set - self.T_Air)

        self.heat_roof.alg = k_amb_roof * (self.T_Roof - self.T_preTemRoof)
        self.heat_extWall.alg = k_amb_ext * (self.T_ExtWall - self.T_preTemWall)
        # self.heat_window.alg = k_amb_win * (self.T_Win - self.T_preTemWin)

        self.T_ExtWall_sur.alg = (coeff_dict['T_ext_sur']['T_Air'] * self.T_Air + coeff_dict['T_ext_sur']['T_ext'] * self.T_ExtWall +
                                  coeff_dict['T_ext_sur']['T_int'] * self.T_IntWall + coeff_dict['T_ext_sur']['T_roof'] * self.T_Roof +
                                  coeff_dict['T_ext_sur']['Q_RadSol'] * self.Q_RadSol + coeff_dict['T_ext_sur']['q_ig_rad'] * q_ig_rad +
                                  coeff_dict['T_ext_sur']['T_preTemWin'] * self.T_preTemWin)
        self.T_IntWall_sur.alg = (coeff_dict['T_int_sur']['T_Air'] * self.T_Air + coeff_dict['T_int_sur']['T_ext'] * self.T_ExtWall +
                                  coeff_dict['T_int_sur']['T_int'] * self.T_IntWall + coeff_dict['T_int_sur']['T_roof'] * self.T_Roof +
                                  coeff_dict['T_int_sur']['Q_RadSol'] * self.Q_RadSol + coeff_dict['T_int_sur']['q_ig_rad'] * q_ig_rad +
                                  coeff_dict['T_int_sur']['T_preTemWin'] * self.T_preTemWin)
        self.T_Roof_sur.alg = (coeff_dict['T_roof_sur']['T_Air'] * self.T_Air + coeff_dict['T_roof_sur']['T_ext'] * self.T_ExtWall +
                               coeff_dict['T_roof_sur']['T_int'] * self.T_IntWall + coeff_dict['T_roof_sur']['T_roof'] * self.T_Roof +
                               coeff_dict['T_roof_sur']['Q_RadSol'] * self.Q_RadSol + coeff_dict['T_roof_sur']['q_ig_rad'] * q_ig_rad +
                               coeff_dict['T_roof_sur']['T_preTemWin'] * self.T_preTemWin)
        self.T_Win_sur.alg = (coeff_dict['T_win_sur']['T_Air'] * self.T_Air + coeff_dict['T_win_sur']['T_ext'] * self.T_ExtWall +
                              coeff_dict['T_win_sur']['T_int'] * self.T_IntWall + coeff_dict['T_win_sur']['T_roof'] * self.T_Roof +
                              coeff_dict['T_win_sur']['Q_RadSol'] * self.Q_RadSol + coeff_dict['T_win_sur']['q_ig_rad'] * q_ig_rad +
                              coeff_dict['T_win_sur']['T_preTemWin'] * self.T_preTemWin)


        self.T_Air.ode = (1 / c_air) * ((self.T_Roof_sur - self.T_Air) * k_roof_air
                                        + (self.T_ExtWall_sur - self.T_Air) * k_ext_air
                                        + (self.T_IntWall_sur - self.T_Air) * k_int_air
                                        + (self.T_Win_sur - self.T_Air) * k_win_air
                                        + (self.T_Floor - self.T_Air) * k_floor_air
                                        + (heat_humans_conv + heat_devices_conv + heat_lights_conv) * self.fac_IG_air
                                        + self.m_flow_ahu * self.air_cp * (self.T_ahu_set - self.T_Air)
                                        + (self.T_Tabs - self.T_Air) * k_tabs_air
                                        + self.Q_RadSol_air * self.fac_sol_air
                                        )

        if has_roof:
            self.T_Roof.ode = (1 / self.CRoof) * ((self.T_Roof_sur - self.T_Roof) * k_roof
                                                  + (self.T_preTemRoof - self.T_Roof) * k_amb_roof)
        if has_floor:
            self.T_Floor.ode = (1 / self.CFloor) * ((self.T_Air - self.T_Floor) * k_air_floor
                                                    + (self.T_ExtWall_sur - self.T_Floor) * k_ext_floor
                                                    + (self.T_Win_sur - self.T_Floor) * k_win_floor
                                                    # + (T_Win - self.T_Floor) * k_win_floor
                                                    + (self.T_Roof_sur - self.T_Floor) * k_roof_floor
                                                    + (self.T_IntWall_sur - self.T_Floor) * k_int_floor
                                                    + split_floor_sol * self.Q_RadSol * self.fac_sol_floor
                                                    + split_floor_ig * q_ig_rad * self.fac_IG_floor)
        else:
            self.T_Floor.ode = 0 * self.T_Air  # not physical, just here to keep agent config the same, even when there is no floor

        self.T_ExtWall.ode = (1 / self.CExt) * ((self.T_ExtWall_sur - self.T_ExtWall) * k_ext
                                                + (self.T_preTemWall - self.T_ExtWall) * k_amb_ext)

        self.T_IntWall.ode = (1 / self.CInt) * ((self.T_IntWall_sur - self.T_IntWall) * k_int)
        self.T_Tabs.ode = (1 / c_tabs) * ((self.T_Air - self.T_Tabs) * k_tabs_air + self.Q_Tabs_set_del)
        # self.T_Tabs.ode = (1 / c_tabs) * ((self.T_Air - self.T_Tabs) * k_tabs_air +
        #                                   self.Q_Tabs_set)

        # endregion

        # region Define ae
        self.Q_Ahu.alg = self.m_flow_ahu * self.air_cp * (self.T_ahu_set - (
                0.95 * self.T_Air + 1.05 * self.T_amb) / 2)  # heat recovery in AHU

        self.Q_ahu_abs.alg = ca.fabs(self.Q_Ahu.sym)
        self.Q_tabs_abs.alg = ca.fabs(self.Q_Tabs_set_del.sym)
        self.P_el_c.alg = (ca.fabs(self.Q_Ahu.sym) + ca.fabs(self.Q_Tabs_set_del.sym)) / self.COP
        self.E_stored.alg = ((self.T_Air*c_air + self.T_Roof*CRoof + self.T_Floor*CFloor
                              + self.T_ExtWall*CExt + self.T_IntWall*CInt + self.T_Tabs*c_tabs)
                              / (3600 * 1000))  # stored energy in kWh

        # endregion

        comp1 = ca.if_else(self.mode.sym > 0, -inf, 0)
        comp3 = ca.if_else(self.mode.sym > 0, 0, inf)

        mode_constraint_tabs = (comp1, self.Q_Tabs_set, comp3)
        mode_constraint_ahu = (comp1, self.Q_Ahu, comp3)

        # mode_constraint_tabs = self.mode.sym*(-inf, self.Q_Tabs_set, 0) + (1-self.mode.sym)*(0, self.Q_Tabs_set, inf)
        # self.test_constr.alg = self.mode.sym*1 + (1-self.mode.sym)*2
        # if self.config.mode == 1:
        #     ...
        # else:
        #     ...

        # region Constraints: List[(lower bound, function, upper bound)]
        self.constraints = [
            # soft constraints
            # T_air
            (-inf, self.T_Air - self.T_slack, self.T_upper),
            (self.T_lower, self.T_Air + self.T_slack, inf),
            (0, self.T_slack, inf),
            # q_tabs
            (0, self.Q_tabs_slack1, inf),
            (0, self.Q_tabs_slack2, inf),
            (0, self.Q_tabs_abs, inf),
            (0, self.Q_Tabs_set + self.Q_tabs_slack1 - self.Q_tabs_slack2, 0),
            mode_constraint_tabs,
            # q_ahu
            (0, self.Q_ahu_slack1, inf),
            (0, self.Q_ahu_slack2, inf),
            (0, self.Q_ahu_abs, inf),
            (0, self.Q_Ahu + self.Q_ahu_slack1 - self.Q_ahu_slack2, 0),
            # mode_constraint_ahu
        ]
        # endregion

        # region Objective function
        objective = sum(
            [
                (self.s_T / (self.s_T + self.s_Pel)) * self.T_slack ** 2,
                (self.r_pel / 3600) * (self.s_Pel / (self.s_T + self.s_Pel)) * (
                        self.Q_tabs_slack1 + self.Q_tabs_slack2 + self.Q_ahu_slack1 + self.Q_ahu_slack2) / self.COP / 1000,
            ]
        )
        # endregion

        return objective
