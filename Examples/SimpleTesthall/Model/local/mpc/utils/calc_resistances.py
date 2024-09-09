from sympy import *


def calc_resistances(tz_par, split_sol, split_ig):
    """Calculates the coefficients for the algebraic equations of the surface temperatures.

    """
    # inputs
    T_preTemWin, Q_RadSol, q_ig_rad = symbols('T_preTemWin, Q_RadSol, q_ig_rad')
    # states
    T_Air = symbols('T_Air')
    T_ext = symbols('T_ext')
    T_int = symbols('T_int')
    # T_floor = symbols('T_floor')
    T_roof = symbols('T_roof')
    # surface temps
    T_win_sur = symbols('T_win_sur')
    T_ext_sur = symbols('T_ext_sur')
    T_int_sur = symbols('T_int_sur')
    T_roof_sur = symbols('T_roof_sur')

    # parameters
    # split factors
    split_win_ig, split_win_sol = split_ig['win'], split_sol['win']
    split_ext_ig, split_ext_sol = split_ig['ext'], split_sol['ext']
    split_int_ig, split_int_sol = split_ig['int'], split_sol['int']
    # split_floor_ig, split_floor_sol = symbols('split_floor_ig, split_floor_sol')
    split_roof_ig, split_roof_sol = split_ig['roof'], split_sol['roof']

    # thermal transmittances
    # roof
    k_air_roof = tz_par['hConRoof'] * tz_par['ARooftot']
    k_ext_roof = tz_par['hRad'] * min(tz_par['AExttot'],
                                 tz_par['ARooftot'])
    k_int_roof = tz_par['hRad'] * min(tz_par['AInttot'],
                                 tz_par['ARooftot'])
    k_amb_roof = 1 / (1 / ((tz_par['hConRoofOut'] + tz_par['hRadRoof']) * tz_par['ARooftot']) + tz_par['RRoofRem'])
    k_win_roof = min(tz_par['AWintot'],
                     tz_par['ARooftot']) * tz_par['hRad']
    k_roof = 1 / tz_par['RRoof']
    # ext
    k_air_ext = tz_par['hConExt'] * tz_par['AExttot']
    k_roof_ext = k_ext_roof
    k_int_ext = tz_par['hRad'] * min(tz_par['AExttot'], tz_par['AInttot'])
    k_amb_ext = 1 / (1 / ((tz_par['hConWallOut'] + tz_par['hRadWall']) * tz_par['AExttot']) + tz_par['RExtRem'])
    k_win_ext = min(tz_par['AExttot'], tz_par['AWintot']) * tz_par['hRad']
    k_ext = 1 / tz_par['RExt']
    # air
    k_roof_air = k_air_roof
    k_ext_air = k_air_ext
    k_int_air = tz_par['hConInt'] * tz_par['AInttot']
    k_win_air = tz_par['hConWin'] * tz_par['AWintot']
    # int
    k_air_int = k_int_air
    k_ext_int = k_int_ext
    k_roof_int = k_int_roof
    k_win_int = min(tz_par['AWintot'], tz_par['AInttot']) * tz_par['hRad']
    k_int = 1 / tz_par['RInt']
    # win
    k_roof_win = k_win_roof
    k_ext_win = k_win_ext
    k_int_win = k_win_int
    k_air_win = k_win_air
    k_amb_win = 1/(1/((tz_par['hConWinOut'] + tz_par['hRadWall']) * tz_par['AWintot']) + tz_par['RWin'])
    k_win_amb = k_amb_win

    # equations
    eq_win = Eq(k_win_amb * (T_preTemWin - T_win_sur) +
                k_win_air * (T_Air - T_win_sur) +
                k_win_ext * (T_ext_sur - T_win_sur) +
                k_win_int * (T_int_sur - T_win_sur) +
                # k_win_floor * (T_floor_sur - T_win_sur) +
                k_win_roof * (T_roof_sur - T_win_sur) +
                q_ig_rad * split_win_ig + split_win_sol * Q_RadSol, 0)
    eq_ext = Eq(k_ext * (T_ext - T_ext_sur) +
                k_ext_air * (T_Air - T_ext_sur) +
                k_ext_win * (T_win_sur - T_ext_sur) +
                k_ext_int * (T_int_sur - T_ext_sur) +
                # k_ext_floor * (T_floor_sur - T_ext_sur) +
                k_ext_roof * (T_roof_sur - T_ext_sur) +
                q_ig_rad * split_ext_ig + split_ext_sol * Q_RadSol, 0)
    eq_int = Eq(k_int * (T_int - T_int_sur) +
                k_int_air * (T_Air - T_int_sur) +
                k_int_ext * (T_ext_sur - T_int_sur) +
                k_int_win * (T_win_sur - T_int_sur) +
                # k_int_floor * (T_floor_sur - T_int_sur) +
                k_int_roof * (T_roof_sur - T_int_sur) +
                q_ig_rad * split_int_ig + split_int_sol * Q_RadSol, 0)
    # eq_floor = Eq(k_floor * (T_floor - T_floor_sur) +
    #               k_floor_air * (T_Air - T_floor_sur) +
    #               k_floor_ext * (T_ext_sur - T_floor_sur) +
    #               k_floor_int * (T_int_sur - T_floor_sur) +
    #               k_floor_win * (T_win_sur - T_floor_sur) +
    #               k_floor_roof * (T_roof_sur - T_floor_sur) +
    #               q_ig_rad * split_floor_ig + split_floor_sol * Q_RadSol, 0)
    eq_roof = Eq(k_roof * (T_roof - T_roof_sur) +
                 k_roof_air * (T_Air - T_roof_sur) +
                 k_roof_ext * (T_ext_sur - T_roof_sur) +
                 k_roof_int * (T_int_sur - T_roof_sur) +
                 # k_roof_floor * (T_floor_sur - T_roof_sur) +
                 k_roof_win * (T_win_sur - T_roof_sur) +
                 q_ig_rad * split_roof_ig + split_roof_sol * Q_RadSol, 0)

    # sol = solve([eq_win, eq_ext, eq_int, eq_floor, eq_roof],
    #             [T_win_sur, T_ext_sur, T_int_sur, T_floor_sur, T_roof_sur], check=False, rational=False, simplify=False)
    sol = solve([eq_win, eq_ext, eq_int, eq_roof],
                [T_win_sur, T_ext_sur, T_int_sur, T_roof_sur])

    # Extract coefficients from the solution
    coefficients = {}

    # Iterate over the equations in the solution
    for var_sur, eq in sol.items():
        # Extract coefficients for each symbolic variable
        coeffs = {}
        for var in [T_Air, T_ext, T_int, T_roof, Q_RadSol, q_ig_rad, T_preTemWin]:
            coeffs[str(var)] = float(eq.coeff(var))

        # Store coefficients for the current equation
        coefficients[str(var_sur)] = coeffs

    return coefficients


if __name__ == "__main__":
    from utils.modelica_parser import parse_modelica_record
    path_to_mos = r"D:\03_Python\Tools\wandb\mpc\ASHRAE140_900.mo"
    tz_par = parse_modelica_record(path_to_mos)
    tz_par['AWintot'] = 12
    tz_par['AInttot'] = 48
    tz_par['AExttot'] = 63.6
    tz_par['ARooftot'] = 48
    tz_par['AFloortot'] = 0
    split_ig_dict = {'int': 0.4,
                     'roof': 0.3,
                     'ext': 0.3,
                     'floor': 0,
                     'win': 0}
    split_sol_dict = {'int': 0.32,
                      'roof': 0.32,
                      'ext': 0.36,
                      'floor': 0,
                      'win': 0}
    calc_resistances(tz_par=tz_par, split_ig=split_ig_dict, split_sol=split_sol_dict)
