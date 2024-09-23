from sympy import *

def calc_resistances(tz_par, split_sol, split_ig):
    """Calculates the coefficients for the algebraic equations of the surface temperatures."""

    # inputs
    T_preTemWin, Q_RadSol, q_ig_rad = symbols('T_preTemWin, Q_RadSol, q_ig_rad')

    # states
    T_Air = symbols('T_Air')
    T_ext = symbols('T_ext')
    T_int = symbols('T_int')
    T_roof = symbols('T_roof')
    T_floor = symbols('T_floor')
    T_radiator_sur = symbols("T_radiator_sur")

    # surface temps
    T_win_sur = symbols('T_win_sur')
    T_ext_sur = symbols('T_ext_sur')
    T_int_sur = symbols('T_int_sur')
    T_roof_sur = symbols('T_roof_sur')
    T_floor_sur = symbols('T_floor_sur')
    T_rad = symbols('T_rad')
    T_rad_exp = symbols('T_rad_exp')

    # zu testen: erstmal direkt integer
    sigma, eps_rad_twostar, A_in_internal = 5.67e-8, 0.9, 10
    q_radiator_rad = sigma * eps_rad_twostar * A_in_internal * (T_radiator_sur - T_rad_exp)

    q_ig_rad_total = q_ig_rad + q_radiator_rad

    # parameters
    split_win_ig, split_win_sol = split_ig['win'], split_sol['win']
    split_ext_ig, split_ext_sol = split_ig['ext'], split_sol['ext']
    split_int_ig, split_int_sol = split_ig['int'], split_sol['int']
    split_floor_ig, split_floor_sol = split_ig['floor'], split_sol['floor']
    split_roof_ig, split_roof_sol = split_ig['roof'], split_sol['roof']

    k_air_roof = tz_par['hConRoof'] * tz_par['ARooftot']
    k_ext_roof = tz_par['hRad'] * min(tz_par['AExttot'], tz_par['ARooftot'])
    k_int_roof = tz_par['hRad'] * min(tz_par['AInttot'], tz_par['ARooftot'])
    k_win_roof = tz_par['hRad'] * min(tz_par['AWintot'], tz_par['ARooftot'])
    k_floor_roof = tz_par['hRad'] * min(tz_par['AFloortot'], tz_par['ARooftot'])
    k_roof = 1 / tz_par['RRoof']

    # Floor
    k_air_floor = tz_par['hConFloor'] * tz_par['AFloortot']
    k_ext_floor = tz_par['hRad'] * min(tz_par['AFloortot'], tz_par['AExttot'])
    k_win_floor = tz_par['hRad'] * min(tz_par['AFloortot'], tz_par['AWintot'])
    k_roof_floor = k_floor_roof
    k_int_floor = tz_par['hRad'] * min(tz_par['AFloortot'], tz_par['AInttot'])
    k_floor = 1 / tz_par['RFloor']

    # Ext_wall
    k_air_ext = tz_par['hConExt'] * tz_par['AExttot']
    k_roof_ext = k_ext_roof
    k_int_ext = tz_par['hRad'] * min(tz_par['AExttot'], tz_par['AInttot'])
    k_win_ext = min(tz_par['AExttot'], tz_par['AWintot']) * tz_par['hRad']
    k_ext = 1 / tz_par['RExt']
    k_floor_ext = k_ext_floor

    # Indoor Air
    k_roof_air = k_air_roof
    k_ext_air = k_air_ext
    k_int_air = tz_par['hConInt'] * tz_par['AInttot']
    k_win_air = tz_par['hConWin'] * tz_par['AWintot']
    k_floor_air = k_air_floor

    # Interior Walls
    k_ext_int = k_int_ext
    k_roof_int = k_int_roof
    k_win_int = min(tz_par['AWintot'], tz_par['AInttot']) * tz_par['hRad']
    k_int = 1 / tz_par['RInt']
    k_floor_int = k_int_floor

    # windows
    k_roof_win = k_win_roof
    k_ext_win = k_win_ext
    k_int_win = k_win_int
    k_amb_win = 1/(1/((tz_par['hConWinOut'] + tz_par['hRadWall']) * tz_par['AWintot']) + tz_par['RWin'])
    k_win_amb = k_amb_win
    k_floor_win = k_win_floor

    # equations
    eq_win = Eq(k_win_amb * (T_preTemWin - T_win_sur) +
                k_win_air * (T_Air - T_win_sur) +
                k_win_ext * (T_ext_sur - T_win_sur) +
                k_win_int * (T_int_sur - T_win_sur) +
                k_win_floor * (T_floor_sur - T_win_sur) +
                k_win_roof * (T_roof_sur - T_win_sur) +
                q_ig_rad_total * split_win_ig + split_win_sol * Q_RadSol, 0)
    eq_ext = Eq(k_ext * (T_ext - T_ext_sur) +
                k_ext_air * (T_Air - T_ext_sur) +
                k_ext_win * (T_win_sur - T_ext_sur) +
                k_ext_int * (T_int_sur - T_ext_sur) +
                k_ext_floor * (T_floor_sur - T_ext_sur) +
                k_ext_roof * (T_roof_sur - T_ext_sur) +
                q_ig_rad_total * split_ext_ig + split_ext_sol * Q_RadSol, 0)
    eq_int = Eq(k_int * (T_int - T_int_sur) +
                k_int_air * (T_Air - T_int_sur) +
                k_int_ext * (T_ext_sur - T_int_sur) +
                k_int_win * (T_win_sur - T_int_sur) +
                k_int_floor * (T_floor_sur - T_int_sur) +
                k_int_roof * (T_roof_sur - T_int_sur) +
                q_ig_rad_total * split_int_ig + split_int_sol * Q_RadSol, 0)
    eq_floor = Eq(k_floor * (T_floor - T_floor_sur) +
                  k_floor_air * (T_Air - T_floor_sur) +
                  k_floor_ext * (T_ext_sur - T_floor_sur) +
                  k_floor_int * (T_int_sur - T_floor_sur) +
                  k_floor_win * (T_win_sur - T_floor_sur) +
                  k_floor_roof * (T_roof_sur - T_floor_sur) +
                  q_ig_rad_total * split_floor_ig + split_floor_sol * Q_RadSol, 0)
    eq_roof = Eq(k_roof * (T_roof - T_roof_sur) +
                 k_roof_air * (T_Air - T_roof_sur) +
                 k_roof_ext * (T_ext_sur - T_roof_sur) +
                 k_roof_int * (T_int_sur - T_roof_sur) +
                 k_roof_floor * (T_floor_sur - T_roof_sur) +
                 k_roof_win * (T_win_sur - T_roof_sur) +
                 q_ig_rad_total * split_roof_ig + split_roof_sol * Q_RadSol, 0)
    eq_t_rad = Eq(split_ig['int'] * T_int_sur + split_ig['roof'] * T_roof_sur +
                  split_ig['ext'] * T_ext_sur + split_ig['floor'] * T_floor_sur +
                  split_ig['win'] * T_win_sur - T_rad, 0)

    # Solve the system of equations: surface temperature T_win_sur, T_ext_sur, T_int_sur, T_roof_sur, using symbolic variables
    sol = solve([eq_win, eq_ext, eq_int, eq_roof, eq_floor, eq_t_rad],
               [T_win_sur, T_ext_sur, T_int_sur, T_roof_sur, T_floor_sur, T_rad])#
    #initial_guess = [300, 295, 290, 285, 300, 295]
    #sol = nsolve([eq_win, eq_ext, eq_int, eq_roof, eq_floor, eq_t_rad],[T_win_sur, T_ext_sur, T_int_sur, T_roof_sur, T_floor_sur, T_rad],initial_guess)

    # Extract coefficients from the solution
    coefficients = {}

    # Iterate over the equations in the solution
    for var_sur, eq in sol.items():
        # Expand the equation to get individual components
        eq = expand(eq)

        # Extract coefficients for each symbolic variable
        coeffs = {}
        for var in [T_Air, T_ext, T_int, T_roof, T_floor, Q_RadSol, q_ig_rad, T_preTemWin, T_radiator_sur, T_rad_exp]:
            coeffs[str(var)] = eq.coeff(var)

        # Store coefficients for the current equation
        coefficients[str(var_sur)] = coeffs

    return coefficients


if __name__ == "__main__":
    from Examples.SimpleTesthall.Model.utils.modelica_parser import parse_modelica_record
    path_to_mos = r"D:\01_Git\02_Python\flexquant_branch\flexquant\Examples\SimpleTesthall\Model\local\mpc\ASHRAE140_900.mo"
    tz_par = parse_modelica_record(path_to_mos)
    tz_par['AWintot'] = 12
    tz_par['AInttot'] = 48
    tz_par['AExttot'] = 63.6
    tz_par['ARooftot'] = 48
    tz_par['AFloortot'] = 48
    split_ig_dict = {'int': 0.21857923497267756,
                     'roof': 0.21857923497267756,
                     'ext': 0.2896174863387978,
                     'floor': 0.21857923497267756,
                     'win': 0.05464480874316939}
    split_sol_dict = {'int': 0.23916292974588937,
                      'roof': 0.23916292974588937,
                      'ext': 0.23766816143497757,
                      'floor': 0.23916292974588937,
                      'win': 0.04484304932735426}
    coeff = calc_resistances(tz_par=tz_par, split_sol=split_sol_dict, split_ig=split_ig_dict)
    print(coeff)
