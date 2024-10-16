import json
import os
import utilities.fmu_handler as fmu_handler
from utilities.pickle_handler import *
import datetime
from utilities.modelica_parser import parse_modelica_record
from local.utils.check_generate import check_and_generate_file


class DisturbanceGenerator:
    def __init__(self, setup):
        """
        Initialize FMU and load mapping
        """
        """ Load Simulation Setup """

        # Get the directory of the current script and
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Get the base directory of the project
        project_base_dir = os.path.abspath(os.path.join(script_dir, '..'))

        with open(setup, 'r') as f:
            self.setup = json.load(f)

        self.days = self.setup['days']
        sim_tolerance = self.setup['sim_tolerance']  # tolerance
        start_time = self.setup['start_time']  # start time
        stop_time = 3600 * 24 * self.days + start_time # stop time
        self.step_size = self.setup['step_size']  # step size in s
        self.save_name = self.setup['save_name']

        path_mapping = os.path.join(project_base_dir, self.setup['path_mapping'])
        path_fmu = os.path.join(project_base_dir, self.setup['path_fmu'])

        self.path_record = self.setup['path_zone_record']
        nOrientations = int(parse_modelica_record(self.path_record)['nOrientations'])
        has_floor = parse_modelica_record(self.path_record)['AFloor'] > 0

        # fmu setup
        self.fmu = fmu_handler.fmu_handler(start_time=start_time,
                                           stop_time=stop_time,
                                           step_size=self.step_size,
                                           sim_tolerance=sim_tolerance,
                                           fmu_file=path_fmu,
                                           instanceName='fmu2')

        with open(path_mapping, 'r') as f:
            mapping = json.load(f)

        # adjust Q_RadSol_or_i in mapping
        mapping['disturbances'] = {k: v for k, v in mapping['disturbances'].items() if not k.startswith('Q_RadSol_or_')}
        for i in range(1, nOrientations + 1):
            #mapping['disturbances'][f'Q_RadSol_or_{i}'] = f'thermalZone1.ROM.radHeatSol[{i}].Q_flow'
            mapping['disturbances'][f'Q_RadSol_or_{i}'] = f'multizone.zone[1].ROM.radHeatSol[{i}].Q_flow'
        # adjust T_Floor in mapping states
        mapping['states'] = {k: v for k, v in mapping['states'].items() if k != "T_Floor"}
        if has_floor:
            #new_state = {"T_Floor": "thermalZone1.ROM.floorRC.thermCapExt[1].T"}
            new_state = {"T_Floor": "multizone.zone[1].ROM.floorRC.thermCapExt[1].T"}
            mapping['states'].update(new_state)

        # Save the updated mapping_disturbance configuration
        with open(path_mapping, 'w') as f:
            json.dump(mapping, f, indent=4)

        self.vars = self.fmu.find_vars('multizone.zone[1].weaBus')
        self.vars_dict = {}
        for key in mapping.keys():
            for var in mapping[key].keys():
                self.vars.append(mapping[key][var])
                self.vars_dict.update({var: mapping[key][var]})

        self.inv_dict = {v: k for k, v in self.vars_dict.items()}

    def perform_initial_simulation(self):
        """
        Simulate Fmu and save relevant disturbances as pandas dataframe
        :return:
        """
        self.fmu.setup()
        #TODO: config this initial value as a config to let fmu run!!
        self.fmu.set_variables({'T_in': 328.15})
        self.fmu.initialize()
        finished = False
        init_df = False
        df_interval = 1 * self.fmu.step_size  # data storage interval
        while not finished:
            # read variables
            res = self.fmu.read_variables(self.vars)
            # store data in dataframe
            if not init_df:
                df = pd.DataFrame(res, index=[0])
                init_df = True
            else:
                if self.fmu.current_time % df_interval == 0:
                    df = df._append(pd.DataFrame(res, index=[0]), ignore_index=True)
            print(str(self.fmu.current_time / 3600) + ' hours')
            finished = self.fmu.do_step()

        # close fmu
        print('finished, closing fmu')
        self.fmu.close()
        self.disturbances = df
        self.disturbances = self.disturbances.rename(columns=self.inv_dict)
        self.disturbances['Q_RadSol'] = self.disturbances.filter(like='Q_RadSol_or_').sum(axis=1)

        write_pickle(self.save_name, self.disturbances)

    def generate_boundaries(self, LB_emp=290.15, LB_use=293.15, UB_emp=299.15, UB_use=295.15,
                            m_flow_ahu_emp=12000 * 1 / 3600 * 1.224, m_flow_ahu_use=12000 * 3 / 3600 * 1.224, opening=7,
                            closing=17, appendix=''):
        """
        Generate Boundaries for Room temperature
        :return:
        """

        start = datetime.datetime(year=2018, month=1, day=1)
        time_list = [start]
        LB_list = [LB_emp]
        UB_list = [UB_emp]
        m_flow_ahu_list = [m_flow_ahu_emp]
        numSteps = len(self.disturbances)
        for i in range(1, numSteps):
            time_list.append(start + datetime.timedelta(seconds=i * self.step_size))
            if datetime.date.weekday(time_list[i]) < 5 and time_list[i].hour >= opening and time_list[
                i].hour < closing + 1:
                LB_list.append(LB_use)
                UB_list.append(UB_use)
                m_flow_ahu_list.append(m_flow_ahu_use)
            else:
                LB_list.append(LB_emp)
                UB_list.append(UB_emp)
                m_flow_ahu_list.append(m_flow_ahu_emp)
        ComfortCon = pd.DataFrame({'UB': UB_list, 'LB': LB_list, 'm_flow_ahu': m_flow_ahu_list}, index=time_list)

        self.disturbances.index = ComfortCon.index

        self.disturbances['T_Air_UB' + appendix] = ComfortCon['UB']
        self.disturbances['T_Air_LB' + appendix] = ComfortCon['LB']
        self.disturbances['m_flow_ahu' + appendix] = ComfortCon['m_flow_ahu']

    def create_disturbances(self):
        """
        Try to load pickle file and simulate a new one otherwise
        :return:
        """

        if check_and_generate_file(self.save_name):
            self.perform_initial_simulation()
        else:
            self.disturbances = read_pickle(self.save_name)

       #try:
        #    self.disturbances = read_pickle(self.save_name)
        #except:
       #     self.perform_initial_simulation()
        # Generate Boundaries for TZ_1
        self.generate_boundaries(LB_emp=self.setup["T_LB_emp"],
                                 LB_use=self.setup["T_LB_use"],
                                 UB_emp=self.setup["T_UB_emp"],
                                 UB_use=self.setup["T_UB_use"],
                                 m_flow_ahu_emp=self.setup["m_flow_ahu_emp"],
                                 m_flow_ahu_use=self.setup["m_flow_ahu_use"],
                                 opening=self.setup["opening"],
                                 closing=self.setup["closing"],
                                 appendix='')

        self.create_relaxed_bounds()
        # save disturbance file
        write_pickle(self.save_name, self.disturbances)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_base_dir = os.path.abspath(os.path.join(script_dir, '..'))
        csv_path = os.path.join(project_base_dir, 'predictor', "Disturbances.csv")
        self.disturbances.to_csv(csv_path)

        return csv_path

    def create_relaxed_bounds(self, rel=6):
        """

        :param rel: relaxation in hours
        :return:
        """

        self.generate_boundaries(LB_emp=self.setup["T_LB_emp"],
                                 LB_use=self.setup["T_LB_use"],
                                 UB_emp=self.setup["T_UB_emp"],
                                 UB_use=self.setup["T_UB_use"],
                                 m_flow_ahu_emp=self.setup["m_flow_ahu_emp"],
                                 m_flow_ahu_use=self.setup["m_flow_ahu_use"],
                                 opening=self.setup["opening"] - rel / 2,
                                 closing=self.setup["closing"] + rel / 2,
                                 appendix='_rel')

        self.disturbances['T_Air_UB_rel'] = self.disturbances['T_Air_UB_rel'].rolling(
            window=int(rel * 60 / 5), center=True).mean()
        self.disturbances['T_Air_UB_rel'] = self.disturbances['T_Air_UB_rel'].bfill()
        self.disturbances['T_Air_UB_rel'] = self.disturbances['T_Air_UB_rel'].ffill()
        self.disturbances['T_Air_LB_rel'] = self.disturbances['T_Air_LB_rel'].rolling(
            window=int(rel * 60 / 5), center=True).mean()
        self.disturbances['T_Air_LB_rel'] = self.disturbances['T_Air_LB_rel'].bfill()
        self.disturbances['T_Air_LB_rel'] = self.disturbances['T_Air_LB_rel'].ffill()


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    Dist = DisturbanceGenerator(setup=r"setup_disturbances.json")
    Dist.create_disturbances()
