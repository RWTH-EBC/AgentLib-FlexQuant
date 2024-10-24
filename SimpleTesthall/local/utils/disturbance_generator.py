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

    def generate_index(self):
        """
        Generate Boundaries for Room temperature
        :return:
        """

        start = datetime.datetime(year=2018, month=1, day=1)
        time_list = [start]

        numSteps = len(self.disturbances)
        for i in range(1, numSteps):
            time_list.append(start + datetime.timedelta(seconds=i * self.step_size))

        self.disturbances.index = time_list

    def create_disturbances(self):
        """
        Try to load pickle file and simulate a new one otherwise
        :return:
        """

        if check_and_generate_file(self.save_name, "disturbance"):
            self.perform_initial_simulation()
        else:
            self.disturbances = read_pickle(self.save_name)

        self.generate_index()

        # save disturbance file
        write_pickle(self.save_name, self.disturbances)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_base_dir = os.path.abspath(os.path.join(script_dir, '..'))
        csv_path = os.path.join(project_base_dir, 'predictor', "Disturbances.csv")
        self.disturbances.to_csv(csv_path)

        return csv_path


if __name__ == '__main__':

    Dist = DisturbanceGenerator(setup=r"setup_disturbances.json")
    Dist.create_disturbances()
