from teaser.project import Project
import os
import re
from local.utils.radiator_model_temp import template_radiator
import pathlib
import numpy as np
import shutil
from ebcpy import FMU_API, TimeSeriesData
from dymola.dymola_interface import DymolaInterface
import json
from local.utils.select_radiator import create_radiator_record,find_radiator_type


class gen_building:
    def __init__(self,setup,
                 export_prj_path=r"DataResource"):
        self.setup = setup
        self.path_AixLib = setup['AixLibPath']
        self.work_dir_path = setup['Dymola_Work_Dir']
        self.methode =setup["methode"]
        self.prj_name = setup["project_name"]
        self.tz_name = setup["tz_name"]
        # local
        parent_directory = pathlib.Path(__file__).parent.parent

        self.export_prj_path = os.path.normpath(os.path.join(parent_directory, export_prj_path))
        self.weather_file_path = os.path.normpath(os.path.join(parent_directory, setup["weather_file_path"]))
        self.path_to_model = os.path.join(self.export_prj_path,
                                     self.prj_name,
                                     self.tz_name,
                                     f"{self.tz_name}.mo")
        self.path_to_model_max_hd = os.path.join(self.export_prj_path,
                                     self.prj_name,
                                     self.tz_name,
                                     f"{self.tz_name}_max_heatdemand.mo")
        self.path_to_model_w_rad = os.path.join(self.export_prj_path,
                                     self.prj_name,
                                     self.tz_name,
                                     f"{self.tz_name}_with_radiator.mo")

        self.usage = setup["usage"]
        self.year_of_construction = setup["year_of_construction"]
        self.height_of_floors = setup["height_of_floors"]
        self.net_leased_area = setup["net_leased_area"]
        # construction_type = setup["construction_type"]
        self.ROM_elements = setup["ROM_elements"]
        self.standard_norm_temp = setup['standard_outside_temp']
        self.location = setup['location']
        self.t_outside = setup['standard_outside_temp']


    def thermal_zone_from_teaser(self):
        """
        creating building by TEASER
        Returns:

        """
        prj = Project(load_data=True)
        prj.name = self.prj_name
        prj.add_residential(
            method=self.methode,
            usage=self.usage,
            name=self.tz_name,
            with_ahu=False,
            year_of_construction=self.year_of_construction,
            number_of_floors=1,
            height_of_floors=self.height_of_floors,
            net_leased_area=self.net_leased_area,
            construction_type=None)

        prj.used_library_calc = 'AixLib'
        prj.number_of_elements_calc = self.ROM_elements
        prj.weather_file_path = self.weather_file_path

        prj.calc_all_buildings()

        # If you only want to export one specific building, you can
        # pass over the internal_id of that building and only this model will be
        # exported. In this case we want to export all buildings to our home
        # directory, thus we are passing over None for it.

        path = prj.export_aixlib(
            internal_id=None,
            path=self.export_prj_path)
        return path

    def replace_t_set_idealheater(self,t_set=295.15):
        """
        Overwrite the internally interpreted 'TsetHeat' in TEASER.
        Args:
            t_set: set temperature for heating

        Returns:

        """
        path_to_tsetheat = os.path.join(self.export_prj_path,
                                         self.prj_name,
                                         self.tz_name,
                                         f"TsetHeat_{self.tz_name}.txt")
        # Read the file
        with open(path_to_tsetheat, 'r') as file:
            for line in file:
                parts = line.split()
                if len(parts) == 2:
                    old_value = parts[1] #Old value detected
                    break  # Exit after detecting the first value

            # Step 3: Replace the old value in the whole file
        with open(path_to_tsetheat, 'r') as file:
            data = file.read()

        new_t_set = f"{t_set}"
            # Replace all occurrences of the old value
        modified_data = data.replace(old_value, new_t_set)

            # Step 4: Write the modified data back to the file
        with open(path_to_tsetheat, 'w') as file:
            file.write(modified_data)

        return print("modified set temperature for ideal heater")

    def create_model_idealheater(self,baseACH=0.4):
        """
        creating a building model with ideal heater with desired ACH rate, adjusted from original TEASER output.
        Args:
            baseACH:

        Returns:

        """
        start_temp = self.setup['start_temp']

        self.modify_zone_para_record(need_idealheater=True,start_temp=start_temp, baseACH=baseACH)

        os.makedirs(os.path.dirname(self.path_to_model_max_hd), exist_ok=True)

        with open(self.path_to_model, 'r') as file:
            model_lines = file.readlines()

        inserted_io = False

        for i, line in enumerate(model_lines):
            if f"model {self.tz_name}" in line:
                line_model = f"model {self.tz_name}_max_heatdemand\n"
                model_lines[i] = line_model
            elif "T_start =" in line:
                # 找到目标行，将其中的数值部分替换为变量 start_temp 的值
                line_t_start = f"    T_start = {start_temp},\n"
                model_lines[i] = line_t_start
            elif "numZones" in line:
                lines_add_ACH = ["    use_MechanicalAirExchange=false,\n",
                            "    use_NaturalAirExchange=true,\n"]
                model_lines[i + 1:i + 1] = lines_add_ACH
            elif "AixLib.BoundaryConditions.WeatherData.ReaderTMY3 weaDat(" in line:
                line_t_standard = f"    TDryBulSou=AixLib.BoundaryConditions.Types.DataSource.Input,\n"
                model_lines[i + 1:i + 1] = line_t_standard
            # #### Adding input and output block and Inserting connection ####
            # Adding input/output blocks and inserting connections
            elif "equation" in line and not inserted_io:
                # Adding input and output blocks
                new_io_content = [
                    '  Modelica.Blocks.Interfaces.RealInput T_standard_outside\n',
                    '    "Input dry bulb temperature"\n',
                    '    annotation (Placement(transformation(extent={{-142,28},{-102,68}})));\n',
                    '  Modelica.Blocks.Interfaces.RealOutput PHeater1[1]\n',
                    '    "Power for heating"\n',
                    '    annotation (Placement(transformation(extent={{64,-14},{84,6}})));\n'
                ]
                model_lines[i:i] = new_io_content  # Insert input/output before 'equation'

                # Adding the connection blocks
                new_io_connection = [
                    '  connect(weaDat.TDryBul_in, T_standard_outside) annotation (Line(points={{-83,\n',
                    '          49},{-98,49},{-98,48},{-122,48}}, color={0,0,127}));\n',
                    '  connect(multizone.PHeater, PHeater1) annotation (Line(points={{51,-5},{57.5,\n',
                    '          -5},{57.5,-4},{74,-4}}, color={0,0,127}));\n'
                ]
                model_lines[i + len(new_io_content)+1:i + len(new_io_content)+1] = new_io_connection

                # Set flag to True to prevent further insertions
                inserted_io = True

            elif "end" in line:
                line_model = f"end {self.tz_name}_max_heatdemand;"
                model_lines[i] = line_model
                break

        with open(self.path_to_model_max_hd, 'w') as file:
            file.writelines(model_lines)

        ## remove custom para from teaser
        self.remove_cus_zone_declare(self.path_to_model_max_hd)

        ## add model to pacakge
        model_package_order = os.path.join(self.export_prj_path,
                                     self.prj_name,
                                     self.tz_name,
                                     "package.order")
        self.edit_model_package(model_package_order,f"{self.tz_name}_max_heatdemand")

        return print("Successfully created building model with ideal heater")


    def create_model_w_radiator(self,baseACH=0.4):
        start_temp = self.setup['start_temp']
        self.modify_zone_para_record(need_idealheater=False,start_temp=start_temp, baseACH=baseACH)

        os.makedirs(os.path.dirname(self.path_to_model_w_rad), exist_ok=True)
        inserted_io = False
        with open(self.path_to_model, 'r') as file:
            model_lines = file.readlines()

        for i, line in enumerate(model_lines):
            if f"model {self.tz_name}" in line:
                line_model = f"model {self.tz_name}_with_radiator\n"
                model_lines[i] = line_model
            elif "T_start =" in line:
                # 找到目标行，将其中的数值部分替换为变量 start_temp 的值
                line_t_start = f"    T_start = {start_temp},\n"
                model_lines[i] = line_t_start
            elif "numZones" in line:
                lines_add_ACH = ["    use_MechanicalAirExchange=false,\n",
                                 "    use_NaturalAirExchange=true,\n"]
                model_lines[i + 1:i + 1] = lines_add_ACH
            # #### Adding input and output block and Inserting connection ####
            # Adding input/output blocks and inserting connections
            elif "equation" in line and not inserted_io:
                # TODO: adjust model according to ROM_element
                # Adding input and output blocks
                new_io_content,new_io_connection = template_radiator()
                # Insert input/output before 'equation'
                model_lines[i:i] = new_io_content
                # Adding the connection blocks
                model_lines[i + len(new_io_content) + 1:i + len(new_io_content) + 1] = new_io_connection
                # Set flag to True to prevent further insertions
                inserted_io = True
            elif "end" in line:
                line_model = f"end {self.tz_name}_with_radiator;"
                model_lines[i] = line_model
                break

        with open(self.path_to_model_w_rad, 'w') as file:
            file.writelines(model_lines)

        ## remove custom para from teaser
        self.remove_cus_zone_declare(self.path_to_model_w_rad)

        ## add model to pacakge
        model_package_order = os.path.join(self.export_prj_path,
                                           self.prj_name,
                                           self.tz_name,
                                           "package.order")
        self.edit_model_package(model_package_order,f"{self.tz_name}_with_radiator")

        return print("Successfully created building model with radiator")

    def rad_record_to_model(self,
                                path_rad_record,
                                change_mflow=False):
        parent_directory = pathlib.Path(__file__).parent.parent
        path_rad_record = os.path.normpath(os.path.join(parent_directory, path_rad_record))
        m_flow = self.setup['m_flow_radiator']

        # adjust and move radiator record to DataBase
        with open(path_rad_record, 'r') as file:
            record_lines = file.readlines()
        for i,line in enumerate(record_lines):
            # edit first line in radiator record
            if "within" in line:
                line_path = f"within {self.prj_name}.{self.tz_name}.{self.tz_name}_DataBase;\n"
                record_lines[i] = line_path
            # get name
            if "record" in line:
                rad_type = line.split(" ")[-1]
                break
        path_to_DataBase= os.path.join(self.export_prj_path,
                                       self.prj_name,
                                       self.tz_name,
                                         f"{self.tz_name}_DataBase")
        shutil.copy2(path_rad_record, path_to_DataBase)
        record_in_Database = os.path.join(path_to_DataBase, f"{rad_type.strip()}.mo")
        with open(record_in_Database, 'w') as file:
            file.writelines(record_lines)

        # path to database package.order and add record to package.order
        model_package_order = os.path.join(path_to_DataBase,
                                           "package.order")
        self.edit_model_package(model_package_order,f"{rad_type}")

        # full path of current radiator record
        record_type_path= f"{self.prj_name}.{self.tz_name}.{self.tz_name}_DataBase.{rad_type}".strip()
        #os.makedirs(os.path.dirname(self.path_to_model_w_rad), exist_ok=True)

        # insert radiator record to building model
        with open(self.path_to_model_w_rad, 'r') as file:
            model_lines = file.readlines()

        for i, line in enumerate(model_lines):
            if "radiatorType" in line:
                line_record = f'    redeclare {record_type_path} radiatorType, N=3, calc_dT=AixLib.Fluid.HeatExchangers.Radiators.BaseClasses.CalcExcessTemp.exp) "Radiator"\n'
                model_lines[i] = line_record
            elif "AixLib.Fluid.Sources.MassFlowSource_T" in line and change_mflow is True:
                # Use regular expression to find and replace the m_flow value
                line_m_flow = re.sub(r'm_flow=\d+(\.\d+)?', f'm_flow={m_flow}', line)
                # Assign the updated line back to the list
                model_lines[i] = line_m_flow

        with open(self.path_to_model_w_rad, 'w') as file:
            file.writelines(model_lines)

        # translate building model to FMU after all
        ModelPath = f"{self.prj_name}.{self.tz_name}.{self.tz_name}_with_radiator".strip()
        self.fmu_export(AixLibPath=self.path_AixLib,ModelPath=ModelPath,FMUname =f"{self.tz_name}_with_radiator_{self.year_of_construction}_{self.location}")

    def modify_zone_para_record(self,need_idealheater: bool,start_temp=290.15,baseACH=0.4):
        if self.methode == 'urbanrenet':
            path_to_zone_para = os.path.join(self.export_prj_path,
                                     self.prj_name,
                                     self.tz_name,
                                     f"{self.tz_name}_DataBase",
                                     f"{self.tz_name}_Apartment1.mo")
        elif self.methode == 'tabula_de' or 'iwu':
            path_to_zone_para = os.path.join(self.export_prj_path,
                                             self.prj_name,
                                             self.tz_name,
                                             f"{self.tz_name}_DataBase",
                                             f"{self.tz_name}_SingleDwelling.mo")
        else:
            return print("Error: cannot find record file in given address")

        with open(path_to_zone_para, 'r') as file:
            lines = file.readlines()
            # 新内容，需要插入的两行
        add_para = [
            "    heaLoadFacOut=0,\n",
            "    heaLoadFacGrd=0);\n"  # 注意这里的 ';' 改为 ')'
        ]

        # 找到要插入的位置
        for i, line in enumerate(lines):
            if "T_start =" in line:
                # 找到目标行，将其中的数值部分替换为变量 start_temp 的值
                new_line = f"    T_start = {start_temp},\n"
                lines[i] = new_line
            elif need_idealheater is False and "HeaterOn" in line:
                lines[i] = line.replace("true", "false")
            elif "useConstantACHrate" in line:
                lines[i] = line.replace("false", "true")
            elif "baseACH" in line:
                # 找到目标行，将其中的数值部分替换为变量 start_temp 的值
                new_line = f"    baseACH = {baseACH},\n"
                lines[i] = new_line
            elif "TThresholdCooler = " in line:
                # 替换 ');' 为 ','，然后插入新的两行
                lines[i] = line.replace(");", ",")  # 保证这里是 ',' 而不是多余的换行符
                lines[i + 1:i + 1] = add_para  # 插入新内容时避免重复换行符
                break

            # 将修改后的内容写回文件
        with open(path_to_zone_para, 'w') as file:
            file.writelines(lines)

        return print(f"modified zonePara record, set ideal heater as {need_idealheater} ")

    def remove_cus_zone_declare(self, model_path):

        with open(model_path, 'r') as file:
            lines = file.readlines()

        # 标记是否处于 'zone(ROM(...))' 块中
        inside_zone_rom = False
        cleaned_lines = []

        # 逐行遍历文件内容
        for line in lines:
            if 'zone(ROM(' in line:
                #
                inside_zone_rom = True
            if inside_zone_rom:
                # 在 'zone(ROM(...))' 块中，直到遇到闭合的 ')'
                if '))))),' in line:
                    inside_zone_rom = False  # 结束忽略块
                continue  # 跳过该行，不加入 cleaned_lines
            else:
                # 不在 'zone(ROM(...))' 块中，保留该行
                cleaned_lines.append(line)

        # 将修改后的内容写回文件
        with open(model_path, 'w') as file:
            file.writelines(cleaned_lines)

    def edit_model_package(self, model_package_order,adding_model_name):

        adding_model_name = f"{adding_model_name}\n"
        add_order = True
        with open(model_package_order, 'r') as p:
            lines = p.readlines()
            if adding_model_name in lines:
                add_order = False
        if add_order:
            with open(model_package_order, 'a') as p:
                p.write(adding_model_name)


    def find_max_power(self,
            working_directory=None,
            n_cpu=1,
            log_fmu=True,
            n_sim=1,
            output_interval=3600,
            standard_outside_temp=264.15
    ):

        """
        Exports an FMU from the building model with an ideal heater, runs the FMU,
        and reads the results to find the minimal heat demand required to maintain
        the desired room temperature under standard outside temperature for this location.
        The FMU file 'building_max_heatdemand.fmu' is stored within the building package.

        Args:
            working_directory: dymola working directory
            n_cpu:  Number of processes to use
            log_fmu: Whether to get the FMU log output
            n_sim: Number of simulations to run
            output_interval: Output interval / step size of the simulation
            standard_outside_temp: standard outside temp at this location

        Returns:
            mininal required heat demand in W

        """

        # General settings
        if working_directory is None:
            working_directory = self.work_dir_path
        # translate model.mo to FMU
        model_full_path= f"{self.prj_name}.{self.tz_name}.{self.tz_name}_max_heatdemand".strip()
        FMUOutputPath =  os.path.join(self.export_prj_path,
                                     self.prj_name,
                                     self.tz_name)
        self.fmu_export(AixLibPath=self.path_AixLib,ModelPath=model_full_path, FMUOutputPath=FMUOutputPath,
                        FMUname=f"{self.tz_name}_max_heatdemand")

        # Simulation API Instantiation
        model_name = os.path.join(self.export_prj_path,
                                     self.prj_name,
                                     self.tz_name,
                                     f"{self.tz_name}_max_heatdemand.fmu")
        fmu_api = FMU_API(model_name=model_name,
                          working_directory=working_directory,
                          n_cpu=n_cpu,
                          log_fmu=log_fmu)

        # Change the simulation settings:
        simulation_setup = {"start_time": 0,
                            "stop_time": 31536000,
                            "output_interval": output_interval}
        fmu_api.set_sim_setup(sim_setup=simulation_setup)
        # Add Standard outside temperature as Inputs
        time_index = np.arange(
            fmu_api.sim_setup.start_time,
            fmu_api.sim_setup.stop_time,
            fmu_api.sim_setup.output_interval
        )
        T_standard_outside = standard_outside_temp
        df_inputs = TimeSeriesData({"T_standard_outside": T_standard_outside}, index=time_index)
        #  Results to store
        fmu_api.result_names = ["PHeater1[1]"]
        print("Results that will be stored", fmu_api.result_names)
        # Execute simulation
        results = fmu_api.simulate(inputs=df_inputs)
        # Closing
        fmu_api.close()

        results_pheat = results['PHeater1[1]']
        # drop results at beginning because it is too high
        filtered_data = results_pheat[results_pheat.index >= 432000]
        max_pheat = filtered_data.max()
        print(f"Heat power {filtered_data.max().values[0]} W is needed to keep room temperature")

        return max_pheat

    def fmu_export(self, AixLibPath,ModelPath,FMUname, ProjPath=None, WorkDirPath=None, FMUOutputPath=None):
        """
        :param AixLibPath: path of your local AixLibPath, end with package.mo
        :param ModelPath: full name of to be opened model, example
                        "AixLib.Systems.EONERC_MainBuilding.Examples.RoomModels.Ashrae140_ideal_heater"
        :param ProjPath: path of generated building model project,
                        default local/DataResource/building_projects/prj_name/package.mo
        :param WorkDirPath: work directory for dymola, all support data for FMU translation will be located here.
        :param FMUOutputPath: output path of generated FMU, default local/fmu
        """

        if ProjPath is None:
            ProjPath = os.path.join(self.export_prj_path,self.prj_name,"package.mo")
        if WorkDirPath is None:
            WorkDirPath = self.work_dir_path
        if FMUOutputPath is None:
            parent_directory = pathlib.Path(__file__).parent.parent
            FMUOutputPath = os.path.normpath(os.path.join(parent_directory, "fmu"))

        dymola = DymolaInterface()
        dymola.ExecuteCommand(f'cd("{WorkDirPath}")')

        if dymola.openModel(path=AixLibPath, changeDirectory=False):
            print("Sucessfully loaded Aixlib")
        if dymola.openModel(path=ProjPath, changeDirectory=False):
            print("Sucessfully loaded building model")
        # ModelPath= "Test_inter_teaser.Test_EFH_urbanrenet2.Test_EFH_urbanrenet2"

        print("Translating to FMU...")
        result = dymola.translateModelFMU(modelToOpen=ModelPath, modelName=FMUname, fmiType="cs", fmiVersion="2")

        if result:
            fmu_temp_path = os.path.join(WorkDirPath, FMUname + '.fmu')
            shutil.copy2(fmu_temp_path, FMUOutputPath)
            print(f"FMU successfully created at {FMUOutputPath}")
        else:
            print("Failed to generate FMU")

        dymola.close()

    def reloc_zone_record(self):
        """
        move generated record to "DataResource/data"
        Returns:

        """
        if self.methode == 'urbanrenet':
            path_to_zone_para = os.path.join(self.export_prj_path,
                                     self.prj_name,
                                     self.tz_name,
                                     f"{self.tz_name}_DataBase",
                                     f"{self.tz_name}_Apartment1.mo")
        elif self.methode == 'tabula_de' or 'iwu':
            path_to_zone_para = os.path.join(self.export_prj_path,
                                             self.prj_name,
                                             self.tz_name,
                                             f"{self.tz_name}_DataBase",
                                             f"{self.tz_name}_SingleDwelling.mo")
        else:
            return print("Error: cannot find record file in given address")
        parent_directory = pathlib.Path(__file__).parent.parent
        aim_path = os.path.normpath(os.path.join(parent_directory, "DataResource","data"))
        # New file name for the copied file
        rename_record = f"{self.tz_name}_with_radiator_{self.year_of_construction}_{self.location}.mo"  # You can change this to any name you want

        # Combine aim_path with new file name to specify the new file location
        aim_path = os.path.join(aim_path, rename_record)

        # Copy the file and rename it
        shutil.copy2(path_to_zone_para, aim_path)

if __name__ == '__main__':
    setup =r"local\config_main.json"
    with open(setup, 'r') as f:
        setup = json.load(f)

    setup_building = setup['building_info']
    standard_outside_temp = setup_building['standard_outside_temp']
    building_model = gen_building(setup_building)
    building_model.thermal_zone_from_teaser()

    # ### figure out max. heat demand with standard outside temperature
    # TODO: select radiator type after that
    building_model.replace_t_set_idealheater(t_set=295.15)
    building_model.create_model_idealheater(baseACH=0.4)
    heat_demand = building_model.find_max_power(n_cpu=1, log_fmu=False, n_sim=1, output_interval=3600, standard_outside_temp=standard_outside_temp)
    # ### end finding max. Power
    # ### generate model with radiator and export FMU
    building_model.create_model_w_radiator(baseACH=0.4)
    path_record = create_radiator_record(find_radiator_type(heat_demand))
    #path_record = create_radiator_record(find_radiator_type(2373))
    building_model.rad_record_to_model(path_rad_record=path_record,change_mflow=True)
    building_model.reloc_zone_record()

    print("Successfully exported Building Model!")

