import string
import os
import shutil
from dymola.dymola_interface import DymolaInterface
from utilities.modelica_parser import parse_modelica_record


class ModelicaProcessor:
    def __init__(self, setup):
        """
        """

        self.record_path = setup['RecordPath']
        self.template_model_path = setup['TemplateModelPath']
        self.output_name = setup['OutputName']
        self.output_fmu_dir = setup['OutputFMUDir']
        self.dymola_work_dir = setup['DymolaWorkDir']
        self.weather_data_path = setup['WeatherdataPath']

    def read_record_parameters(self):
        parameters = {}
        with open(self.record_path, 'r') as file:
            lines = file.readlines()
        for line in lines:
            try:
                line = line.strip()
                if not line or line.startswith('//'):
                    continue
                if '=' in line:
                    param, value = line.split('=', 1)
                    param = param.strip()
                    value = value.strip().rstrip(',;)')
                    parameters[param] = value
            except:
                continue
        return parameters

    def add_blocks_connections(self):
        """
        determines if new blocks and connections need to be added to a Modelica model based on
        certain parameters nOrientation and AFloor from a record file.

        :return:new block and connection in list
        """
        tz_par = parse_modelica_record(self.record_path)
        has_floor = tz_par['AFloor'] > 0
        new_block = []
        connection = []

        # add output if component floor exists
        if has_floor:
            new_block += [
                "  Modelica.Blocks.Interfaces.RealOutput T_Floor ""\n",
                "    annotation ();\n",
                "  Modelica.Blocks.Sources.RealExpression realExpression18(y=thermalZone1.ROM.floorRC.thermCapExt[1].T)\n",
                "    annotation ();\n"
            ]
            connection += [
                "  connect(realExpression18.y, T_Floor) annotation (Line());\n"
            ]

        # add solar radiation output according to nOrientations
        nOrientations = int(tz_par['nOrientations'])
        if nOrientations > 1:
            for i in range(1, nOrientations + 1):
                if i > 1:
                    new_block.append(
                        f"  Modelica.Blocks.Sources.RealExpression realExpression2{i}(y=thermalZone1.ROM.radHeatSol[{i}].Q_flow)\n"
                    )
                    new_block.append("    annotation ();\n")
                    connection.append(f"  connect(realExpression2{i}.y, multiSum.u[{i}]) annotation (Line());\n")

        return new_block, connection

    def update_model_and_package(self):
        """
        add new block and connection in model.mo based on template model "Template_Ashrae140_ideal_heater"
        update model name in .mo and save the edited version as a new .mo under given output name
        edit package.
        """
        # process model name
        new_block, new_connection = self.add_blocks_connections()
        tz_par = parse_modelica_record(self.record_path)
        nOrientations = int(tz_par['nOrientations'])
        model_path_parse = PathParse(self.template_model_path)
        templ_model_name = model_path_parse.get_model_basename()
        output_model_path = os.path.join(model_path_parse.parse_model_dir(), f'{self.output_name}.mo')
        package_order = os.path.join(model_path_parse.parse_model_dir(), 'package.order')

        with open(self.template_model_path) as f:
            system = f.readlines()

        index = 0
        for index, line in enumerate(system):
            if line.strip() == f"model {templ_model_name}":
                system[index] = f"model {self.output_name}\n"
            elif line.strip().startswith("Modelica.Blocks.Math.MultiSum"):
                system[index] = f"Modelica.Blocks.Math.MultiSum multiSum(nu={nOrientations})\n"
            elif line.strip() == "equation":
                break

        system[index:index] = new_block
        system[(index + len(new_block) + 1):(index + len(new_block) + 1)] = new_connection
        system[-1] = f"end {self.output_name};\n"

        with open(output_model_path, 'w') as f:
            f.writelines(system)

        print(f"created a new model named {self.output_name} in package")
        # add new model to package
        add_model = f"{self.output_name}\n"
        add_order = True

        with open(package_order, 'r') as p:
            lines = p.readlines()
            if add_model in lines:
                add_order = False
        if add_order:
            with open(package_order, 'a') as p:
                p.write(add_model)

    def insert_record_into_model(self, ModelPath=None, StoredPath=None):
        """
        insert record as zoneparam in model

        :param ModelPath: path of model.mo
        :param StoredPath: save path of the edited model.mo, if not given saved in the same file.
        :return:
        """
        if ModelPath is None:
            model_path_parse = PathParse(self.template_model_path)
            model_file_path = os.path.join(model_path_parse.parse_model_dir(), f'{self.output_name}.mo')
        else:
            model_file_path = ModelPath

        parameters = self.read_record_parameters()
        with open(model_file_path, 'r') as file:
            model_lines = file.readlines()

        i = 0
        while i < len(model_lines):
            if 'zoneParam(' in model_lines[i]:
                indent = model_lines[i][:len(model_lines[i]) - len(model_lines[i].lstrip())]
                j = i
                while j < len(model_lines) and ')' not in model_lines[j]:
                    model_lines[j] = ''
                    j += 1
                if j < len(model_lines):
                    model_lines[j] = f"{indent}zoneParam(\n"
                insertion_lines = [f"{indent}{param}={value},\n" for param, value in list(parameters.items())[:-1]]
                last_param, last_value = list(parameters.items())[-1]
                insertion_lines.append(f"{indent}{last_param}={last_value}),\n")
                model_lines[i + 1:i + 1] = insertion_lines
                break
            i += 1

        if StoredPath is None:
            with open(model_file_path, 'w') as file:
                file.writelines(model_lines)
        else:
            with open(StoredPath, 'w') as file:
                file.writelines(model_lines)
        print("record updated")

    def update_weather_data(self, ModelPath=None, StoredPath=None, WeatherPath=None):
        """
        updates the weather data file path in the Modelica model
        Args:
            ModelPath:
            StoredPath:
            WeatherPath:

        Returns:

        """
        if WeatherPath is None:
            weather_path = os.path.normpath(self.weather_data_path)
            weather_path = weather_path.replace("\\", "//")

        if ModelPath is None:
            model_path_parse = PathParse(self.template_model_path)
            model_file_path = os.path.join(model_path_parse.parse_model_dir(), f'{self.output_name}.mo')
        else:
            model_file_path = ModelPath

        with open(model_file_path) as f:
            model_lines = f.readlines()

        for index, line in enumerate(model_lines):
            if line.strip().startswith("filNam=ModelicaServices.ExternalReferences.loadResource("):
                model_lines[index + 1] = f'"{weather_path}"))\n'
                break

        if StoredPath is None:
            with open(model_file_path, 'w') as file:
                file.writelines(model_lines)
        else:
            with open(StoredPath, 'w') as file:
                file.writelines(model_lines)
        print("weather data updated")

    def export_fmu(self, WorkDir=None, ModelPath=None, FMUOutputPath=None):
        """
        generate FMU for co-simulation and save it under desired project path

        :param WorkDir: work directory for dymola, all support data for FMU translation will be located here.
        :param ModelPath: full name of to be opened model, example "AixLib.Systems.EONERC_MainBuilding.Examples.RoomModels.Ashrae140_ideal_heater"
        :param FMUOutputPath: output path of generated FMU

        :return:
        """

        dymola = DymolaInterface()
        if WorkDir is None:
            WorkDir = self.dymola_work_dir
        WorkDir = os.path.normpath(WorkDir).replace("\\", "/")

        dymola.ExecuteCommand(f'cd("{WorkDir}")')

        if ModelPath is None:
            model_path_parse = PathParse(self.template_model_path)
            OutputModelName = model_path_parse.change_model_name(self.output_name)
            FMUName = self.output_name
        else:
            model_path_parse = PathParse(ModelPath)
            OutputModelName = model_path_parse.parse_full_model_name()
            FMUName = model_path_parse.get_model_basename()

        AixLibdir = os.path.join(model_path_parse.parse_model_dir(), 'package.mo')
        AixLibdir = AixLibdir.replace("\\", "//")
        dymola.openModel(path=AixLibdir, changeDirectory=False)

        print("Translating to FMU...")
        result = dymola.translateModelFMU(modelToOpen=OutputModelName, modelName=FMUName, fmiType="cs", fmiVersion="2")

        if result:
            if FMUOutputPath is None:
                fmu_output_path = os.path.join(self.output_fmu_dir, FMUName + '.fmu')
            else:
                fmu_output_path = os.path.join(FMUOutputPath, FMUName + '.fmu')
            fmu_temp_path = os.path.join(WorkDir, FMUName + '.fmu')
            shutil.move(fmu_temp_path, fmu_output_path)
            print(f"FMU successfully created at {fmu_output_path}")
        else:
            print("Failed to generate FMU")

        dymola.close()


class PathParse:
    """
    convert the given model absolute path to model path/package path of AixLib Library
    """

    def __init__(self, ModelPath):
        self.normalized_path = os.path.normpath(ModelPath)
        self.path_parts = self.normalized_path.split(os.sep)
        try:
            self.aixlibindex = self.path_parts.index("AixLib")
        except ValueError:
            raise ValueError("The path does not contain 'AixLib'.")

    def parse_full_model_name(self):
        """
        parse full model name, exemple "AixLib.ThermalZones.ReducedOrder.Examples.ThermalZone"
        :return:
        """
        model_parts = self.path_parts[self.aixlibindex:]
        model_name = '.'.join(model_parts)
        if model_name.endswith('.mo'):
            model_name = model_name[:-3]

        return model_name

    def parse_model_dir(self):
        dir_package = self.path_parts[:-1]
        if len(dir_package) > 1:
            model_dir = os.path.join(*dir_package[1:])
            model_dir = f"{dir_package[0]}\\{model_dir}"
        else:
            model_dir = dir_package[0]
        return model_dir

    def get_model_basename(self):
        model_name = self.parse_full_model_name()
        return model_name.split('.')[-1]

    def change_model_name(self, new_basename):
        full_model_name = self.parse_full_model_name()
        model_parts = full_model_name.split('.')
        model_parts[-1] = new_basename
        new_full_model_name = '.'.join(model_parts)
        return new_full_model_name

def fmu_export(AixLibPath,ProjPath,WorkDirPath, ModelPath,FMUOutputPath):
        """
        generate FMU for co-simulation and save it under desired project path

        :param WorkDir: work directory for dymola, all support data for FMU translation will be located here.
        :param ModelPath: full name of to be opened model, example "AixLib.Systems.EONERC_MainBuilding.Examples.RoomModels.Ashrae140_ideal_heater"
        :param FMUOutputPath: output path of generated FMU

        :return:
        """

        dymola = DymolaInterface()
        dymola.ExecuteCommand(f'cd("{WorkDirPath}")')
        FMUName = "test_export"
        dymola.openModel(path=AixLibPath, changeDirectory=False)
        dymola.openModel(path=ProjPath, changeDirectory=False)
        #ModelPath= "Test_inter_teaser.Test_EFH_urbanrenet2.Test_EFH_urbanrenet2"
        print("Translating to FMU...")
        result = dymola.translateModelFMU(modelToOpen=ModelPath, modelName=FMUName, fmiType="cs", fmiVersion="2")

        if result:
            #FMUOutputPath = r"D:\sle-gzh\repo\optimization\SimpleTesthall\local\DataResource\export_test_temp\Test_inter_teaser\Test_EFH_urbanrenet2"
            fmu_temp_path = os.path.join(WorkDirPath, FMUName + '.fmu')
            shutil.move(fmu_temp_path, FMUOutputPath)
            print(f"FMU successfully created at {FMUOutputPath}")
        else:
            print("Failed to generate FMU")

        dymola.close()

if __name__ == "__main__":
    processor = ModelicaProcessor(setup=r'config_main.json')
    processor.update_model_and_package()
    processor.insert_record_into_model()
    processor.update_weather_data()
    processor.export_fmu()
