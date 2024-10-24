import json
import pathlib
import os
import re


def find_radiator_type(min_demand_heat):
    # According to the guide from the manufacturer, using the conversion factors (with an average exponent of 1.3),
    # calculate the adjusted heating output under other operating conditions compared to the standard 75/65/20.
    # (Note: slight performance deviations may occur due to the exponent.)
    # The minimal standard heat output (EN 442  75/65/20) is calculated as: Heat demand * conversion factors.
    # Currently, this only supports converting from radiator type EN442 75/65/20 to 55/45/20, operating room temperature of 22°C

    # import all radiator JSON data
    parent_directory = pathlib.Path(__file__).parent.parent
    path_data_75_65_20 = os.path.normpath(os.path.join(parent_directory, f"DataResource/Template/data_radiatortype_75_65_20.json"))
    path_data_55_45_20 = os.path.normpath(os.path.join(parent_directory, f"DataResource/Template/data_radiatortype_55_45_20.json"))
    path_data_vol = os.path.normpath(os.path.join(parent_directory, f"DataResource/Template/data_radiator_volume.json"))
    path_data_weight = os.path.normpath(os.path.join(parent_directory, f"DataResource/Template/data_radiator_weight.json"))
    with open(path_data_75_65_20, "r") as f:
        data_75_65_20 = json.load(f)
    with open(path_data_55_45_20, "r") as f:
        data = json.load(f)
    with open(path_data_vol, "r") as f:
        data_vol = json.load(f)
    with open(path_data_weight, "r") as f:
        data_weight = json.load(f)

    min_value = float('inf')
    min_standard_heat = min_demand_heat * 2.15  # factor for heat demand with room temperature 22°C
    radiator_info = None
    # find corresponding radiator type 75/65/20
    for height, main_values in data_75_65_20.items():
        for type, sub_values in main_values.items():
            if type == "exponent":
                continue  # Skip the "exponent" key during the main iteration
            for length, heatpower in sub_values.items():
                if heatpower > min_standard_heat and heatpower < min_value:
                    min_value = heatpower
                    radiator_info = {
                        "height": height,
                        "type": type,
                        "length": length
                    }

    # find heat power and exponent in operating heating condition 55/45/20
    radiator_info["norm_power"]  = data.get(radiator_info["height"], {}).get(radiator_info["type"], {}).get(radiator_info["length"], None)
    radiator_info["exponent"] = data.get(radiator_info["height"], {}).get("exponent", {}).get(radiator_info["type"], None)

    # find water volume
    if radiator_info["height"] in data_vol:
        vol_dic = data_vol[radiator_info["height"]]
        # 查找类型
        for key in vol_dic:
            types = [t.strip() for t in key.split(",")]
            if radiator_info["type"] in types:
                radiator_info["water_vol"] = vol_dic[key]
    # find radiator weight
    if radiator_info["height"] in data_weight:
        radiator_info["weight"] = data_weight.get(radiator_info["height"], {}).get(radiator_info["type"], {}).get(radiator_info["length"], None)

    return radiator_info

#TODO: add select weight
def create_radiator_record(radiator):
    """
    using a template raditaor record to create a new one depending on given radiator information
    and stored it in DataResource/data
    :param radiator: a dictionary including all required radiator information
    :return:
    """
    record_template ={}

    lengh = float(radiator['length']) * 0.001
    record_template['NominalPower'] = round(radiator['norm_power'] / lengh, 2)
    record_template['RT_nom'] = "{328.15, 318.15, 293.15}"  # currently constant
    record_template['PressureDrop'] = 1017878.0 # currently constant
    record_template['Exponent'] = radiator['exponent']
    record_template['VolumeWater'] = radiator['water_vol']
    record_template['MassSteel'] = round(radiator['weight'] / lengh, 2)
    record_template['DensitySteel'] = 7900.0 # currently constant
    record_template['CapacitySteel'] = 551.0 # currently constant
    record_template['LambdaSteel'] = 60.0 # currently constant
    # ## editing radiator type
    type_number = ''.join(re.findall(r'\d+', radiator['type']))
    # ## in aixlib no type 33 support, if using type 33 transfer it to 32
    if type_number == 33:
        type_number == 32
    record_template['Type'] =f"AixLib.Fluid.HeatExchangers.Radiators.BaseClasses.RadiatorTypes.PanelRadiator{type_number}"
    record_template['length'] = round(lengh, 2)  # [m]
    record_template['height'] = round(float(radiator['height'])*0.001, 2)  # [m]

    radiator_name = f"Radiator_type{type_number}_{radiator['norm_power']}"
    parent_directory = pathlib.Path(__file__).parent.parent
    rel_path = f"DataResource/data/{radiator_name}.mo"
    abs_path = os.path.normpath(os.path.join(parent_directory,rel_path))
    with open(abs_path,'w') as f:
        f.write(format_radiator_record(record_template, radiator_name))

    return rel_path


def format_radiator_record(record,radiator_name):
    record_lines = []

    record_lines.append("within Test_proj.Test_zone.Test_building_DataBase;")
    record_lines.append(f"record {radiator_name}")
    record_lines.append('  "-"')
    record_lines.append("  extends AixLib.DataBase.Radiators.RadiatorBaseDataDefinition(")

    for key, value in list(record.items())[:-1]:  # Add all but the last item
        record_lines.append(f"      {key}={value},")

    # Add the last key-value pair with the closing )
    last_key = list(record.items())[-1][0]
    last_value = list(record.items())[-1][1]
    record_lines.append(f"      {last_key}={last_value})")  # Last item ends with ')'

    record_lines.append("  annotation ();")
    record_lines.append(f"end {radiator_name};")

    return "\n".join(record_lines)


if __name__ == '__main__':
    # Example usage
    heat_demand = 2374  # Replace with the value you want to search for

    result = find_radiator_type(heat_demand)
    create_radiator_record(result)
    # Display the result
    if result:
        print(f"Minimum value greater than {heat_demand}: {result['norm_power']}")
        print(f"Corresponding info: main_key={result['height']}, sub_key={result['type']}, size_key={result['length']}")
    else:
        print(f"No value greater than {heat_demand} found.")

