import ast
import string
def parse_modelica_record(path):
    constants = {}
    with open(path) as f:
        for i, line in enumerate(f):
            # Stop if the line contains 'annotation' to avoid parsing further
            if "annotation" in line:
                break
            try:
                # Translate to remove whitespaces and strip end-of-line
                line = line.translate({ord(c): None for c in string.whitespace})[:-1]
                # Stop processing if line ends with ");" to limit parsing scope
                if line.endswith(');'):
                    break
                # Remove trailing parenthesis from multiline records
                if line.endswith(')'):
                    line = line[:-1]
                # Replace braces to make lists compatible with Python
                line = line.replace("{", "[")
                line = line.replace("}", "]")
                # Split into name and value
                name, value = line.split("=")
                # Evaluate Booleans
                if value == 'true':
                    constants[name] = True
                elif value == 'false':
                    constants[name] = False
                # Evaluate Lists
                elif value.startswith('['):
                    value = ast.literal_eval(value)
                    if len(value) == 1:
                        constants[name] = value[0]
                    else:
                        constants[name] = value
                # Evaluate Floats or Strings
                else:
                    try:
                        constants[name] = float(value)
                    except ValueError:
                        # If it's not a float, treat it as a string
                        constants[name] = value
            except:
                continue

    return constants


def parse_rad_record(path):
    parsed_constants = parse_modelica_record(path)
    radiators_fac = {
        "SectionalRadiator": {
            "convective_fraction": 0.70,
            "radiative_fraction": 0.30
        },
        "PanelRadiator10": {
            "convective_fraction": 0.50,
            "radiative_fraction": 0.50
        },
        "PanelRadiator11": {
            "convective_fraction": 0.65,
            "radiative_fraction": 0.35
        },
        "PanelRadiator12": {
            "convective_fraction": 0.75,
            "radiative_fraction": 0.25
        },
        "PanelRadiator20": {
            "convective_fraction": 0.65,
            "radiative_fraction": 0.35
        },
        "PanelRadiator21": {
            "convective_fraction": 0.80,
            "radiative_fraction": 0.20
        },
        "PanelRadiator22": {
            "convective_fraction": 0.85,
            "radiative_fraction": 0.15
        },
        "PanelRadiator30": {
            "convective_fraction": 0.80,
            "radiative_fraction": 0.20
        },
        "PanelRadiator31": {
            "convective_fraction": 0.85,
            "radiative_fraction": 0.15
        },
        "PanelRadiator32": {
            "convective_fraction": 0.90,
            "radiative_fraction": 0.10
        },
        "ConvectorHeaterUncovered": {
            "convective_fraction": 0.95,
            "radiative_fraction": 0.05
        },
        "ConvectorHeaterCovered": {
            "convective_fraction": 1.00,
            "radiative_fraction": 0.00
        }
    }
    full_type = parsed_constants.get('Type', None)

    try:
        if full_type:
            # Extract the substring after the last period '.'
            radiator_type = full_type.split(".")[-1]

            # Check if this type exists in the radiators_fac dictionary
            if radiator_type in radiators_fac:
                parsed_fac = radiators_fac[radiator_type]

                # Add parsed_fac to parsed_constants with the key 'ratio'
                parsed_constants['rad_fac'] = parsed_fac['radiative_fraction']

    except Exception as e:
        print(f"Error processing radiator type: {e}")

    return parsed_constants

if __name__ == "__main__":
    parsed_constants = parse_rad_record(r'D:\sle-gzh\test_project\Test_simle_radiator\mpc\Radiator_Bathroom.mo')
    print(parsed_constants)