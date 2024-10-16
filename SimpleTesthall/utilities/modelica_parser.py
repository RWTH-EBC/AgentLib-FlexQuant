import string
import ast


def parse_modelica_record(path):
    constants = {}
    with open(path) as f:
        for i, line in enumerate(f):
            try:
                line = line.translate({ord(c): None for c in string.whitespace})[:-1]
                if line.endswith(')'):
                    line = line[:-1]
                line = line.replace("{", "[")
                line = line.replace("}", "]")
                name, value = line.split("=")
                if value == 'true':  # Evaluate Booleans
                    constants[name] = True
                elif value == 'false':
                    constants[name] = False
                elif value.startswith('['):
                    value = ast.literal_eval(value)  # Evaluate Lists
                    if len(value) == 1:
                        constants[name] = value[0]
                    else:
                        constants[name] = value
                else:
                    constants[name] = float(value)  # Evaluate Floats
            except:
                continue
        f.close()

    return constants


if __name__ == '__main__':
    parameters = parse_modelica_record(
        r"Examples\SimpleTesthall\local\mpc\ASHRAE140_900.mo")
    print(parameters)
