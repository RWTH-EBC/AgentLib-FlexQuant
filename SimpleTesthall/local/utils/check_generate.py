import hashlib
import os
import pathlib
import json
def hash_conditions(building, simulation):
    # generate hash value
    conditions_str = str(building) + str(simulation)
    return hashlib.md5(conditions_str.encode('utf-8')).hexdigest()

def check_and_generate_file(disturbances_file):
    hash_file = disturbances_file + ".hash"  # store hash file
    parent_directory = pathlib.Path(__file__).parent.parent
    setup_main = os.path.normpath(os.path.join(parent_directory, "config_main.json"))
    with open(setup_main, 'r') as f:
        setup = json.load(f)
    setup_building = setup['building_info']
    setup_building_str = json.dumps(setup_building, sort_keys=True)

    setup_disturbances = os.path.normpath(os.path.join(parent_directory, "predictor","setup_disturbances.json"))
    with open(setup_disturbances, 'r') as f:
        setup = json.load(f)
    setup_sim_disturbances = str(setup['days'])+str(setup['sim_tolerance'])+str(setup['start_time'])+str(setup['step_size'])

    # 计算当前条件的哈希值
    current_hash = hash_conditions(setup_building_str, setup_sim_disturbances)

    # Check if the hash file exists
    if os.path.exists(hash_file):
        with open(hash_file, 'r') as f:
            previous_hash = f.read()
        # If the hash values are the same, the conditions haven't changed
        if previous_hash == current_hash:
            print(f"Conditions have not changed, skipping disturbances.csv generation")
            return False
        else:
            print(f"Conditions have changed, regenerating disturbances.csv")
    else:
        print(f"First time generating disturbances.csv")

    # Write the current hash value to the hash file
    with open(hash_file, 'w') as f:
        f.write(current_hash)

    return True