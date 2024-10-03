import json
import yaml


def get_model_name_from_path(model_path):
    model_path = model_path.lower()
    if 'clip' in model_path:
        if '336' in model_path:
            return 'clip336'
        else:
            return 'clip'
    elif 'llava' in model_path:
        if '1.5' in model_path:
            if '13' in model_path:
                return 'llava1.5-13b'
            else:
                return 'llava1.5-7b'
        else:
            return 'llava'
    elif 'blip2' in model_path:
        return 'blip2'
    elif 'instruct' in model_path:
        return 'instructblip'
    

def read_file(path: str):
    if path.endswith('.txt'):
        with open(path) as f:
            return f.readlines()
    elif path.endswith('.json'):
        with open(path) as f:
            return json.load(f)
    elif path.endswith('.jsonl'):
        with open(path) as f:
            return [json.loads(q) for q in f]
    else:
        raise NotImplementedError("Only support .txt, .jsonl and .json files")
    
def make_descriptor_sentence(descriptor):
    if descriptor.startswith('a') or descriptor.startswith('an'):
        return f"which is {descriptor}"
    elif descriptor.startswith('has') or descriptor.startswith('often') or descriptor.startswith('typically') or descriptor.startswith('may') or descriptor.startswith('can'):
        return f"which {descriptor}"
    elif descriptor.startswith('used'):
        return f"which is {descriptor}"
    else:
        return f"which has {descriptor}"

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'



def load_yaml_parameters(filename, key):
    """
    Load parameters for a specific function from a YAML file.

    :param filename: The name of the YAML file containing the parameters.
    :param function_name: The name of the function whose parameters are to be loaded.
    :return: A dictionary containing the parameters, or None if the function is not found.
    """

    try:
        # Open the YAML file and load all contents
        with open(filename, 'r') as file:
            all_parameters = yaml.safe_load(file)
        
        # Retrieve the parameters for the specified function
        parameters = all_parameters.get(key)
        
        if parameters is None:
            print(f"No parameters found for target '{key}'.")
            return None
        
        return parameters
    except FileNotFoundError:
        print(f"The file '{filename}' was not found.")
        return None
    except yaml.YAMLError as exc:
        print(f"An error occurred while parsing the YAML file: {exc}")
        return None
