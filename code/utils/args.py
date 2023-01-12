def read_ARGS(path):
    import importlib
    spec = importlib.util.spec_from_file_location("config", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    ARGS = getattr(module,'ARGS')()
    return ARGS

def isPrimitive(cast_type):
    return cast_type in [int, float, str, bool]


def ismodule(class_, module_):
    return hasattr(module_, class_.__name__)    
    
def cast_bool(cast_from):
    if type(cast_from) is str:
        if cast_from == 'True':
            return True
        elif cast_from == "False":
            return False
    
    if type(cast_from) is bool:
        return cast_from
    
def cast_like(cast_from, cast_to, cast_candidate):
    cast_type = type(cast_to)
    
    if isPrimitive(cast_type):
        if cast_type is bool:
            return cast_bool(cast_from)
        else:
            return cast_type(cast_from)
    
    for cast_candidate_ in cast_candidate:
        if ismodule(cast_to ,cast_candidate_):
            return getattr(cast_candidate_, cast_from)
    
    raise LookupError(f"Can not find appropriate cast_candidate for {cast_from}")

'''
from typing import Tuple, Union
import yaml
def Refine_OptionalArgs_to_Template(
    config_path, 
    optional, 
    Template:Union[DLPG_ARGS_Template,None], 
    cast_candidate: Tuple = ()
    )->Union[DLPG_ARGS_Template,None]:
    
    # for _ in configs.keys(): # Ignore this line

    # Make configs with .yaml file
    with open(config_path.__str__()) as f:
        configs = yaml.load(f, Loader=yaml.Loader)["ARGS"]
    _field_defaults = Template._field_defaults
    
    # Assert 
    for key in configs.keys():
        assert key in _field_defaults.keys()
    # Override when arguments are specified
    i = 0
    while i < len(optional):
        if '=' in optional[i]:
            key,item = str.split(optional[i], '=')
            key = key.replace("-","")
            
            if not key in _field_defaults.keys():
                raise KeyError(f"Invalid Arguments Given: {key}")
            configs[key] = cast_like(item, _field_defaults[key], cast_candidate)
            i = i+1
            
        else:
            key = optional[i]
            key = key.replace("-","")
            item = optional[i+1]
            
            if not key in _field_defaults.keys():
                raise KeyError(f"Invalid Arguments Given: {key}")
            configs[key] = cast_like(item, _field_defaults[key], cast_candidate)
            i = i+2
            
    return Template(**configs)
'''