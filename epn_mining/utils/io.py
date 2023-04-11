import os
import copy
import pickle

def check_end_character(string, character):
    if string != '':
        if string[-1] != character:
            string = f'{string}{character}'
    return string

def check_underscore(string):
    """Assure string ends with an underscore

    Parameter
    ---------
    string:str

    Returns
    -------
    string:str
    """
    return check_end_character(string, "_")

def check_slash(string):
    """Assure string ends with a slash

    Parameter
    ---------
    string:str

    Returns
    -------
    string:str
    """
    return check_end_character(string, "/")

def set_state_name(state_prefix, variables):
    """Set the state_prefix variable

    Concatenates variables included in the `variables' list as suffixes to `state_prefix'.

    Parameters
    ----------
    state_prefix:str
        Main string descriptor
    variables:list
        List of parameters to be concatenated as suffix to state_prefix

    Returns
    -------
    state_prefix:str
        Updated state_prefix string
    """
    _prefix = copy.deepcopy(state_prefix)
    if isinstance(variables, list):
        for v in variables:
            if v != None:
                if isinstance(v, list):
                    for vv in v:
                        if _prefix == '':
                            _prefix = str(vv)
                        else:
                            _prefix += '_' + str(vv)
                else:
                    if _prefix == '':
                        _prefix = v
                    else:
                        _prefix += '_' + v
    else:
        _prefix = f"{state_prefix}_{variables}"

    return _prefix

def state_full_location(prefix:str,
                        variables:list,
                        suffix:str=None,
                        folder:str='states/',
                        extension='.pickle'):
    name = f'{set_state_name(prefix, variables)}'
    if suffix is not None:
        name += f'_{suffix}'
    return f'{check_slash(folder)}{name}{extension}'

def save(variable, data, protocol=pickle.HIGHEST_PROTOCOL, state_prefix='', folder='states/', verbose=False, return_filename=False):
    """Save variable's data state into a pickle file

    Parameter
    ---------
    variable:str
        Name of the attribute state to be stored
    data:obj
        Data to be serialized by pickle
    protocol:pickle.protocol
        Pickle security protocol (default: pickle.HIGHEST_PROTOCOL)
    state_prefix:str
        String identifier for this current state (prefix to variable) (default: '')
    folder:str
        Output folder (default: 'states/')
    Returns
    -------
    string:str
    """
    if not os.path.exists(folder):
        os.makedirs(folder)

    if variable not in ['epn_metadata', 'population']:
        filename = f'{folder}{check_underscore(state_prefix)}{variable}.pickle'
        with open(filename, 'wb') as f:
            pickle.dump(data, f, protocol)
    else:
        filename = f'{folder}{variable}.pickle'
        with open(filename, 'wb') as f:
            pickle.dump(data, f, protocol)

    if verbose:
        print(f"Saved to {filename}")

    if return_filename:
        return filename

def load(variable, state_prefix='', folder='states/'):
    if variable not in ['epn_metadata', 'population']:
        filename = f'{folder}{check_underscore(state_prefix)}{variable}.pickle'
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                try:
                    d = pickle.load(f)
                except:
                    d = None
                return d
        else:
            return None
    else:
        filename = f'{folder}{variable}.pickle'
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                try:
                    d = pickle.load(f)
                except:
                    d = None
                return d
        else:
            return None
