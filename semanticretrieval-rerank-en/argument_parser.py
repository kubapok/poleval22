def get_params_dict(params_str):
    params_dict =  {a.split('=')[0]:a.split('=')[1] for a in params_str.split(',') }
    params_dict['PARAM_K1'] = float(params_dict['PARAM_K1'])
    params_dict['PARAM_B'] = float(params_dict['PARAM_B'])
    params_dict['EPSILON'] = float(params_dict['EPSILON'])
    return params_dict
