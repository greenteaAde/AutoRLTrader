import numpy as np


def change_format(num_str:str):
    try:
        if '.' in num_str:
            return format(np.float32(num_str), '.2f')
        else:
            return format(np.int32(num_str), ',d')
    except:
        return num_str

