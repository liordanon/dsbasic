import numpy as np

def rolling_window(array, window_size, cycle = False):

    array = np.asarray(array)

    # if cyclic rolling window (len(array) == len(rolling_window(array)))
    if cycle :
        array = np.append(array , array[:window_size - 1])

    # calculate shape
    length = array.size
    shape = (length - window_size + 1, window_size)

    # calculate stride
    strides = array.strides*2

    return np.lib.stride_tricks.as_strided(array, shape, strides)



