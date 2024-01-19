import numpy as np

def ensure_numpy(arr_like):
    module = type(arr_like).__module__
    if module == "pandas.core.series":
        return arr_like.to_numpy()
    if module == "numpy":
        return arr_like
    return np.array(arr_like)