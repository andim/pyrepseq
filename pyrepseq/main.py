import numpy as np
import scipy.special

def coincidence_probability(array):
    """
    Calculates probability that two distinct elements of a list are the same.
    """
    array = np.asarray(array)
    _, counts = np.unique(array, return_counts=True)
    return 2*np.sum(scipy.special.binom(counts[counts>1], 2))/(array.shape[0]*(array.shape[0]-1))
