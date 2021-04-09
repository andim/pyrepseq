import numpy as np
import scipy.special

aminoacids = 'ACDEFGHIKLMNPQRSTVWY'

def coincidence_probability(array):
    """
    Calculates probability that two distinct elements of a list are the same.

    Note: this is also known as the Simpson or Hunter-Gaston index
    """
    array = np.asarray(array)
    _, counts = np.unique(array, return_counts=True)
    # 2*(n choose 2) = n * (n-1)
    return np.sum(counts*(counts-1))/(array.shape[0]*(array.shape[0]-1))
