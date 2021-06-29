import numpy as np
import scipy.special

aminoacids = 'ACDEFGHIKLMNPQRSTVWY'
_aminoacids_set = set(aminoacids)

def isvalidaa(string):
    "returns true if string is composed only of characters from the standard amino acid alphabet"
    return all(c in _aminoacids_set for c in string)

def coincidence_probability(array):
    """
    Calculates probability that two distinct elements of a list are the same.

    Note: this is also known as the Simpson or Hunter-Gaston index
    """
    array = np.asarray(array)
    _, counts = np.unique(array, return_counts=True)
    # 2*(n choose 2) = n * (n-1)
    return np.sum(counts*(counts-1))/(array.shape[0]*(array.shape[0]-1))


from functools import reduce
import pandas as pd

def multimerge(dfs, on, suffixes=None):
    """Merge multiple dataframes on a common column.

    Provides support for custom suffixes"""
    if suffixes:
        dfs_new = []
        for df, suffix in zip(dfs, suffixes):
            df = df.set_index(on)
            dfs_new.append(df.add_suffix('_'+suffix))
        return reduce(lambda left, right: pd.merge(left, right,
                                                   right_index=True, left_index=True,
                                                   how='outer'),
                      dfs_new)
    return reduce(lambda left, right: pd.merge(left, right, on), dfs)
