from pyrepseq import pdist, cdist
import numpy as np
import scipy.spatial.distance
import Levenshtein

def test_pdist():
    strings = ['abs', 'asdf', 'adfl']
    distances = pdist(strings)
    distances_scipy = scipy.spatial.distance.pdist(np.array(strings, dtype=str)[:, np.newaxis],
                                                   metric=lambda u, v: Levenshtein.distance(u[0], v[0]))
    assert all(distances == distances_scipy)

def test_cdist():
    strings = ['abs', 'asdf', 'adfl']
    distances = cdist(strings, strings)
    array = np.array(strings, dtype=str)[:, np.newaxis]
    distances_scipy = scipy.spatial.distance.cdist(array, array,
                                                   metric=lambda u, v: Levenshtein.distance(u[0], v[0]))
    assert (distances == distances_scipy).all()
