import numpy as np

from scipy.spatial.distance import squareform

from Levenshtein import hamming as hamming_distance
from Levenshtein import distance as levenshtein_distance


def pdist(strings, metric=None, dtype=np.uint8, **kwargs):
    """Pairwise distances between strings.
        (equivalent to scipy.spatial.distance.pdist)

    Parameters
    ----------
    strings : iterable of strings
        An m-length iterable.
    metric : function, optional
        The distance metric to use. Default: Levenshtein distance.
    dtype : np.dtype
        data type of the distance matrix, default: np.uint8

    Returns
    -------
    Y : ndarray
        Returns a condensed distance matrix Y.  For
        each :math:`i` and :math:`j` (where :math:`i<j<m`),where m is the number
        of original observations. The metric ``dist(u=X[i], v=X[j])``
        is computed and stored in entry 
        ``m * i + j - ((i + 2) * (i + 1)) // 2``.
    """
    if metric is None:
        metric = levenshtein_distance
    strings = list(strings)
    m = len(strings)
    dm = np.empty((m * (m - 1)) // 2, dtype=dtype)
    k = 0
    for i in range(0, m-1):
        for j in range(i+1, m):
            dm[k] = metric(strings[i], strings[j], **kwargs)
            k += 1
    return dm

def cdist(stringsA, stringsB, metric=None, dtype=np.uint8, **kwargs):
    """Pairwise distances between strings in two sets.
        (equivalent to scipy.spatial.distance.cdist)

    Parameters
    ----------
    stringsA : iterable of strings
        An mA-length iterable.
    stringsB : iterable of strings
        An mB-length iterable.
    metric : function, optional
        The distance metric to use. Default: Levenshtein distance.
    dtype : np.dtype
        data type of the distance matrix, default: np.uint8

    Returns
    -------
    Y : ndarray
        A :math:`m_A` by :math:`m_B` distance matrix is returned.
        For each :math:`i` and :math:`j`, the metric
        ``dist(u=XA[i], v=XB[j])`` is computed and stored in the
        :math:`ij` th entry.
    """
    if metric is None:
        metric = levenshtein_distance
    stringA = list(stringsA)
    stringB = list(stringsB)
    mA = len(stringA)
    mB = len(stringB)

    dm = np.empty((mA, mB), dtype=dtype)
    for i in range(0, mA):
        for j in range(0, mB):
            dm[i, j] = metric(stringA[i], stringB[j], **kwargs)
    return dm
