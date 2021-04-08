import Levenshtein
import numpy as np

def pdist(strings, metric=Levenshtein.distance, **kwargs):
    """Pairwise distances between strings.
        (equivalent to scipy.spatial.distance.pdist)

    Parameters
    ----------
    strings : iterable of strings
        An m-length iterable.
    metric : function, optional
        The distance metric to use.

    Returns
    -------
    Y : ndarray
        Returns a condensed distance matrix Y.  For
        each :math:`i` and :math:`j` (where :math:`i<j<m`),where m is the number
        of original observations. The metric ``dist(u=X[i], v=X[j])``
        is computed and stored in entry 
        ``m * i + j - ((i + 2) * (i + 1)) // 2``.
    """
    m = len(strings)
    dm = np.empty((m * (m - 1)) // 2, dtype=np.double)
    k = 0
    for i in range(0, m-1):
        for j in range(i+1, m):
            dm[k] = metric(strings[i], strings[j], **kwargs)
            k += 1
    return dm
