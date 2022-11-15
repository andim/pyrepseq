import numpy as np
import pandas as pd
import scipy.optimize
import scipy.special

def powerlaw_sample(size=1, xmin=1.0, alpha=2.0):
    """ Draw samples from a discrete power-law.

    Uses an approximate transformation technique, see Eq. D6 in Clauset et al. arXiv 0706.1062v2 for details.

    Parameters
    ----------
    size: number of values to draw
    xmin: minimal value
    alpha: power-law exponent

    Returns
    -------
    array of integer samples
    """
    r = np.random.rand(int(size))
    return np.floor((xmin - 0.5)*(1.0-r)**(-1.0/(alpha-1.0)) + 0.5)

def _discrete_loglikelihood(x, alpha, xmin):
    "Power-law loglikelihood"
    x = x[x>=xmin]
    n = len(x)
    return -n*np.log(scipy.special.zeta(alpha, xmin)) - alpha*np.sum(np.log(x))

def powerlaw_mle_alpha(c, cmin=1.0, method='exact', **kwargs):
    """Maximum likelihood estimate of the power-law exponent.

    
    for discrete counts.

    Parameters
    ----------
    c : counts
    cmin: only counts >= cmin are included in fit
    continuitycorrection: use continuitycorrection (more accurate for integer counts)
    method: one of ['simple', 'continuitycorrection', 'exact']
        'simple': Uses an analytical formula  that is exact in the continuous case
            (Eq. B17 in Clauset et al. arXiv 0706.1062v2)
        'continuitycorrection': applies a continuity correction to the analytical formula
        'exact': Numerically maximizes the discrete loglikelihood
    kwargs: dict
        passed on to `scipy.optimize.minimize_scalar`
        Default: bounds=[1.5, 4.5], method='bounded'

    Returns
    -------
    float
        estimated power-law exponent
    """
    if not method in ['simple', 'continuitycorrection', 'exact']:
        raise ValueError("Method not in ['simple', 'continuitycorrection', 'exact']")
    c = np.asarray(c)
    c = c[c>=cmin]
    if method == 'continuitycorrection':
        return 1.0 + len(c)/np.sum(np.log(c/(cmin-0.5)))
    if method == 'exact':
        optkwargs = dict(bounds=[1.5, 4.5], method='bounded')
        optkwargs.update(kwargs)
        result = scipy.optimize.minimize_scalar(lambda alpha: -_discrete_loglikelihood(c, alpha, cmin), **optkwargs)
        if not result.success:
            raise Exception('fitting failed')
        return result.x
    return 1.0 + len(c)/np.sum(np.log(c/cmin))


def pc(array):
    r"""Estimate the coincidence probability :math:`p_C` from a sample.
    :math:`p_C` is equal to the probability that two distinct sampled elements are the same.
    If :math:`n_i` are the counts of the i-th unique element and 
    :math:`N = \sum_i n_i` the length of the array, then:
    :math:`p_C = \sum_i n_i (n_i-1)/(N(N-1))`
    
    Note: This measure is also known as the Simpson or Hunter-Gaston index
    """
    array = np.asarray(array)
    N = array.shape[0]
    _, counts = np.unique(array, return_counts=True)
    return np.sum(counts*(counts-1))/(N*(N-1))

def jaccard_index(A, B):
    """
    Calculate the Jaccard index for two sets.

    This measure is defined  defined as 

    math:`J(A, B) = |A intersection B| / |A union B|`
    A, B: iterables (will be converted to sets). If A, B are pd.Series na values will be dropped first
    """
    if type(A) == pd.Series:
        A = A.dropna()
    if type(B) == pd.Series:
        B = B.dropna()
    A = set(A)
    B = set(B)
    return len(A.intersection(B))/(len(A.union(B)))

def overlap_coefficient(A, B):
    """
    Calculate the overlap coefficient for two sets.

    This measure is defined  defined as 
    :math:`O(A, B) = |A intersection B| / min(|A|, |B|)`

    A, B: iterables (will be converted to sets). If A, B are pd.Series na values will be dropped first
    """
    if type(A) == pd.Series:
        A = A.dropna()
    if type(B) == pd.Series:
        B = B.dropna()
    A = set(A)
    B = set(B)
    return len(A.intersection(B))/min(len(A), len(B))
