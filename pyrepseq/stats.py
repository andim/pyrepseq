import numpy as np
import pandas as pd
import scipy.optimize
import scipy.special

def powerlaw(size=1, xmin=1.0, alpha=2.0):
    """ Draw samples from a discrete power-law.

    Uses an approximate transformation technique, see Eq. D6 in Clauset et al. arXiv 0706.1062v2 for details.
    """
    r = np.random.rand(int(size))
    return np.floor((xmin - 0.5)*(1.0-r)**(-1.0/(alpha-1.0)) + 0.5)

def mle_alpha(c, cmin=1.0, continuitycorrection=True):
    """Maximum likelihood estimate of the power-law exponent.
    
    see Eq. B17 in Clauset et al. arXiv 0706.1062v2
    """
    c = np.asarray(c)
    c = c[c>=cmin]
    if continuitycorrection:
        return 1.0 + len(c)/np.sum(np.log(c/(cmin-0.5)))
    return 1.0 + len(c)/np.sum(np.log(c/cmin))

def _discrete_loglikelihood(x, alpha, xmin):
    "Power-law loglikelihood"
    x = x[x>=xmin]
    n = len(x)
    return -n*np.log(scipy.special.zeta(alpha, xmin)) - alpha*np.sum(np.log(x))

def mle_alpha_discrete(c, cmin=1.0, **kwargs):
    """Maximum likelihood estimate of the power-law exponent for discrete data.

    Numerically maximizes the discrete loglikelihood.

    kwargs are passed to scipy.optimize.minimize_scalar.
    Default kwargs: bounds=[1.5, 4.5], method='bounded'
    """
    optkwargs = dict(bounds=[1.5, 4.5], method='bounded')
    optkwargs.update(kwargs)
    c = np.asarray(c)
    c = c[c>=cmin]
    result = scipy.optimize.minimize_scalar(lambda alpha: -_discrete_loglikelihood(c, alpha, cmin), **optkwargs)
    if not result.success:
        raise Exception('fitting failed')
    return result.x

def halfsample_sd(data, statistic, bootnum=1000):
    """
    Calculate an empirical estimate of the standard deviation of a statistic by sampling random halves of the data.
    """
    halfsampled = [statistic(np.random.choice(data,
                                            size=int(len(data)//2),
                                            replace=False))
                    for i in range(bootnum)]
    return np.std(halfsampled)/2**.5

def coincidence_probability(array):
    """
    Calculates probability that two distinct elements of an array are the same.

    If n_i are the counts of the i-th unique element and 
    N = \sum_i n_i the length of the array, then:

    \hat{p_C} = \sum_i n_i (n_i-1)/(N(N-1))
    
    Note: this is also known as the Simpson or Hunter-Gaston index
    """
    array = np.asarray(array)
    N = array.shape[0]
    _, counts = np.unique(array, return_counts=True)
    return np.sum(counts*(counts-1))/(N*(N-1))

def jaccard(A, B):
    """
    The Jaccard index is defined as 
    J(A, B) = |A intersection B| / |A union B|
    A, B: iterables (will be converted to sets)
    If A, B are pd.Series na values will be dropped first
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
    The overlap coefficient is defined as 
    O(A, B) = |A intersection B| / min(|A|, |B|)
    A, B: iterables (will be converted to sets)
    If A, B are pd.Series na values will be dropped first
    """
    if type(A) == pd.Series:
        A = A.dropna()
    if type(B) == pd.Series:
        B = B.dropna()
    A = set(A)
    B = set(B)
    return len(A.intersection(B))/min(len(A), len(B))
