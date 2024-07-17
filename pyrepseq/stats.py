import numpy as np
from numpy import ndarray
import pandas as pd
from pandas import DataFrame
import scipy.optimize
import scipy.special
from scipy.spatial.distance import squareform
import warnings
import itertools
from pyrepseq.util import convert_tuple_to_dataframe_if_necessary, ensure_numpy
from typing import Iterable, Optional, Union

def powerlaw_sample(size=1, xmin=1.0, alpha=2.0):
    """Draw samples from a discrete power-law.

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
    return np.floor((xmin - 0.5) * (1.0 - r) ** (-1.0 / (alpha - 1.0)) + 0.5)

def subsample(counts, n):
    """Randomly subsample from a vector of counts without replacement.

    Parameters
    ----------
    counts : Vector of counts (integers) to randomly subsample from.
    n : Number of items to subsample from `counts`. Must be less than or equal
        to the sum of `counts`.

    Returns
    -------
    indices, counts: Subsampled vector of counts where the sum of the elements equals `n`
    """
    n = int(n)
    unpacked = np.concatenate([np.repeat(np.array(i,), count) for i, count in enumerate(counts)])
    sample = np.random.choice(unpacked, size=n, replace=False)
    unique, counts = np.unique(sample, return_counts=True)
    return unique, counts

def _discrete_loglikelihood(x, alpha, xmin):
    "Power-law loglikelihood"
    x = x[x >= xmin]
    n = len(x)
    return -n * np.log(scipy.special.zeta(alpha, xmin)) - alpha * np.sum(np.log(x))


def powerlaw_mle_alpha(c, cmin=1.0, method="exact", **kwargs):
    """Maximum likelihood estimate of the power-law exponent.

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
    estimated power-law exponent
    """
    if not method in ["simple", "continuitycorrection", "exact"]:
        raise ValueError("Method not in ['simple', 'continuitycorrection', 'exact']")
    c = np.asarray(c)
    c = c[c >= cmin]
    if method == "continuitycorrection":
        return 1.0 + len(c) / np.sum(np.log(c / (cmin - 0.5)))
    if method == "exact":
        optkwargs = dict(bounds=[1.5, 4.5], method="bounded")
        optkwargs.update(kwargs)
        result = scipy.optimize.minimize_scalar(
            lambda alpha: -_discrete_loglikelihood(c, alpha, cmin), **optkwargs
        )
        if not result.success:
            raise Exception("fitting failed")
        return result.x
    return 1.0 + len(c) / np.sum(np.log(c / cmin))


def pc_n(n):
    r"""Estimate the coincidence probability :math:`p_C` from sampled counts.
    :math:`p_C` is equal to the probability that two distinct sampled elements are the same.
    If :math:`n_i` are the counts of the i-th unique element and
    :math:`N = \sum_i n_i` the length of the array, then:
    :math:`p_C = \sum_i n_i (n_i-1)/(N(N-1))`

    Note: This measure is also known as the Simpson or Hunter-Gaston index

    Parameters
    ----------
    n : array-like
        list of counts
    """
    
    n = ensure_numpy(n)
    N = np.sum(n)
    return np.sum(n * (n - 1)) / (N * (N - 1))


def pc(array: Iterable, array2: Optional[Iterable] = None):
    r"""
    Estimate the coincidence probability :math:`p_C` from a sample.
    :math:`p_C` is equal to the probability that two distinct sampled elements are the same.
    If :math:`n_i` are the counts of the i-th unique element and
    :math:`N = \sum_i n_i` the length of the array, then:
    :math:`p_C = \sum_i n_i (n_i-1)/(N(N-1))`

    Note: This measure is also known as the Simpson or Hunter-Gaston index

    Parameters
    ----------
    array: Iterable
        Iterable of sampled elements
    array2: Optional[Iterable]
        Second Iterable of sampled elements: if provided probability of cross-coincidences is calculated as :math:`p_C = (\sum_i n_{1i} n_{2i}) / (N_1 N_2)`
    """
    array = convert_tuple_to_dataframe_if_necessary(array)
    array2 = convert_tuple_to_dataframe_if_necessary(array2)

    def convert_to_array(array: Union[Iterable, DataFrame]) -> ndarray:
        if not isinstance(array, DataFrame):
            return np.asarray(array)
        
        df = array.fillna("")
        unique_strings = df.apply(
            lambda row: ".".join(str(val) for val in row),
            axis=1
        )
        return unique_strings.to_numpy()
    
    array = convert_to_array(array)
    if array2 is None:
        N = array.shape[0]
        _, counts = np.unique(array, return_counts=True)
        return np.sum(counts * (counts - 1)) / (N * (N - 1))

    array2 = convert_to_array(array2)
    v, c = np.unique(array, return_counts=True)
    v2, c2 = np.unique(array2, return_counts=True)
    v_int, ind1_int, ind2_int = np.intersect1d(
        v, v2, assume_unique=True, return_indices=True
    )
    return np.sum(c[ind1_int] * c2[ind2_int]) / (len(array) * len(array2))

def pc_joint(df, on, df_2=None, gap_token='_'):
    """Joint coincidence probability estimator
    
    Parameters
    ----------
    df : pandas DataFrame
    on: list of strings
        columns on which to obtain a joint probability of coincidence
    df_2: None or pd.DataFrame
        second DataFrame for cross-coincidence calculations
    gap_token: string
        character to be added for feature concatenization

    Returns
    ----------
    float:
        pc computed on the concatenation of each specified column in on
    
    """
    
    if df_2 is None:
        return pc(df[on].apply(lambda x: gap_token.join(x.astype(str)), axis=1))
    return pc(df[on].apply(lambda x: gap_token.join(x.astype(str)), axis=1), df_2[on].apply(lambda x: gap_token.join(x.astype(str)), axis=1))
    
def pc_grouped_cross(df, by, on):
    """Cross-group coincidence probability estimator

    Parameters
    ----------
    df : pandas DataFrame
    by : mapping, function, label, or list of labels
      see pd.DataFrame.groupby
    on: list of strings
        columns on which to obtain a joint probability of coincidence

    Returns
    ----------
    pd.DataFrame:
        pc computed on the concatenation of each specified column in on
    
    """
    groups = sorted(list(df.groupby(by)))
    data = []
    index = []
    for ((name1, d1)), (name2, d2) in itertools.combinations(groups, 2):
        if type(on) == list:
            pc_cross_group = pc_joint(d1, on, d2)
        else:
            pc_cross_group = pc(d1[on], d2[on])
            
        index.append([name1, name2])
        data.append(pc_cross_group)      
    data = np.array(data)
    
    names = [name for name, dfg in groups]
    data_square = squareform(data)
    np.fill_diagonal(
        data_square, np.nan
    )
    return pd.DataFrame(data_square, index=names, columns=names)

def pc_conditional(df, by, on, group_weights=None):
    """Conditional coincidence probability estimator
    
    Parameters
    ----------
    df : pandas DataFrame
    by: list
        conditioning parameters used to group input dataframe
    on: string/list of strings
        column or columns to compute probability of coincidence or joint probability of coincidence on. If type(on) == list 
        then joint pc is computed on the concatenations of each specified column
    group_weights: array-like
        weight groups non-uniformly according to square of these values
    
    Returns
    ----------: 
    pandas DataFrame/float:
        pc of df[on] averaged over groups 
    """
    
    if type(by) == list and len(by) == 1:
        by = by[0]
        
    #Mask df entries where pc cannot be computed
    df = df.groupby(by).filter(lambda x: len(x) > 1)
    if len(df) < 2:
        return np.nan
        
    if type(on) == list:
        conditional_pcs = df.groupby(by).apply(lambda x: pc_joint(x, on))

    else:
        conditional_pcs = df.groupby(by).apply(lambda x: pc(x[on]))
        
    if group_weights is None:
        group_weights = np.ones(len(df[by].value_counts()))
    else:
        group_weights = np.asarray(group_weights)
    
    adjusted_group_weights = (group_weights**2)/np.sum(group_weights**2)
        
    return np.sum(adjusted_group_weights*conditional_pcs)

def varpc_n(n):
    "Variance estimator for Simpson's index"
    N = np.sum(n)
    p2_hat = np.sum(n * (n - 1)) / (N * (N - 1))
    p3_hat = np.sum(n * (n - 1) * (n - 2)) / (N * (N - 1) * (N - 2))
    beta = 2 * (2 * N - 3) / ((N - 2) * (N - 3))
    var = (
        4 * (N - 2) / (N * (N - 1)) * (1 + beta) * p3_hat
        - beta * p2_hat**2
        + 2 / (N * (N - 1)) * (1 + beta) * p2_hat
    )
    return var


def stdpc_n(n):
    "Std.dev. estimator for Simpson's index"
    
    return varpc_n(n)** 0.5

def stdpc(array):
    "Std.dev. estimator for Simpson's index"
    array = np.asarray(array)
    _, n = np.unique(array, return_counts=True)
    return stdpc_n(n)


def stdpc_joint(df, on, gap_token = '_'):
    "Std.dev. estimator for joint Simpson's index"

    return stdpc(df[on].apply(lambda x: gap_token.join(x.astype(str)), axis=1))

def chao1(counts):
    """Estimate richness from sampled counts.

    hatSchao1 = Sobs + f1^2/(2 f2)
    """
    
    f1 = counts[0]
    Sobs = np.sum(counts)

    if (len(counts) == 1) or (counts[1] == 0):
        return Sobs + (f1*(f1-1))/2

    f2 = counts[1]
    return Sobs + f1**2/(2*f2)

def var_chao1(counts):
    """Variance estimator for Chao1 richness."""
     
    f1 = counts[0]
    
    if len(counts) == 1:
        return np.nan
    if counts[1] == 0:
        return np.nan
    
    f2 = counts[1]
    ratio = f1 / f2
    return f2 * ((ratio / 4) ** 4 + ratio**3 + (ratio / 2) ** 2)

def chao2(counts, m):
    """Estimate richness from incidence data

    counts: incidence count vector
    m: number of replicates
    """
  
    q1 = counts[0]
    Sobs = np.sum(counts)

    if (len(counts) == 1) or (counts[1] == 0):
        return np.nan

    q2 = counts[1]
    return Sobs + q1**2/(2*q2) 

def var_chao2(counts, m):
    """Variance estimator for Chao2 richness.

    counts: incidence count vector
    m: number of replicates
    """
    
    q1 = counts[0]
    Sobs = np.sum(counts)

    if (len(counts) == 1) or (counts[1] == 0):
        return np.nan
    
    ratio = q1/q2
    return q2*(0.5*ratio**2+ratio**3+0.25*ratio**4)
        
        
def jaccard_index(A, B):
    """
    Calculate the Jaccard index for two sets.

    This measure is defined  defined as

    :math:`J(A, B) = |A \\cap B| / |A \\cup B|`

    A, B: iterables (will be converted to sets). If A, B are pd.Series na values will be dropped first
    """
    if type(A) == pd.Series:
        A = A.dropna()
    if type(B) == pd.Series:
        B = B.dropna()
    A = set(A)
    B = set(B)
    return len(A.intersection(B)) / (len(A.union(B)))


def overlap(A, B):
    """
    Calculate the number of overlapping elements of two sets.

    This measure is defined as
    :math:`|A \\cap B|`

    A, B: iterables (will be converted to sets). na values will be dropped first
    """
    if type(A) != pd.Series:
        A = pd.Series(A)
    if type(B) != pd.Series:
        B = pd.Series(B)
    A = A.dropna()
    B = B.dropna()
    A = set(A)
    B = set(B)
    return len(A.intersection(B))


def overlap_coefficient(A, B):
    """
    Calculate the overlap coefficient for two sets.

    This measure is defined as
    :math:`O(A, B) = |A \\cap B| / min(|A|, |B|)`

    A, B: iterables (will be converted to sets). na values will be dropped first
    """
    if type(A) != pd.Series:
        A = pd.Series(A)
    if type(B) != pd.Series:
        B = pd.Series(B)
    A = A.dropna()
    B = B.dropna()
    A = set(A)
    B = set(B)
    if len(A) == 0 or len(B) == 0:
        return np.nan
    
    return len(A.intersection(B)) / min(len(A), len(B))
