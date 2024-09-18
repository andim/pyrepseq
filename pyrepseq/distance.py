from .io import aminoacids
from .stats import pc
import os.path
import itertools
import numpy as np
from Levenshtein import distance as levenshtein_distance
import pandas as pd
from pandas import DataFrame
from pyrepseq.metric import Metric, Levenshtein
from pyrepseq.metric.tcr_metric import AlphaCdr3Levenshtein, BetaCdr3Levenshtein, Cdr3Levenshtein
from pyrepseq.util import convert_tuple_to_dataframe_if_necessary
from scipy.spatial.distance import squareform
import scipy.cluster.hierarchy as hc
from typing import Iterable, Optional, Union
from warnings import warn


def pdist(strings, metric=None, dtype=np.uint8, **kwargs):
    """
    Pairwise distances between collection of strings.
    (`scipy.spatial.distance.pdist` equivalent for strings)

    Parameters
    ----------
    strings : iterable of strings
        An m-length iterable.
    metric : function, optional
        The distance metric to use. Default: Levenshtein distance.
    dtype : np.dtype
        data type of the distance matrix, default: np.uint8
        Note: make sure to change the dtype, if the metric does not return integers

    Returns
    -------
    Y : ndarray
        Returns a condensed distance matrix Y.  For
        each :math:`i` and :math:`j` (where :math:`i<j<m`), where m is the number
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
    for i in range(0, m - 1):
        for j in range(i + 1, m):
            dm[k] = metric(strings[i], strings[j], **kwargs)
            k += 1
    return dm


def cdist(stringsA, stringsB, metric=None, dtype=np.uint8, **kwargs):
    """
    Compute distance between each pair of the two collections of strings.
    (`scipy.spatial.distance.cdist` equivalent for strings)


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
        Note: make sure to change the dtype, if the metric does not return integers

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


def downsample(seqs: Union[Iterable[str], DataFrame, None], maxseqs: Optional[int] = None):
    """
    Random downsampling of a list of sequences.
    Also works for standard pyrepseq TCR DataFrames (see :py:func:`pyrepseq.io.standardize_dataframe`).

    Parameters
    ----------
    seqs: Union[Iterable[str], DataFrame]
        Input Iterable of strings, or TCR DataFrame.

    maxseqs: Optional[int]
        Max number of sequences to keep.
        Defaults to None.

    Returns
    -------
    Random subset of maxseqs elements from the input collection.
    If maxseqs is None, returns the input collection without modification.
    """
        
    if maxseqs is None or seqs is None:
        return seqs

    if len(seqs) <= maxseqs:
        return seqs

    if isinstance(seqs, DataFrame):
        return seqs.sample(n=maxseqs)
    
    return np.random.choice(seqs, maxseqs, replace=False)


def pcDelta(
    seqs: Iterable, seqs2: Optional[Iterable] = None, metric: Metric = None, bins: Union[int, Iterable] = None, normalize: bool = True, pseudocount: float = 0.0, maxseqs: Optional[int] = None
):
    r"""
    Calculates binned near-coincidence probabilities :math:`p_C(\Delta)` among input sequences.

    Parameters
    ----------
    seqs: Iterable
        A collection of elements to measure distances between.

    seqs2: Optional[Iterable]
        A second collection of elements for cross-comparisons.
        
    metric: :py:class:`pyrepseq.metric.Metric`
        The metric used to compute distances between elements.
        If not set, a default is inferred from the input data type of `seqs`.
        If `seqs` is a standard pyrepseq TCR DataFrame (see :py:func:`pyrepseq.io.standardize_dataframe`), then the metric can default to either a :py:class:`pyrepseq.metric.tcr_metric.Cdr3Levenshtein`, :py:class:`pyrepseq.metric.tcr_metric.AlphaCdr3Levenshtein`, or :py:class:`pyrepseq.metric.tcr_metric.BetaCdr3Levenshtein`, depending on what columns are available.
        In all other cases, the metric defaults to :py:class:`pyrepseq.metric.Levenshtein`.

    bins: Union[int, Iterable]
        bins for the distances Delta. (Default: range(0, 25))
        bins=0: Calculate exact coincidence probability

    normalize: bool
        whether to return pc (normalized) or raw counts

    pseudocount : float
       for a Bayesian estimation of coincidence frequencies
       e.g. can use Jeffrey's prior value of 0.5

    maxseqs: Optional[int]
        maximal number of sequences to keep by random downsampling

    Returns
    -------
    np.ndarray
        (normalized) histogram of sequence distances
    """
    if bins == 0:
        return pc(seqs, seqs2)
   
    seqs = convert_tuple_to_dataframe_if_necessary(seqs)
    seqs2 = convert_tuple_to_dataframe_if_necessary(seqs2)

    seqs = downsample(seqs, maxseqs)
    seqs2 = downsample(seqs2, maxseqs)

    if metric is None:
        metric = get_default_metric_for_input_data(seqs)

    if bins is None:
        bins = np.arange(0, 25)

    if seqs2 is None:
        hist, _ = np.histogram(metric.calc_pdist_vector(seqs), bins=bins)
    else:
        hist, _ = np.histogram(metric.calc_cdist_matrix(seqs, seqs2), bins=bins)

    if not normalize:
        return hist

    if not pseudocount:
        return hist / np.sum(hist)
    
    hist_sum = np.sum(hist) + 2 * pseudocount
    hist = hist.astype(np.float64) + pseudocount
    return hist / hist_sum


def get_default_metric_for_input_data(input_data: Union[Iterable[str], DataFrame]) -> Metric:
    if isinstance(input_data, DataFrame):
        if "CDR3A" in input_data and "CDR3B" in input_data:
            return Cdr3Levenshtein()
        elif "CDR3A" in input_data:
            return AlphaCdr3Levenshtein()
        elif "CDR3B" in input_data:
            return BetaCdr3Levenshtein()
    return Levenshtein()


def pcDelta_grouped(df, by, seq_columns, **kwargs):
    """Near-coincidence probabilities conditioned to within-group comparisons.

    Parameters
    ----------
    df : pd.DataFrame
    by : mapping, function, label, or list of labels
      see pd.DataFrame.groupby
    seq_columns : string
       The data frame column on which we want to apply the pcDelta analysis
    **kwargs : keyword arguments
        passed on to pcDelta

    Returns
    -------
    pcs : pd.DataFrame
        Returns a DataFrame of pC(delta) for each group

    """

    def pcDelta_within_group(dfg):
        index = kwargs.get("bins")
        index = [index] if isinstance(index, int) else index
        return pd.Series(pcDelta(dfg[seq_columns], **kwargs), name="Delta", index=index)

    return df.groupby(by).apply(pcDelta_within_group)


def pcDelta_grouped_cross(df, by, seq_columns, condensed=False, **kwargs):
    """Near-coincidence probabilities conditioned to cross-group comparisons.

    Parameters
    ----------
    df : pd.DataFrame
    by : mapping, function, label, or list of labels
      see pd.DataFrame.groupby
    seq_columns : string
       The data frame column on which we want to apply the pcDelta analysis
    condensed : bool
        Return a condensed instead of squareform matrix (default: False)
    **kwargs : keyword arguments
        passed on to pcDelta

    Returns
    -------
    pcs : pd.DataFrame
        Returns a DataFrame of pC(delta) across pairs of groups

    """

    groups = sorted(list(df.groupby(by)))
    data = []
    index = []
    for ((name1, d1)), (name2, d2) in itertools.combinations(groups, 2):
        pcg = pcDelta(d1[seq_columns], d2[seq_columns], **kwargs)
        index.append([name1, name2])
        data.append(pcg)
    data = np.array(data)
    if condensed:
        return pd.DataFrame(
            data, index=pd.MultiIndex.from_tuples(index, names=["group1", "group2"])
        )
    names = [name for name, dfg in groups]
    data_square = squareform(data)
    np.fill_diagonal(
        data_square, pcDelta_grouped(df, by, seq_columns=seq_columns, **kwargs)
    )
    return pd.DataFrame(data_square, index=names, columns=names)


def load_pcDelta_background(return_bins=True):
    """
    Loads pre-computed background pcDelta distributions calculated for PBMC TCRs.

    Data: Sample W_F1_2018 from Minervina et al. https://zenodo.org/record/4065547/

    Returns
    -------
    back : pd.DataFrame
        DataFrame with coincidence probabilities
    bins : ndarray [if return_bins = True]
        Delta bins to be used as bins for other data

    """
    folder = os.path.dirname(__file__)
    path = os.path.join(folder, "data", "pcdelta_pbmc_minervina.csv")
    back = pd.read_csv(path, index_col=0)
    if not return_bins:
        return back
    bins = list(back.index)
    bins.append(bins[-1] + 1)
    bins = np.array(bins)
    return back, bins


def levenshtein_neighbors(x, alphabet=aminoacids):
    """Iterator over Levenshtein neighbors of a string x"""
    # deletion
    for i in range(len(x)):
        # only delete first repeated amino acid
        if (i > 0) and (x[i] == x[i - 1]):
            continue
        yield x[:i] + x[i + 1 :]
    # replacement
    for i in range(len(x)):
        for aa in alphabet:
            # do not replace with same amino acid
            if aa == x[i]:
                continue
            yield x[:i] + aa + x[i + 1 :]
    # insertion
    for i in range(len(x) + 1):
        for aa in alphabet:
            # only insert after first repeated amino acid
            if (i > 0) and (aa == x[i - 1]):
                continue
            # insertion
            yield x[:i] + aa + x[i:]


def hamming_neighbors(x, alphabet=aminoacids, variable_positions=None):
    """Iterator over Hamming neighbors of a string x.

    Parameters
    ----------
    alphabet : iterable of characters
    variable_positions: iterable of positions to be varied (default: all)
    """

    if variable_positions is None:
        variable_positions = range(len(x))
    for i in variable_positions:
        for aa in alphabet:
            if aa == x[i]:
                continue
            yield x[:i] + aa + x[i + 1 :]


def _flatten_list(inlist):
    return [item for sublist in inlist for item in sublist]


def next_nearest_neighbors(x, neighborhood, maxdistance=2):
    """Set of next nearest neighbors of a string x.

    Parameters
    ----------
    alphabet : iterable of characters
    neighborhood: neighborhood iterator
    maxdistance : go up to maxdistance nearest neighbor

    Returns
    -------
    set of neighboring sequences
    """

    neighbors = [list(neighborhood(x))]
    distance = 1
    while distance < maxdistance:
        neighbors_dist = []
        for y in neighbors[-1]:
            neighbors_dist.extend(neighborhood(y))
        neighbors.append(set(neighbors_dist))
        distance += 1
    neighbor_set = set(_flatten_list(neighbors))
    neighbor_set.discard(x)
    return neighbor_set


def find_neighbor_pairs(seqs, neighborhood=hamming_neighbors):
    """Find neighboring sequences in a list of unique sequences.

    Parameters
    ----------
    neighborhood: callable returning an iterable of neighbors

    Returns
    -------
    list of tuples (seq1, seq2)
    """
    reference = set(seqs)
    pairs = []
    for x in sorted(set(seqs)):
        for y in set(neighborhood(x)) & reference:
            pairs.append((x, y))
        reference.remove(x)
    return pairs


def find_neighbor_pairs_index(seqs, neighborhood=hamming_neighbors):
    """Find neighboring sequences in a list of unique sequences.

    Parameters
    ----------
    neighborhood: callable returning an iterable of neighbors

    Returns
    -------
    list of tuples (index1, index2)
    """
    reference = set(seqs)
    seqs_list = list(seqs)
    pairs = []
    for i, x in enumerate(seqs):
        for y in set(neighborhood(x)) & reference:
            pairs.append((i, seqs_list.index(y)))
    return pairs


def calculate_neighbor_numbers(
    seqs, reference=None, neighborhood=levenshtein_neighbors
):
    """Calculate the number of neighbors for each sequence in a list.

    Parameters
    ----------
    seqs: list of sequences
    reference: list of sequences, set(seqs) if None
    neighborhood: function returning iterator over neighbors

    Returns
    -------
    integer array of number of neighboring sequences
    """
    if reference is None:
        reference = set(seqs)
    return np.array([len(set(neighborhood(seq)) & reference) for seq in seqs])


def isdist1(x, reference, neighborhood=levenshtein_neighbors):
    """Is the string x distance 1 away from any of the strings in the reference set"""
    for neighbor in neighborhood(x):
        if neighbor in reference:
            return True
    return False


def _isdist2_hamming(x, reference):
    """Is the string x a Hamming distance 2 away from any string in the reference set"""
    for i in range(len(x)):
        for aai in aminoacids:
            if aai == x[i]:
                continue
            si = x[:i] + aai + x[i + 1 :]
            for j in range(i + 1, len(x)):
                for aaj in aminoacids:
                    if aaj == x[j]:
                        continue
                    if si[:j] + aaj + si[j + 1 :] in reference:
                        return True
    return False


def _isdist3_hamming(x, reference):
    """Is the string x a Hamming distance 3 away from any string in the reference set"""
    for i in range(len(x)):
        for aai in aminoacids:
            if aai == x[i]:
                continue
            si = x[:i] + aai + x[i + 1 :]
            for j in range(i + 1, len(x)):
                for aaj in aminoacids:
                    if aaj == x[j]:
                        continue
                    sij = si[:j] + aaj + si[j + 1 :]
                    for k in range(j + 1, len(x)):
                        for aak in aminoacids:
                            if aak == x[k]:
                                continue
                            if sij[:k] + aak + sij[k + 1 :] in reference:
                                return True
    return False


def nndist_hamming(seq, reference, maxdist=4):
    """Calculate the nearest-neighbor distance by Hamming distance

    Parameters
    ----------
    seqs: list of sequences
    seq: sequence instance
    reference: set of referencesequences
    maxdist: distance beyond which to cut off the calculation (needs to be <=4)

    Returns
    -------
    distance of nearest neighbor

    Note: This function does not check if strings are of same length.
    """
    if maxdist > 4:
        raise NotImplementedError
    if seq in reference:
        return 0
    if (maxdist == 1) or isdist1(seq, reference, neighborhood=hamming_neighbors):
        return 1
    if (maxdist == 2) or _isdist2_hamming(seq, reference):
        return 2
    if (maxdist == 3) or _isdist3_hamming(seq, reference):
        return 3
    return 4


def hierarchical_clustering(
    seqs: Iterable,
    metric: Metric = None,
    linkage_kws=dict(method="average", optimal_ordering=True),
    cluster_kws=dict(t=6, criterion="distance"),
):
    """
    Hierarchical clustering by sequence similarity.

    Parameters
    ----------
    seqs: Iterable
        A collection of elements to cluster.

    metric: Metric
        The metric used to compute distances between elements.
        If not set, a default is inferred from the input data type of `seqs`.
        If `seqs` is a standard pyrepseq TCR DataFrame (see :py:func:`pyrepseq.io.standardize_dataframe`), then the metric can default to either a :py:class:`pyrepseq.metric.tcr_metric.Cdr3Levenshtein`, :py:class:`pyrepseq.metric.tcr_metric.AlphaCdr3Levenshtein`, or :py:class:`pyrepseq.metric.tcr_metric.BetaCdr3Levenshtein`, depending on what columns are available.
        In all other cases, the metric defaults to :py:class:`pyrepseq.metric.Levenshtein`.

    linkage_kws:
        keyword arguments for linkage algorithm
    
    cluster_kws:
        keyword arguments for clustering algorithm
    """
    seqs = convert_tuple_to_dataframe_if_necessary(seqs)
    
    if metric is None:
        metric = get_default_metric_for_input_data(seqs)

    distances = metric.calc_pdist_vector(seqs)
    linkage = hc.linkage(distances, **linkage_kws)
    cluster = hc.fcluster(linkage, **cluster_kws)
    return linkage, cluster
