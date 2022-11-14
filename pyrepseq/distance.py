from .io import aminoacids

import numpy as np

from scipy.spatial.distance import squareform
import scipy.cluster.hierarchy as hc

from Levenshtein import hamming as hamming_distance
from Levenshtein import distance as levenshtein_distance


def pdist(strings, metric=None, dtype=np.uint8, **kwargs):
    """Pairwise distances between collection of strings.
       (`scipy.spatial.distance.pdist` equivalent for strings)

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
    for i in range(0, m-1):
        for j in range(i+1, m):
            dm[k] = metric(strings[i], strings[j], **kwargs)
            k += 1
    return dm

def cdist(stringsA, stringsB, metric=None, dtype=np.uint8, **kwargs):
    """ Compute distance between each pair of the two collections of strings.
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

def downsample(seqs, maxseqs):
    """
    Random downsampling of a list of sequences.

    Also works for tuples (seqs_alpha, seqs_beta).
    """
    if maxseqs is None:
        return seqs
    if type(seqs) is tuple:
        seqs_alpha, seqs_beta = seqs
        if len(seqs_alpha) <= maxseqs:
            return seqs
        indices = np.random.choice(np.arange(len(seqs_alpha)), maxseqs, replace=False)
        return np.asarray(seqs_alpha)[indices], np.asarray(seqs_beta)[indices]
    if len(seqs) > maxseqs:
        return np.random.choice(seqs, maxseqs, replace=False)
    return seqs

def pcDelta(seqs, seqs2=None, bins=None,
            normalize=True, pseudocount=0.5, 
            maxseqs=None,
            **kwargs):
    r"""
    Calculates binned near-coincidence probabilities :math:`p_C(\Delta)`
    among input sequences.

    Parameters
    ----------
    seqs: [list of strings | tuple of lists]
        sequences, or (seqs_alpha, seqs_beta)
    seqs2: [list of strings | tuple of lists] (optional)
        second list of sequences for cross-comparisons
    bins: iterable
        bins for the distances Delta. (Default: range(0, 25))
    normalize: bool
        whether to return pc (normalized) or raw counts
    pseudocount : float
       by default uses Jeffrey's prior value of 0.5 
    maxseqs: int
        maximal number of sequences to keep by random downsampling
    **kwargs: dict
        passed on to `pdist` or `cdist`

    Returns
    -------
    np.ndarray
        (normalized) histogram of sequence distances
    """
    if bins is None:
        bins = np.arange(0, 25)
    seqs = downsample(seqs, maxseqs)
    if type(seqs) is tuple:
        seqs_alpha, seqs_beta = seqs
        if seqs2 is None:
            hist, _ = np.histogram(pdist(seqs_alpha, **kwargs) + pdist(seqs_beta, **kwargs), bins=bins)
        else:
            seqs_alpha2, seqs_beta2 = seqs2
            hist, _ = np.histogram(cdist(seqs_alpha, seqs_alpha2, **kwargs) + cdist(seqs_beta, seqs_beta2, **kwargs), bins=bins)
    else:
        if seqs2 is None:
            hist, _ = np.histogram(pdist(seqs, **kwargs), bins=bins)
        else:
            hist, _ = np.histogram(cdist(seqs, seqs2, **kwargs), bins=bins)
    if not normalize:
        return hist
    if not pseudocount:
        return hist/np.sum(hist)
    hist_sum = np.sum(hist)+2*pseudocount
    hist = hist.astype(np.float64)+pseudocount
    return hist/hist_sum

def pcDelta_grouped(df, by, seq_columns):
    """Near-coincidence probabilities conditioned to within-group comparisons.
    
    Parameters
    ----------
    df : pd.DataFrame
    by : mapping, function, label, or list of labels
      see pd.DataFrame.groupby
    seq_columns : string
       The data frame column on which we want to apply the pcDelta analysis
       
    Returns
    -------
    Y : ndarray
        Returns the the mean of the pC deltas for each group average by delta value
    
    """
    
    
    pcDeltas = []
    for label, dfg in df.groupby(by):
        pcDeltas.append(pcDelta(dfg[seq_columns], pseudocount=0.0, normalize=False))
    pcDeltas = np.array(pcDeltas)
    return np.sum(pcDeltas, axis=0)/np.sum(pcDeltas)

def levenshtein_neighbors(x, alphabet=aminoacids):
    """Iterator over Levenshtein neighbors of a string x"""
    # deletion
    for i in range(len(x)):
        # only delete first repeated amino acid
        if (i > 0) and (x[i] == x[i-1]):
            continue
        yield x[:i]+x[i+1:]
    # replacement
    for i in range(len(x)):
        for aa in alphabet:
            # do not replace with same amino acid
            if aa == x[i]:
                continue
            yield x[:i]+aa+x[i+1:]
    # insertion
    for i in range(len(x)+1):
        for aa in alphabet:
            # only insert after first repeated amino acid
            if (i>0) and (aa == x[i-1]):
                continue
            # insertion
            yield x[:i]+aa+x[i:]

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
            yield x[:i]+aa+x[i+1:]

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
        for x in neighbors[-1]:
            neighbors_dist.extend(neighborhood(x))
        neighbors.append(set(neighbors_dist))
        distance += 1
    neighbor_set = set(_flatten_list(neighbors))
    try:
        neighbor_set.remove(x)
    except KeyError:
        pass
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
        for y in (set(neighborhood(x)) & reference):
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
        for y in (set(neighborhood(x)) & reference):
            pairs.append((i, seqs_list.index(y)))
    return pairs

def calculate_neighbor_numbers(seqs, neighborhood=levenshtein_neighbors):
    """Calculate the number of neighbors for each sequence in a list.

    Parameters
    ----------
    seqs: list of sequences
    neighborhood: function returning iterator over neighbors
    
    Returns
    -------
    integer array of number of neighboring sequences
    """
    reference = set(seqs)
    return np.array([len(set(neighborhood(seq)) & reference) for seq in seqs])

def isdist1(x, reference, neighborhood=levenshtein_neighbors):
    """ Is the string x distance 1 away from any of the strings in the reference set"""
    for neighbor in neighborhood(x):
        if neighbor in reference:
            return True
    return False

def _isdist2(x, reference):
    """ Is the string x a Hamming distance 2 away from any of the kmers in the reference set"""
    for i in range(len(x)):
        for j in range(i+1, len(x)):
            for aai in aminoacids:
                if aai == x[i]:
                    continue
                si = x[:i]+aai+x[i+1:]
                for aaj in aminoacids:
                    if aaj == x[j]:
                        continue
                    if si[:j]+aaj+si[j+1:] in reference:
                        return True
    return False

def _isdist3(x, reference):
    """ Is the string x a Hamming distance 3 away from any of the kmers in the reference set"""
    for i in range(len(x)):
        for j in range(i+1, len(x)):
            for k in range(j+1, len(x)):
                for aai in aminoacids:
                    if aai == x[i]:
                        continue
                    si = x[:i]+aai+x[i+1:]
                    for aaj in aminoacids:
                        if aaj == x[j]:
                            continue
                        sij = si[:j]+aaj+si[j+1:]
                        for aak in aminoacids:
                            if aak == x[k]:
                                continue
                            if sij[:k]+aak+sij[k+1:] in reference:
                                return True
    return False

def nndist_hamming(seq, reference, maxdist=4):
    """Calculate the nearest-neighbor distance by Hamming distance

    Parameters
    ----------
    seqs: list of sequences
    seq: sequence instance
    reference: set of referencesequences
    maxdist: distance beyond which to cut off the calculation (currently needs to be <=4)

    Returns
    -------
    distance of nearest neighbor 

    Note: This function does not check whether neighbors are of same length.
    """
    if maxdist>4:
        raise NotImplementedError
    if seq in reference:
        return 0
    if (maxdist==1) or isdist1(seq, reference):
        return 1
    if (maxdist==2) or _isdist2(seq, reference):
        return 2
    if (maxdist==3) or _isdist3(seq, reference):
        return 3
    return 4

def hierarchical_clustering(seqs, 
                            pdist_kws=dict(),
                            linkage_kws=dict(method='average', optimal_ordering=True),
                            cluster_kws=dict(t=6, criterion='distance')):
    """
    Hierarchical clustering by sequence similarity.

    pdist_kws: keyword arguments for distance calculation
    linkage_kws: keyword arguments for linkage algorithm
    cluster_kws: keyword arguments for clustering algorithm
    """
    if type(seqs) is tuple:
        seqs_alpha, seqs_beta = seqs
        distances_alpha = pdist(seqs_alpha, **pdist_kws)
        distances_beta = pdist(seqs_beta, **pdist_kws)
        distances = distances_alpha + distances_beta
    else:
        raise NotImplementedError('seqs needs to be a tuple')
    linkage = hc.linkage(distances, **linkage_kws)
    cluster = hc.fcluster(linkage, **cluster_kws)
    return linkage, cluster
