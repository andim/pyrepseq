from scipy.spatial import KDTree
import numpy as np
from rapidfuzz.distance.Levenshtein import distance as levenshtein
from rapidfuzz.distance.Hamming import distance as hamming
from scipy.sparse import coo_matrix
from rapidfuzz.process import extract
from multiprocessing import Pool
from .util import flatten_array, ensure_numpy, check_common_input, make_output
from .distance import levenshtein_neighbors, hamming_neighbors

AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'


def histogram_encode(cdr3, compression):
    dimension = int(np.ceil(len(AMINO_ACIDS)/compression))
    position_map = {char: int(np.floor(index/compression)) for index,
                    char in enumerate(AMINO_ACIDS)}

    ans = np.zeros(dimension, dtype='int')
    for char in cdr3:
        ans[position_map[char]] += 1
    return ans


def cal_levenshtein(_args):
    # handle must be a global function for parallelization
    i, y_indices = _args
    seqs, max_edits, limit, custom_distance, _ = _cal_params
    scorer = hamming if custom_distance == 'hamming' else levenshtein
    choices = list(filter(lambda y_index: y_index != i, y_indices))
    result = extract(seqs[i], seqs[choices],
                     score_cutoff=max_edits, scorer=scorer, limit=limit)

    ans = []
    for _, dist, y_index in result:
        ans.append((i, choices[y_index], dist))
    return ans


def cal_custom_dist(_args):
    (i, y_indices) = _args
    seqs, max_edits, limit, dist, max_cust_dist = _cal_params
    query = seqs[i]

    y_indices = filter(lambda y_index: y_index != i, y_indices)
    ans = [(i, y_i, dist(query, seqs[y_i])) for y_i in y_indices]

    def distance_filter(x):
        edit_distance = levenshtein(query, seqs[x[1]])
        return x[2] <= max_cust_dist and edit_distance <= max_edits
    ans = sorted(filter(distance_filter, ans), key=lambda x: x[2])
    return ans if limit is None else ans[0:limit]


def to_triplets(seqs, y_indices, max_edits, limit, n_cpu, custom_distance,
                max_cust_dist):
    global _cal_params
    cal = cal_levenshtein if custom_distance in (
        None, 'hamming') else cal_custom_dist
    _cal_params = (seqs, max_edits, limit, custom_distance, max_cust_dist)
    _loop = enumerate(y_indices)

    if n_cpu == 1:
        result = map(cal, _loop)
    else:
        with Pool(n_cpu) as p:
            result = p.map(cal, _loop, chunksize=int(len(seqs)/n_cpu))
    return flatten_array(result)


def to_len_bucket(seqs):
    ans = {}
    for seq in seqs:
        _len = len(seq)
        if _len not in ans:
            ans[_len] = []
        ans[_len].append(seq)
    return ans


def kdtree(seqs, max_edits=1, max_returns=None, n_cpu=1,
           custom_distance=None, max_custom_distance=float('inf'),
           output_type='triplets', compression=1):
    """
    List all neighboring CDR3B sequences efficiently within the given edit distance.
    With KDTree, the algorithms run with O(N logN) eliminating unnecessary comparisons.
    With RapidFuzz library, the edit distance comparison is efficiently written in C++.
    With multiprocessing, the algorithm can take advantage of multiple CPU cores.
    This implementation is faster than hash-based implementation for max_edits > 1

    Parameters
    ----------
    strings : iterable of strings
        list of CDR3B sequences
    max_edits : int
        maximum edit distance defining the neighbors
    max_returns : int or None
        maximum neighbor size
    n_cpu : int
        number of CPU cores running in parallel
    custom_distance : Function(str1, str2) or "hamming"
        custom distance function to use, must statisfy 4 properties of distance (https://en.wikipedia.org/wiki/Distance#Mathematical_formalization)
    max_custom_distance : float
        maximum distance to include in the result, ignored if custom distance is not supplied
    output_type: string
        format of returns, can be "triplets", "coo_matrix", or "ndarray"

    Returns
    -------
    neighbors : array of 3D-tuples, sparse matrix, or dense matrix
        neigbors along with their edit distances according to the given output_type
        if "triplets" returns are [(x_index, y_index, edit_distance)]
        if "coo_matrix" returns are scipy's sparse matrix where C[i,j] = distance(X_i, X_j) or 0 if not neighbor
        if "ndarray" returns numpy's 2d array representing dense matrix
    """
    check_common_input(seqs, max_edits, max_returns, n_cpu,
                       custom_distance, max_custom_distance, output_type)

    if custom_distance == 'hamming':
        buckets, ans = to_len_bucket(seqs), []
        for bucket in buckets.values():
            ans += kdtree_leven(bucket, max_edits, max_returns, n_cpu,
                                custom_distance, max_custom_distance,
                                'triplets', compression)
        return make_output(ans, output_type)
    return kdtree_leven(seqs, max_edits, max_returns, n_cpu, custom_distance,
                        max_custom_distance, output_type, compression)


def kdtree_leven(seqs, max_edits=1, max_returns=None, n_cpu=1,
                 custom_distance=None, max_custom_distance=float('inf'),
                 output_type='triplets', compression=1):
    # boilerplate
    seqs = ensure_numpy(seqs)
    params = {'r': np.sqrt(2)*max_edits, 'workers': n_cpu}

    # algorithm
    matrix = [histogram_encode(x, compression) for x in seqs]
    tree = KDTree(matrix, compact_nodes=True, balanced_tree=True)
    y_indices = tree.query_ball_point(matrix, **params)
    triplets = to_triplets(seqs, y_indices, max_edits,
                           max_returns, n_cpu, custom_distance,
                           max_custom_distance)
    return make_output(triplets, output_type)


def generate_neighbors(query, max_edits, is_hamming):
    neighbor_func = hamming_neighbors if is_hamming else levenshtein_neighbors
    ans = {query: 0}
    for edit_distance in range(1, max_edits+1):
        for seq in ans.copy():
            for new_seq in neighbor_func(seq):
                if new_seq not in ans:
                    ans[new_seq] = edit_distance
    return ans


def build_index(seqs):
    ans = {}
    for index, seq in enumerate(seqs):
        if seq not in ans:
            ans[seq] = []
        ans[seq].append(index)
    return ans


def single_lookup(args):
    index, max_edits, limit, custom_distance, max_cust_dist = _params
    ans, (x_index, seq), is_hamming = [], args, custom_distance == 'hamming'
    neighbors = generate_neighbors(seq, max_edits, is_hamming)

    for possible_edit, edit_distance in neighbors.items():
        if possible_edit in index and edit_distance <= max_edits:
            for y_index in index[possible_edit]:
                if x_index == y_index:
                    continue
                if custom_distance in (None, 'hamming'):
                    ans.append((x_index, y_index, edit_distance))
                else:
                    dist = custom_distance(seq, possible_edit)
                    if dist <= max_cust_dist:
                        ans.append((x_index, y_index, dist))
    return ans if limit is None else ans[0:limit]


def lookup(index, seqs, max_edits, max_returns, n_cpu,
           custom_distance, max_cust_dist):
    global _params
    _params = (index, max_edits, max_returns, custom_distance, max_cust_dist)

    _loop = enumerate(seqs)
    if n_cpu == 1:
        result = map(single_lookup, _loop)
    else:
        with Pool(n_cpu) as p:
            chunk = int(len(seqs)/n_cpu)
            result = p.map(single_lookup, _loop, chunksize=chunk)
    return flatten_array(result)


def hash_based(seqs, max_edits=1, max_returns=None, n_cpu=1,
               custom_distance=None, max_custom_distance=float('inf'),
               output_type='triplets'):
    """
    List all neighboring CDR3B sequences efficiently for small edit distances.
    The idea is to list all possible sequences within a given distance and lookup the dictionary if it exists.
    This implementation is faster than kdtree implementation for max_edits == 1

    Parameters
    ----------
    strings : iterable of strings
        list of CDR3B sequences
    max_edits : int
        maximum edit distance defining the neighbors
    max_returns : int or None
        maximum neighbor size
    n_cpu : int
        number of CPU cores running in parallel
    custom_distance : Function(str1, str2) or "hamming"
        custom distance function to use, must statisfy 4 properties of distance (https://en.wikipedia.org/wiki/Distance#Mathematical_formalization)
    max_custom_distance : float
        maximum distance to include in the result, ignored if custom distance is not supplied
    output_type: string
        format of returns, can be "triplets", "coo_matrix", or "ndarray"

    Returns
    -------
    neighbors : array of 3D-tuples, sparse matrix, or dense matrix
        neigbors along with their edit distances according to the given output_type
        if "triplets" returns are [(x_index, y_index, edit_distance)]
        if "coo_matrix" returns are scipy's sparse matrix where C[i,j] = distance(X_i, X_j) or 0 if not neighbor
        if "ndarray" returns numpy's 2d array representing dense matrix
    """

    # boilerplate
    check_common_input(seqs, max_edits, max_returns, n_cpu,
                       custom_distance, max_custom_distance, output_type)
    seqs = ensure_numpy(seqs)

    # algorithm
    index = build_index(seqs)
    triplets = lookup(index, seqs, max_edits,
                      max_returns, n_cpu, custom_distance, max_custom_distance)
    return make_output(triplets, output_type)


def nearest_neighbor(seqs, max_edits=1, max_returns=None, n_cpu=1,
                     custom_distance=None, max_custom_distance=float('inf'),
                     output_type='triplets'):
    """
    List all neighboring CDR3B sequences efficiently within the given distance.
    The distance can be given in terms of hamming, levenshtein, or custom.

    Parameters
    ----------
    strings : iterable of strings
        list of CDR3B sequences
    max_edits : int
        maximum edit distance defining the neighbors
    max_returns : int or None
        maximum neighbor size
    n_cpu : int
        number of CPU cores running in parallel
    custom_distance : Function(str1, str2) or "hamming"
        custom distance function to use, must statisfy 4 properties of distance (https://en.wikipedia.org/wiki/Distance#Mathematical_formalization)
    max_custom_distance : float
        maximum distance to include in the result, ignored if custom distance is not supplied
    output_type: string
        format of returns, can be "triplets", "coo_matrix", or "ndarray"

    Returns
    -------
    neighbors : array of 3D-tuples, sparse matrix, or dense matrix
        neigbors along with their edit distances according to the given output_type
        if "triplets" returns are [(x_index, y_index, edit_distance)]
        if "coo_matrix" returns are scipy's sparse matrix where C[i,j] = distance(X_i, X_j) or 0 if not neighbor
        if "ndarray" returns numpy's 2d array representing dense matrix
    """

    if max_edits == 1 and len(seqs) > 10000:
        return hash_based(seqs, max_edits, max_returns, n_cpu,
                          custom_distance, max_custom_distance, output_type)
    return kdtree(seqs, max_edits, max_returns, n_cpu,
                  custom_distance, max_custom_distance, output_type)
