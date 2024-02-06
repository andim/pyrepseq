from scipy.spatial import KDTree
import numpy as np
import pandas as pd
from rapidfuzz.distance.Levenshtein import distance as levenshtein
from rapidfuzz.distance.Hamming import distance as hamming
from scipy.sparse import coo_matrix
from rapidfuzz.process import extract
from multiprocessing import Pool
from .distance import levenshtein_neighbors, hamming_neighbors
from itertools import combinations, chain
from .util import ensure_numpy
import os
import pwseqdist
import re


_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

# ===================================
# kdtree
# ===================================


def _histogram_encode(cdr3, compression):
    dimension = int(np.ceil(len(_AMINO_ACIDS) / compression))
    position_map = {
        char: int(np.floor(index / compression))
        for index, char in enumerate(_AMINO_ACIDS)
    }

    ans = np.zeros(dimension, dtype="int")
    for char in cdr3:
        ans[position_map[char]] += 1
    return ans


def _cal_levenshtein(_args):
    # handle must be a global function for parallelization
    i, y_indices = _args
    seqs, max_edits, limit, custom_distance, _ = _cal_params
    scorer = hamming if custom_distance == "hamming" else levenshtein
    choices = list(filter(lambda y_index: y_index != i, y_indices))
    result = extract(
        seqs[i], seqs[choices], score_cutoff=max_edits, scorer=scorer, limit=limit
    )

    ans = []
    for _, dist, y_index in result:
        ans.append((i, choices[y_index], dist))
    return ans


def _cal_custom_dist(_args):
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


def _to_triplets(
    seqs, y_indices, max_edits, limit, n_cpu, custom_distance, max_cust_dist
):
    global _cal_params
    cal = _cal_levenshtein if custom_distance in (None, "hamming") else _cal_custom_dist
    _cal_params = (seqs, max_edits, limit, custom_distance, max_cust_dist)
    _loop = enumerate(y_indices)

    if n_cpu == 1:
        result = map(cal, _loop)
    else:
        with Pool(n_cpu) as p:
            result = p.map(cal, _loop, chunksize=int(len(seqs) / n_cpu))
    return _flatten_array(result)


def _to_len_bucket(seqs):
    ans = {}
    for seq in seqs:
        _len = len(seq)
        if _len not in ans:
            ans[_len] = []
        ans[_len].append(seq)
    return ans


def kdtree(
    seqs,
    max_edits=1,
    max_returns=None,
    n_cpu=1,
    custom_distance=None,
    max_custom_distance=float("inf"),
    output_type="triplets",
    compression=1,
):
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
    _check_common_input(
        seqs,
        max_edits,
        max_returns,
        n_cpu,
        custom_distance,
        max_custom_distance,
        output_type,
    )

    if custom_distance == "hamming":
        buckets, ans = _to_len_bucket(seqs), []
        for bucket in buckets.values():
            ans += _kdtree_leven(
                bucket,
                max_edits,
                max_returns,
                n_cpu,
                custom_distance,
                max_custom_distance,
                "triplets",
                compression,
            )
        return _make_output(ans, output_type, seqs, seqs)
    return _kdtree_leven(
        seqs,
        max_edits,
        max_returns,
        n_cpu,
        custom_distance,
        max_custom_distance,
        output_type,
        compression,
    )


def _kdtree_leven(
    seqs,
    max_edits=1,
    max_returns=None,
    n_cpu=1,
    custom_distance=None,
    max_custom_distance=float("inf"),
    output_type="triplets",
    compression=1,
):
    # boilerplate
    seqs = ensure_numpy(seqs)
    params = {"r": np.sqrt(2) * max_edits, "workers": n_cpu}

    # algorithm
    matrix = [_histogram_encode(x, compression) for x in seqs]
    tree = KDTree(matrix, compact_nodes=True, balanced_tree=True)
    y_indices = tree.query_ball_point(matrix, **params)
    triplets = _to_triplets(
        seqs,
        y_indices,
        max_edits,
        max_returns,
        n_cpu,
        custom_distance,
        max_custom_distance,
    )
    return _make_output(triplets, output_type, seqs, seqs)


# ===================================
# hash-based
# ===================================

def _generate_neighbors(query, max_edits, is_hamming):
    neighbor_func = hamming_neighbors if is_hamming else levenshtein_neighbors
    ans = {query: 0}
    for edit_distance in range(1, max_edits + 1):
        for seq in ans.copy():
            for new_seq in neighbor_func(seq):
                if new_seq not in ans:
                    ans[new_seq] = edit_distance
    return ans


def _build_index(seqs):
    ans = {}
    for index, seq in enumerate(seqs):
        if seq not in ans:
            ans[seq] = []
        ans[seq].append(index)
    return ans


def _single_lookup(args):
    index, max_edits, limit, custom_distance, max_cust_dist = _params
    ans, (x_index, seq), is_hamming = [], args, custom_distance == "hamming"
    neighbors = _generate_neighbors(seq, max_edits, is_hamming)

    for possible_edit, edit_distance in neighbors.items():
        if possible_edit in index and edit_distance <= max_edits:
            for y_index in index[possible_edit]:
                if x_index == y_index:
                    continue
                if custom_distance in (None, "hamming"):
                    ans.append((x_index, y_index, edit_distance))
                else:
                    dist = custom_distance(seq, possible_edit)
                    if dist <= max_cust_dist:
                        ans.append((x_index, y_index, dist))
    return ans if limit is None else ans[0:limit]


def lookup(index, seqs, max_edits, max_returns, n_cpu, custom_distance, max_cust_dist):
    global _params
    _params = (index, max_edits, max_returns, custom_distance, max_cust_dist)

    _loop = enumerate(seqs)
    if n_cpu == 1:
        result = map(_single_lookup, _loop)
    else:
        with Pool(n_cpu) as p:
            chunk = int(len(seqs) / n_cpu)
            result = p.map(_single_lookup, _loop, chunksize=chunk)
    return _flatten_array(result)


def hash_based(
    seqs,
    max_edits=1,
    max_returns=None,
    n_cpu=1,
    custom_distance=None,
    max_custom_distance=float("inf"),
    output_type="triplets",
):
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
    _check_common_input(
        seqs,
        max_edits,
        max_returns,
        n_cpu,
        custom_distance,
        max_custom_distance,
        output_type,
    )
    seqs = ensure_numpy(seqs)

    # algorithm
    index = _build_index(seqs)
    triplets = lookup(
        index, seqs, max_edits, max_returns, n_cpu, custom_distance, max_custom_distance
    )
    return _make_output(triplets, output_type, seqs, seqs)


# ===================================
# symspell
# ===================================


def _comb_gen(seq, max_edits):
    _len, ans = len(seq), set([seq])
    for edit in range(1, max_edits+1):
        for indexes in combinations(range(_len), edit):
            new_seq, offset = [], 0
            for index in indexes:
                new_seq.append(seq[offset:index])
                offset = index+1
            new_seq.append(seq[offset:_len])
            ans.add(''.join(new_seq))
    return ans


def _generate_index(seqs, max_edits):
    ans = {}
    for i, seq in enumerate(seqs):
        for comb in _comb_gen(seq, max_edits):
            if comb in ans:
                ans[comb].append(i)
            else:
                ans[comb] = [i]
    return ans


def _hamming_replacement(seq_a, seq_b):
    # the older versions of the library did not throw error, so we force it
    if len(seq_a) != len(seq_b):
        raise ValueError()
    return hamming(seq_a, seq_b)


def _symspell_lookup(index, seqs, max_edits, max_returns,
                    custom_distance, max_custom_dist, seqs2, single_seqs_mode):
    ans = []
    threshold = max_custom_dist
    if custom_distance in (None, 'hamming') or max_custom_dist == float('inf'):
        threshold = max_edits
    max_returns = max_returns if max_returns is not None else float('inf')
    if custom_distance == 'hamming':
        custom_distance = _hamming_replacement
    elif custom_distance is None:
        custom_distance = levenshtein

    for i, seq in enumerate(seqs2):
        j_indices, count = set(), 0
        for comb in _comb_gen(seq, max_edits):
            if comb not in index:
                continue
            for j_index in index[comb]:
                j_indices.add(j_index)

        for j_index in j_indices:
            if i == j_index and single_seqs_mode:
                continue
            try:
                dist = custom_distance(seqs2[i], seqs[j_index])
                if dist > threshold:
                    continue
                if count >= max_returns:
                    break
                ans.append((i, j_index, dist))
                count += 1
            except:
                continue
    return ans


def symspell(seqs, max_edits=1, max_returns=None, n_cpu=1,
             custom_distance=None, max_custom_distance=float('inf'),
             output_type='triplets', seqs2=None):
    """
    List all neighboring CDR3B sequences efficiently within the given distance.
    This is an improved version over the hash-based.

    If seqs2 is not provided, every sequences are compared against every other sequences resulting in N(seqs)**2 combinations.
    Otherwise, seqs are compared against seqs2 resulting in N(seqs)*N(seqs2) combinations.

    Parameters
    ----------
    strings : iterable of strings
        list of CDR3B sequences
    max_edits : int
        maximum edit distance defining the neighbors
    max_returns : int or None
        maximum neighbor size
    n_cpu : int
        ignored
    custom_distance : Function(str1, str2) or "hamming"
        custom distance function to use, must statisfy 4 properties of distance (https://en.wikipedia.org/wiki/Distance#Mathematical_formalization)
    max_custom_distance : float
        maximum distance to include in the result, ignored if custom distance is not supplied
    output_type: string
        format of returns, can be "triplets", "coo_matrix", or "ndarray"
    seq2 : iterable of strings or None
        another list of CDR3B sequences to compare against

    Returns
    -------
    neighbors : array of 3D-tuples, sparse matrix, or dense matrix
        neigbors along with their edit distances according to the given output_type
        if "triplets" returns are [(x_index, y_index, edit_distance)]
        if "coo_matrix" returns are scipy's sparse matrix where C[i,j] = distance(X_i, X_j) or 0 if not neighbor
        if "ndarray" returns numpy's 2d array representing dense matrix
    """

    _check_common_input(
        seqs,
        max_edits,
        max_returns,
        n_cpu,
        custom_distance,
        max_custom_distance,
        output_type,
        seqs2
    )
    single_seqs_mode = seqs2 is None
    if single_seqs_mode:
        seqs2 = seqs
    index = _generate_index(seqs, max_edits)
    triplets = _symspell_lookup(index, seqs, max_edits, max_returns,
                               custom_distance, max_custom_distance, seqs2, single_seqs_mode)
    return _make_output(triplets, output_type, seqs, seqs2)


# ===================================
# combined function
# ===================================


def nearest_neighbor(seqs, max_edits=1, max_returns=None, n_cpu=1,
                     custom_distance=None, max_custom_distance=float('inf'),
                     output_type='triplets', seqs2=None):
    """
    List all neighboring sequences efficiently within a given distance.
    The distance can be given in terms of hamming, levenshtein, or custom.

    If seqs2 is not provided, every sequence is compared against every other sequence.

    Parameters
    ----------
    strings : iterable of strings
        list of CDR3B sequences
    max_edits : int
        maximum edit distance defining the neighbors
    max_returns : int or None
        maximum neighbor size
    n_cpu : int
        ignored
    custom_distance : Function(str1, str2) or "hamming"
        custom distance function to use, must statisfy 4 properties of distance (https://en.wikipedia.org/wiki/Distance#Mathematical_formalization)
    max_custom_distance : float
        maximum distance to include in the result, ignored if custom distance is not supplied
    output_type: string
        format of returns, can be "triplets", "coo_matrix", or "ndarray"
    seq2 : iterable of strings or None
        another list of CDR3B sequences to compare against

    Returns
    -------
    neighbors : array of 3D-tuples, sparse matrix, or dense matrix
        neigbors along with their edit distances according to the given output_type
        if "triplets" returns are [(x_index, y_index, edit_distance)]
        if "coo_matrix" returns are scipy's sparse matrix where C[i,j] = distance(X_i, X_j) or 0 if not neighbor
        if "ndarray" returns numpy's 2d array representing dense matrix
    """

    return symspell(seqs, max_edits, max_returns, n_cpu,
                    custom_distance, max_custom_distance, output_type, seqs2)


def _lookup(df, row_labels, col_labels):
    values = df.values
    ridx = df.index.get_indexer(row_labels)
    cidx = df.columns.get_indexer(col_labels)
    flat_index = ridx * len(df.columns) + cidx
    return values.flat[flat_index]

def nearest_neighbor_tcrdist(df, chain='beta', max_edits=1, max_tcrdist=20, **kwargs):
    """
    List all neighboring TCR sequences efficiently within a given edit and TCRdist radius.

    chain: 'alpha' or 'beta'
    max_edits : only return neighbors up to <= this edit distance
    max_tcrdist : only return neighbor up to <= this TCR distance

    **kwargs : passed on to nearest_neighbor function

    """
    chain_letter = chain[0].upper()
    neighbors = nearest_neighbor(list(df[f'CDR3{chain_letter}']),
                                 max_edits=max_edits, **kwargs)

    folder = os.path.dirname(__file__)
    path = os.path.join(folder, "data", f"vdists_{chain}.csv")
    vdists = pd.read_csv(path, index_col=0)

    neighbors_arr = np.array(neighbors)
    edges = neighbors_arr[:, :2]
    tcrdist_v = _lookup(vdists,
                        df[f'TR{chain_letter}V'].iloc[edges[:, 0]],
                        df[f'TR{chain_letter}V'].iloc[edges[:, 1]])
    tcrdist_cdr3 = pwseqdist.apply_pairwise_sparse(metric=pwseqdist.metrics.nb_vector_tcrdist,
                                seqs=np.asarray(df[f'CDR3{chain_letter}']), pairs=edges,
                                use_numba=True)
    tcrdist = tcrdist_v + tcrdist_cdr3
    neighbors_arr[:, 2] = tcrdist

    return neighbors_arr[neighbors_arr[:, 2]<=max_tcrdist]

# ===================================
# util
# ===================================


def _flatten_array(nested_array):
    return list(chain(*nested_array))


def _check_common_input(
    seqs, max_edits, max_returns, n_cpu, custom_distance, max_cust_dist, output_type, seqs2=None
):
    assert len(seqs) > 0, "length must be greater than 0"
    try:
        for seq in seqs:
            assert type(seq) in {
                str,
                np.str_,
            }, "sequences must be an iterable of string"
            assert re.match(
                r"^[ACDEFGHIKLMNPQRSTVWY]+$", seq
            ), "sequences must contain only valid amino acids"
    except TypeError:
        assert False, "sequences must be an iterable of string"
    assert (
        type(max_edits) == int and max_edits > 0
    ), "max_edits must be a positive integer"
    assert (
        type(max_returns) == int and max_returns > 0
    ) or max_returns is None, "max_returns must be a positive integer or None"
    assert type(n_cpu) == int and n_cpu > 0, "n_cpu must be a positive integer"
    try:
        first = next(seqs.__iter__())
        assert (
            custom_distance in (None, "hamming") or custom_distance(first, first) == 0
        ), "custom_distance must be None or custom_distance(x1,x2)==0"
    except AssertionError:
        raise
    except:
        assert False, "custom_distance evaluation fails"
    assert (
        type(max_cust_dist) in (int, float) and max_cust_dist >= 0
    ), "max_custom_distance must be a non-negative number"
    assert output_type in {
        "coo_matrix",
        "triplets",
        "ndarray",
    }, "output must be either coo_matrix, triplets, or ndarray"
    try:
        for seq in seqs2:
            assert type(seq) in {
                str,
                np.str_,
            }, "sequences2 must be an iterable of string"
            assert re.match(
                r"^[ACDEFGHIKLMNPQRSTVWY]+$", seq
            ), "sequences2 must contain only valid amino acids"
    except TypeError:
        assert seqs2 is None, "sequences2 must be an iterable of string or None"


def _make_output(triplets, output_type, seqs, seqs2):
    if output_type == "triplets":
        return triplets

    row, col, data = [], [], []
    for triplet in triplets:
        row += [triplet[0], triplet[1]]
        column += [triplet[1], triplet[0]]
        data += [triplet[2], triplet[2]]
    coo_result = coo_matrix((data, (row, col)), shape=(len(seqs), len(seqs2)))

    return coo_result if output_type == "coo_matrix" else coo_result.toarray()
