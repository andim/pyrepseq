from typing import Literal, List, Tuple
from scipy.spatial import KDTree
import numpy as np
import pandas as pd
import tqdm.auto
from rapidfuzz.distance.Levenshtein import distance as levenshtein
from rapidfuzz.distance.Hamming import distance as hamming
from scipy.sparse import coo_matrix
from rapidfuzz.process import extract
from multiprocessing import Pool
import logging
from pathlib import Path


from .distance import levenshtein_neighbors, hamming_neighbors
from itertools import combinations, chain
from .util import ensure_numpy
from .io import aminoacids
import os

import warnings

try:
    import pwseqdist
except ImportError:
    warnings.warn(
        "optional dependency pwseqdist not installed (tcrdist neighbor search not supported)",
        ImportWarning,
    )
try:
    import sceptr
except ImportError:
    warnings.warn(
        "optional dependency sceptr not installed (sceptr neighbor search not supported)",
        ImportWarning,
    )


logger = logging.getLogger(__file__)


# ===================================
# kdtree
# ===================================


def _histogram_encode(cdr3, compression):
    dimension = int(np.ceil(len(aminoacids) / compression))
    position_map = {
        char: int(np.floor(index / compression))
        for index, char in enumerate(aminoacids)
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
    seqs : iterable of strings
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
        return _make_output(ans, output_type, seqs)
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
    return _make_output(triplets, output_type, seqs)


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


class LookupDB:
    """
    Lookup string variants in a dictionary.

    The dictionary has sequences as keys, and list of sequence indices as values.

    Parameters
    ----------
    seqs : iterable of strings
        list of sequences
    """

    def __init__(self, seqs):
        self.seqs = seqs
        self.seq_dict = {}
        for index, seq in enumerate(seqs):
            if seq not in self.seq_dict:
                self.seq_dict[seq] = []
            self.seq_dict[seq].append(index)

    def lookup(
        self,
        seqs2,
        max_edits=1,
        pdist_mode=False,
        custom_distance=None,
        max_custom_distance=float("inf"),
        output_type="triplets",
        progress=False,
    ):
        """
        Query the database

        Parameters
        ----------
        seq2 : iterable of strings or None
            list of query sequences
        max_edits: int
            maximum number of edits
        pdist_mode: Boolean
            if True, assume seqs2=seqs and filter diagonal
        custom_distance : Function(str1, str2) or "hamming"
            custom distance metric
        max_custom_distance : float
            maximum distance to include in the result, ignored if custom distance is not supplied
        output_type: string
            format of returns, can be "triplets", "coo_matrix", or "ndarray"
        progress : bool
            show progress bar

        Returns
        -------
        neighbors : array of 3D-tuples, sparse matrix, or dense matrix
            neigbors along with their edit distances according to the given output_type
            if "triplets" returns are [(x_index, y_index, edit_distance)]
            if "coo_matrix" returns are scipy's sparse matrix where C[i,j] = distance(X_i, X_j) or 0 if not neighbor
            if "ndarray" returns numpy's 2d array representing dense matrix
        """

        ans = []
        is_hamming = custom_distance == "hamming"
        if is_hamming:
            custom_distance = _hamming_replacement
        elif custom_distance is None:
            custom_distance = levenshtein

        if progress:
            seqs2_loop = tqdm.auto.tqdm(enumerate(seqs2), total=len(seqs2))
        else:
            seqs2_loop = enumerate(seqs2)

        for x_index, seq in seqs2_loop:
            neighbors = _generate_neighbors(seq, max_edits, is_hamming)
            for possible_edit, edit_distance in neighbors.items():
                if possible_edit in self.seq_dict:
                    for y_index in self.seq_dict[possible_edit]:
                        if x_index == y_index:
                            continue
                        if custom_distance in (None, "hamming"):
                            ans.append((x_index, y_index, edit_distance))
                        else:
                            dist = custom_distance(seq, possible_edit)
                            if dist <= max_custom_distance:
                                ans.append((x_index, y_index, dist))
        return _make_output(ans, output_type, self.seqs, seqs2)


def hash_based(
    seqs,
    max_edits=1,
    max_returns=None,
    n_cpu=1,
    custom_distance=None,
    max_custom_distance=float("inf"),
    output_type="triplets",
    progress=False,
):
    """
    List all neighboring CDR3B sequences efficiently for small edit distances.
    The idea is to list all possible sequences within a given distance and lookup the dictionary if it exists.
    This implementation is faster than kdtree implementation for max_edits == 1

    Parameters
    ----------
    seqs : iterable of strings
        list of CDR3B sequences
    max_edits : int
        maximum edit distance defining the neighbors
    max_returns : int or None
        not implemented
    n_cpu : int
        not implemented
    custom_distance : Function(str1, str2) or "hamming"
        custom distance metric
    max_custom_distance : float
        maximum distance to include in the result, ignored if custom distance is not supplied
    output_type: string
        format of returns, can be "triplets", "coo_matrix", or "ndarray"
    progress : bool
        show progress bar

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

    lookupdb = LookupDB(seqs)

    return lookupdb.lookup(
        seqs,
        max_edits=max_edits,
        pdist_mode=True,
        custom_distance=custom_distance,
        max_custom_distance=max_custom_distance,
        output_type=output_type,
        progress=progress,
    )


# ===================================
# symdel
# ===================================


def _comb_gen(seq, max_edits):
    """
    Generate all deletion variants up to a maximum number of deletions.
    """
    _len, ans = len(seq), set([seq])
    for edit in range(1, max_edits + 1):
        for indexes in combinations(range(_len), edit):
            new_seq, offset = [], 0
            for index in indexes:
                new_seq.append(seq[offset:index])
                offset = index + 1
            new_seq.append(seq[offset:_len])
            ans.add("".join(new_seq))
    return ans


class SymdelDB:
    """
    Generate a deletion variant dictionary.

    The dictionary has deletion variants as keys, and list of sequence indices
    as values.

    Parameters
    ----------
    seqs : iterable of strings
        list of sequences
    max_edits : int
        maximum deletion distance
    """

    def __init__(self, seqs, max_edits):
        self.seqs = seqs
        self.max_edits = max_edits
        self.variant_dict = {}
        for i, seq in enumerate(seqs):
            for comb in _comb_gen(seq, max_edits):
                if comb in self.variant_dict:
                    self.variant_dict[comb].append(i)
                else:
                    self.variant_dict[comb] = [i]

    def lookup(
        self,
        seqs2,
        custom_distance=None,
        max_custom_distance=float("inf"),
        output_type="triplets",
        progress=False,
    ):
        """
        Query the database

        Parameters
        ----------
        seq2 : iterable of strings or None
            list of query sequences

        custom_distance : Function(str1, str2) or "hamming"
            custom distance function to use, must statisfy 4 properties of
            distance
            (https://en.wikipedia.org/wiki/Distance#Mathematical_formalization)

        max_custom_distance : float
            maximum distance to include in the result, ignored if custom
            distance is not supplied

        output_type: string
            format of returns, can be "triplets", "coo_matrix", or "ndarray"

        progress : bool
            show progress bar

        Returns
        -------
        neighbors : array of 3D-tuples, sparse matrix, or dense matrix
            Neigbors along with their edit distances according to the given
            output_type. If "triplets" returns are [(seqs_index, seqs2_index,
            edit_distance)]. If "coo_matrix" returns are scipy's sparse matrix
            where C[i,j] = distance(seqs[i], seqs2[j]) or 0 if not neighbor. If
            "ndarray" returns numpy's 2d array representing dense matrix.
        """

        ans = []
        threshold = max_custom_distance
        if custom_distance in (None, "hamming") or max_custom_distance == float("inf"):
            threshold = self.max_edits
        if custom_distance == "hamming":
            custom_distance = _hamming_replacement
        elif custom_distance is None:
            custom_distance = levenshtein

        if progress:
            seqs2_loop = tqdm.auto.tqdm(enumerate(seqs2), total=len(seqs2))
        else:
            seqs2_loop = enumerate(seqs2)

        for seqs2_idx, seq in seqs2_loop:
            seqs_indices = set()
            for comb in _comb_gen(seq, self.max_edits):
                if comb not in self.variant_dict:
                    continue
                for seqs_idx in self.variant_dict[comb]:
                    seqs_indices.add(seqs_idx)
            for seqs_idx in seqs_indices:
                dist = custom_distance(seqs2[seqs2_idx], self.seqs[seqs_idx])
                if dist > threshold:
                    continue
                ans.append((seqs_idx, seqs2_idx, dist))

        return _make_output(ans, output_type, self.seqs, seqs2)


def _hamming_replacement(seq_a, seq_b):
    if len(seq_a) != len(seq_b):
        return np.inf
    return hamming(seq_a, seq_b)


def symdel(
    seqs,
    max_edits=1,
    max_returns=None,
    n_cpu=1,
    custom_distance=None,
    max_custom_distance=float("inf"),
    output_type="triplets",
    seqs2=None,
    progress=False,
):
    """
    List all neighboring sequences efficiently within the given distance. This
    is an improved version over the hash-based.

    If seqs2 is not provided, every sequences are compared against every other
    sequences resulting in :math:`N(seqs)**2` combinations. Otherwise, seqs are
    compared against seqs2 resulting in :math:`N(seqs)*N(seqs2)` combinations.

    Parameters
    ----------
    seqs : iterable of strings
        list of sequences

    max_edits : int
        maximum edit distance defining the neighbors

    max_returns : int or None
        ignored

    n_cpu : int
        ignored

    custom_distance : Function(str1, str2) or "hamming"
        custom distance function to use, must statisfy 4 properties of distance
        (https://en.wikipedia.org/wiki/Distance#Mathematical_formalization)

    max_custom_distance : float
        maximum distance to include in the result, ignored if custom distance is not supplied

    output_type: string
        format of returns, can be "triplets", "coo_matrix", "ndarray"

    seq2 : iterable of strings or None
        another list of sequences to compare against

    progress : bool
        show progress bar

    Returns
    -------
    neighbors : array of 3D-tuples, sparse matrix, or dense matrix
        Neigbors along with their edit distances according to the given
        output_type. If "triplets" returns are [(seqs_index, seqs2_index,
        edit_distance)]. If "coo_matrix" returns are scipy's sparse matrix
        where C[i,j] = distance(seqs[i], seqs2[j]) or 0 if not neighbor. If
        "ndarray" returns numpy's 2d array representing dense matrix.
    """

    _check_common_input(
        seqs,
        max_edits,
        max_returns,
        n_cpu,
        custom_distance,
        max_custom_distance,
        output_type,
        seqs2,
    )
    symdeldb = SymdelDB(seqs, max_edits)

    if seqs2 is None:
        ans = set()
        threshold = max_custom_distance
        if custom_distance in (None, "hamming") or max_custom_distance == float("inf"):
            threshold = max_edits
        if custom_distance == "hamming":
            custom_distance = _hamming_replacement
        elif custom_distance is None:
            custom_distance = levenshtein

        for values in symdeldb.variant_dict.values():
            if len(values) == 1:
                continue
            for i, j in combinations(values, 2):
                dist = custom_distance(seqs[i], seqs[j])
                if dist > threshold:
                    continue
                ans.add((i, j, dist))
                ans.add((j, i, dist))
        return _make_output(ans, output_type, seqs, seqs2)

    return symdeldb.lookup(
        seqs2,
        custom_distance=custom_distance,
        max_custom_distance=max_custom_distance,
        output_type=output_type,
        progress=progress,
    )


# ===================================
# combined function
# ===================================


def nearest_neighbor(
    seqs,
    max_edits=1,
    max_returns=None,
    n_cpu=1,
    custom_distance=None,
    max_custom_distance=float("inf"),
    output_type="triplets",
    seqs2=None,
):
    """
    List all neighboring sequences efficiently within a given distance.
    The distance can be given in terms of hamming, levenshtein, or custom.

    If seqs2 is not provided, every sequence is compared against every other sequence.

    Parameters
    ----------
    seqs : iterable of strings
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

    return symdel(
        seqs,
        max_edits,
        max_returns,
        n_cpu,
        custom_distance,
        max_custom_distance,
        output_type,
        seqs2,
    )


def _vdists_lookup(df: pd.DataFrame, row_labels, col_labels):
    # Add "*01" to the end of a V gene symbol if no allele specifier present
    row_labels = row_labels.str.replace(r"^([-/\w]+?)$", r"\1*01", regex=True)
    col_labels = col_labels.str.replace(r"^([-/\w]+?)$", r"\1*01", regex=True)

    values = df.values

    ridx = df.index.get_indexer(row_labels)
    cidx = df.columns.get_indexer(col_labels)

    if (ridx == -1).any():
        first_offending_idx = (ridx == -1).argmax()
        raise ValueError(
            f"{row_labels.iloc[first_offending_idx]} is not recognized as a valid V gene"
        )

    if (cidx == -1).any():
        first_offending_idx = (cidx == -1).argmax()
        raise ValueError(
            f"{col_labels.iloc[first_offending_idx]} is not recognized as a valid V gene"
        )

    flat_index = ridx * len(df.columns) + cidx
    return values.flat[flat_index]


def nearest_neighbor_tcrdist(
    df: pd.DataFrame,
    chain="beta",
    max_edits=2,
    edit_on_trimmed=True,
    max_tcrdist=20,
    tcrdist_kwargs={},
    df2: pd.DataFrame = None,
    **kwargs,
):
    """
    List all neighboring TCR sequences efficiently within a given edit and
    TCRdist radius.

    [Requires optional dependency pwseqdist]

    Parameters
    ----------
    chain: 'alpha', 'beta', or 'both'
        if both finds candidate neighbors using the beta chain, but filter on
        paired sequence at the end

    max_edits : only return neighbors up to <= this edit distance

    edit_on_trimmed : boolean
        apply TCRdist trimming on sequences before calculating edit distance

    max_tcrdist : only return neighbor up to <= this TCR distance

    tcrdist_kwargs: dict
        customized parameters for TCRdist calculation

    **kwargs : passed on to nearest_neighbor function

    Returns
    --------
    sparse matrix in (i, j, dist) format

    """
    # to reproduce standard TCRdist we multiply the CDR3 distance with three
    # we also need to multiply the gap penalty by this factor
    tcrdist_kwargs_this = dict(
        use_numba=True,
        fixed_gappos=False,
        ntrim=3,
        ctrim=2,
        dist_weight=3,
        gap_penalty=4 * 3,
    )
    tcrdist_kwargs_this.update(tcrdist_kwargs)

    if "output_type" in kwargs and kwargs["output_type"] != "triplets":
        logger.warning('output_type must be set to "triplets" for this function')
        kwargs["output_type"] = "triplets"

    def get_symdel_neighbors(
        chain_letter: Literal["A", "B"],
    ) -> List[Tuple[int, int, int]]:
        if edit_on_trimmed:
            ntrim = tcrdist_kwargs_this["ntrim"]
            ctrim = tcrdist_kwargs_this["ctrim"]
            seqs = list(df[f"CDR3{chain_letter}"].str[ntrim:-ctrim])
            if df2 is not None:
                seqs2 = list(df2[f"CDR3{chain_letter}"].str[ntrim:-ctrim])
            else:
                seqs2 = None
            return nearest_neighbor(seqs, max_edits=max_edits, seqs2=seqs2, **kwargs)
        else:
            seqs = list(df[f"CDR3{chain_letter}"])
            if df2 is not None:
                seqs2 = list(df2[f"CDR3{chain_letter}"])
            else:
                seqs2 = None
            return nearest_neighbor(seqs, max_edits=max_edits, seqs2=seqs2, **kwargs)

    def pairwise_sparse_within(df, pairs, chain_letter):
        return pwseqdist.apply_pairwise_sparse(
            metric=pwseqdist.metrics.nb_vector_tcrdist,
            seqs=df[f"CDR3{chain_letter}"],
            pairs=pairs,
            **tcrdist_kwargs_this,
        )

    def pairwise_sparse_cross(df, df2, pairs, chain_letter):
        adjusted_pairs = pairs.copy()
        adjusted_pairs[:, 1] += len(df)
        all_cdr3s = pd.concat([df[f"CDR3{chain_letter}"], df2[f"CDR3{chain_letter}"]])
        return pwseqdist.apply_pairwise_sparse(
            metric=pwseqdist.metrics.nb_vector_tcrdist,
            seqs=all_cdr3s,
            pairs=adjusted_pairs,
            **tcrdist_kwargs_this,
        )

    if chain in ("alpha", "beta"):
        chain_letter = chain[0].upper()
        triplets = get_symdel_neighbors(chain_letter)
        triplets_arr = np.array(triplets)
        pair_candidates = triplets_arr[:, :2]

        vdists_path = Path(__file__).resolve().parent / "data" / f"vdists_{chain}.csv"
        vdists = pd.read_csv(vdists_path, index_col=0)

        if df2 is not None:
            tcrdist = _vdists_lookup(
                vdists,
                df[f"TR{chain_letter}V"].iloc[pair_candidates[:, 0]],
                df2[f"TR{chain_letter}V"].iloc[pair_candidates[:, 1]],
            ) + pairwise_sparse_cross(df, df2, pair_candidates, chain_letter)
        else:
            tcrdist = _vdists_lookup(
                vdists,
                df[f"TR{chain_letter}V"].iloc[pair_candidates[:, 0]],
                df[f"TR{chain_letter}V"].iloc[pair_candidates[:, 1]],
            ) + pairwise_sparse_within(df, pair_candidates, chain_letter)
    else:  # chain == "both"
        triplet_candidates_a = set(
            [triplet[:2] for triplet in get_symdel_neighbors("A")]
        )
        triplet_candidates_b = set(
            [triplet[:2] for triplet in get_symdel_neighbors("B")]
        )
        triplets = [
            pair + (0,)
            for pair in triplet_candidates_a.intersection(triplet_candidates_b)
        ]
        triplets_arr = np.array(triplets)
        pair_candidates = triplets_arr[:, :2]

        vdists_a_path = Path(__file__).resolve().parent / "data" / f"vdists_alpha.csv"
        vdists_b_path = Path(__file__).resolve().parent / "data" / f"vdists_beta.csv"
        vdists_a = pd.read_csv(vdists_a_path, index_col=0)
        vdists_b = pd.read_csv(vdists_b_path, index_col=0)

        if df2 is not None:
            tcrdist = (
                _vdists_lookup(
                    vdists_a,
                    df[f"TRAV"].iloc[pair_candidates[:, 0]],
                    df2[f"TRAV"].iloc[pair_candidates[:, 1]],
                )
                + _vdists_lookup(
                    vdists_b,
                    df[f"TRBV"].iloc[pair_candidates[:, 0]],
                    df2[f"TRBV"].iloc[pair_candidates[:, 1]],
                )
                + pairwise_sparse_cross(df, df2, pair_candidates, "A")
                + pairwise_sparse_cross(df, df2, pair_candidates, "B")
            )
        else:
            tcrdist = (
                _vdists_lookup(
                    vdists_a,
                    df[f"TRAV"].iloc[pair_candidates[:, 0]],
                    df[f"TRAV"].iloc[pair_candidates[:, 1]],
                )
                + _vdists_lookup(
                    vdists_b,
                    df[f"TRBV"].iloc[pair_candidates[:, 0]],
                    df[f"TRBV"].iloc[pair_candidates[:, 1]],
                )
                + pairwise_sparse_within(df, pair_candidates, "A")
                + pairwise_sparse_within(df, pair_candidates, "B")
            )

    triplets_arr[:, 2] = tcrdist

    return triplets_arr[triplets_arr[:, 2] <= max_tcrdist]


def calculate_sceptrdist_sparse(edges, tcr_data_array):
    """
    Efficiently calculate sparse pairwise distances between vector representations of TCRs.
    """
    # Allocate an array to store distances
    sceptrdist = np.empty(len(edges), dtype=np.float32)
    # Loop over edges and calculate distances
    for i in range(len(edges)):
        tcr1_idx, tcr2_idx = edges[i]
        # Extract TCR vectors from tcr_data_array (NumPy 2D array)
        tcr1_vector = tcr_data_array[tcr1_idx]
        tcr2_vector = tcr_data_array[tcr2_idx]
        # Compute the Euclidean distance
        dist = np.sqrt(np.sum((tcr1_vector - tcr2_vector) ** 2))
        # Store the distance in the result array
        sceptrdist[i] = dist
    return sceptrdist


def nearest_neighbor_sceptrdist(
    df, chain="beta", max_edits=2, max_sceptrdist=1.0, **kwargs
):
    """
    List all neighboring TCR sequences efficiently within a given edit and SCEPTR radius.

    [Requires optional dependency sceptr]

    Parameters
    ----------
    chain: 'alpha', 'beta'
        chain to use for edit distance prefiltering
    max_edits : only return neighbors up to <= this edit distance
    max_sceptrdist : only return neighbor up to <= this TCR distance
    **kwargs : passed on to nearest_neighbor function

    Returns
    --------
    sparse matrix in (i, j, dist) format

    """
    chain_letter = chain[0].upper()
    seqs = list(df[f"CDR3{chain_letter}"])
    neighbors = nearest_neighbor(seqs, max_edits=max_edits, **kwargs)
    neighbors_arr = np.array(neighbors, dtype=object)
    edges = neighbors_arr[:, :2]
    tcr_data_array = sceptr.calc_vector_representations(df)
    sceptrdist = calculate_sceptrdist_sparse(edges, tcr_data_array)
    neighbors_arr[:, 2] = sceptrdist
    return neighbors_arr[neighbors_arr[:, 2] <= max_sceptrdist]


# ===================================
# util
# ===================================


def _flatten_array(nested_array):
    return list(chain(*nested_array))


def _check_common_input(
    seqs,
    max_edits,
    max_returns,
    n_cpu,
    custom_distance,
    max_cust_dist,
    output_type,
    seqs2=None,
):
    assert len(seqs) > 0, "length must be greater than 0"
    try:
        for seq in seqs:
            assert type(seq) in {
                str,
                np.str_,
            }, "sequences must be an iterable of string"
    except TypeError:
        assert False, "sequences must be an iterable of string"
    assert type(max_edits) == int and max_edits > 0, (
        "max_edits must be a positive integer"
    )
    assert (type(max_returns) == int and max_returns > 0) or max_returns is None, (
        "max_returns must be a positive integer or None"
    )
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
    assert type(max_cust_dist) in (int, float) and max_cust_dist >= 0, (
        "max_custom_distance must be a non-negative number"
    )
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
    except TypeError:
        assert seqs2 is None, "sequences2 must be an iterable of string or None"


def _make_output(triplets, output_type, seqs, seqs2=None):
    if output_type == "triplets":
        if type(triplets) != list:
            return list(triplets)
        return triplets

    row, col, data = [], [], []
    for triplet in triplets:
        row += [triplet[0]]
        col += [triplet[1]]
        data += [triplet[2]]

    shape = (len(seqs), len(seqs)) if seqs2 is None else (len(seqs), len(seqs2))
    coo_result = coo_matrix((data, (row, col)), shape=shape)
    return coo_result if output_type == "coo_matrix" else coo_result.toarray()
