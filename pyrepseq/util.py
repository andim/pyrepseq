import numpy as np
import itertools
from scipy.sparse import coo_matrix
import re


def flatten_array(nested_array):
    return list(itertools.chain(*nested_array))


def ensure_numpy(arr_like):
    module = type(arr_like).__module__
    if module == 'pandas.core.series':
        return arr_like.to_numpy()
    if module == 'numpy':
        return arr_like
    return np.array(arr_like)


def check_common_input(seqs, max_edits, max_returns, n_cpu,
                       custom_distance, max_cust_dist, output_type):
    assert len(seqs) > 0, \
        'length must be greater than 0'
    try:
        for seq in seqs:
            assert type(seq) in {str, np.str_}, \
                'sequences must be an iterable of string'
            assert re.match(r'^[ACDEFGHIKLMNPQRSTVWY]+$', seq), \
                'sequences must contain only valid amino acids'
    except TypeError:
        assert False,\
            'sequences must be an iterable of string'
    assert type(max_edits) == int and max_edits > 0, \
        'max_edits must be a positive integer'
    assert (type(max_returns) == int and max_returns > 0) or \
        max_returns is None, 'max_returns must be a positive integer or None'
    assert type(n_cpu) == int and n_cpu > 0,\
        'n_cpu must be a positive integer'
    try:
        first = next(seqs.__iter__())
        assert custom_distance in (None, 'hamming') or \
            custom_distance(first, first) == 0,\
            'custom_distance must be None or custom_distance(x1,x2)==0'
    except AssertionError:
        raise
    except:
        assert False, 'custom_distance evaluation fails'
    assert type(max_cust_dist) in (int, float) and max_cust_dist >= 0, \
        'max_custom_distance must be a non-negative number'
    assert output_type in {'coo_matrix', 'triplets', 'ndarray'}, \
        'output must be either coo_matrix, triplets, or ndarray'


def make_output(triplets, output_type):
    if output_type == 'triplets':
        return triplets

    row, col, data, shape = [], [], [], len(seqs)
    for triplet in triplets:
        row += [triplet[0], triplet[1]]
        column += [triplet[1], triplet[0]]
        data += [triplet[2], triplet[2]]
    coo_result = coo_matrix((data, (row, col)), shape=(shape, shape))

    return coo_result if output_type == 'coo_matrix' else coo_result.toarray()
