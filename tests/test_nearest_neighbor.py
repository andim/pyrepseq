import pytest
from pyrepseq.nn import hash_based, kdtree, symdel, nearest_neighbor_tcrdist, SymdelDB
from itertools import product
from Levenshtein import distance
import numpy as np
import pandas as pd

ALGORITHMS = [kdtree, hash_based, symdel]
fallback_cache = None


def fallback(seqs, max_edits=1):
    """
    basic levenshtein comparison
    """
    ans = []
    for i in range(len(seqs)):
        for j in range(len(seqs)):
            if i == j:
                continue
            dist = distance(seqs[j], seqs[i], score_cutoff=max_edits)
            if dist <= max_edits:
                ans += [(i, j, dist)]
    return ans


def set_equal(list_a, list_b):
    "Test for equality regardless of order"
    if len(list_a) != len(list_b):
        return False
    # normal set equality is slow in python, so we implement one
    set_a, set_b = set(list_a), set(list_b)
    return len(set_a) == len(set_b) == len(set_a & set_b)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_basic(algorithm):
    test_input = ["CAAA", "CDDD", "CADA", "CAAK"]
    test_output = [(0, 2, 1), (0, 3, 1), (2, 0, 1), (3, 0, 1)]
    assert set_equal(algorithm(test_input, max_edits=1), test_output)

    fallback_version = fallback(test_input)
    assert set_equal(fallback_version, test_output)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_duplicate(algorithm):
    test_input = ['CAAA', 'CDDD', 'CADA', 'CAAA']
    test_output = [(0, 3, 0), (0, 2, 1), (2, 0, 1),
                   (2, 3, 1), (3, 0, 0), (3, 2, 1)]
    result = algorithm(test_input, max_edits=1)
    assert set_equal(result, test_output)

    fallback_version = fallback(test_input)
    assert set_equal(fallback_version, test_output)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_compatibility(algorithm):
    test_input = np.array(["CAAA", "CDDD", "CADA"])
    test_output = [(0, 2, 1), (2, 0, 1)]
    assert set_equal(algorithm(test_input, max_edits=1), test_output)

    test_input = pd.Series(['CAAA', 'CDDD', 'CADA'])
    assert set_equal(algorithm(test_input, max_edits=1), test_output)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_hamming(algorithm):
    test_input = ["CAAA", "CDDD", "CADA", "CAAAD"]
    test_output = [(0, 2, 1), (2, 0, 1)]
    assert set_equal(algorithm(test_input, custom_distance='hamming',
                               max_edits=1), test_output)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_custom_distance(algorithm):
    test_input = ["CAAA", "CDDD", "CADA", "CAAK"]
    test_output = [(0, 2, 1), (0, 3, 1), (2, 0, 1), (3, 0, 1)]
    assert set_equal(algorithm(test_input, max_edits=1,
                               custom_distance=distance), test_output)

    test_output = []
    assert set_equal(algorithm(test_input, max_edits=1,
                               custom_distance=distance,
                               max_custom_distance=0), test_output)


def test_symdel_lookup():
    test_reference = ["CAAA", "CDDD", "CADA", "CAAK"]
    test_query = ["CAAF", "CCCC"]
    test_output = [(0, 0, 1), (0, 3, 1)]

    symdeldb = SymdelDB(test_reference, max_edits=1)
    result = symdeldb.lookup(test_query)
    assert set_equal(result, test_output)

    test_duplicate = ["CDDD", "CCCC"]
    test_output = [(0, 1, 0)]
    result = symdeldb.lookup(test_duplicate)
    assert set_equal(result, test_output)

def test_tcrdist():
    df = pd.DataFrame(columns=['CDR3B','TRBV'], data=
        [
        ['CASSGETGQPQHF','TRBV6-1*01'],
        ['CASSTQGIHEQYF','TRBV9*01'],
        ['CASSTQGIHEQYF','TRBV9*01'],
        ['CAWSF','TRBV30*01'],
        ['CSATGYNEQFF','TRBV20-1*01']
        ])
    results = nearest_neighbor_tcrdist(df, max_edits=2,
                             max_tcrdist=0);
    np.array_equal(results,np.array([[1, 2, 0],[2, 1, 0]]))

# symdel is the only algorithm supporting 2-seq mode
def test_two_sequence():
    seq1 = ["CAAA", "CDDD", "CADA", "CAAK"]
    seq2 = ["CIAA","CIAA"]
    output = np.array([[1,1],[0,0],[0,0],[0,0]])
    assert np.array_equal(output,symdel(seq1, seqs2=seq2, output_type='ndarray'))

    seq2 = ["CIII"]
    output = np.array([[0],[0],[0],[0]])
    assert np.array_equal(output,symdel(seq1, seqs2=seq2, output_type='ndarray'))

    output = np.array([[0,0,1,1],[0,0,0,0],[1,0,0,0],[1,0,0,0]])
    assert np.array_equal(output,symdel(seq1, output_type='ndarray'))
    assert np.array_equal(output,hash_based(seq1, output_type='ndarray'))
    assert np.array_equal(output,kdtree(seq1, output_type='ndarray'))


def test_seq2():
    test_input1 = ["CAAA", "CADA", "CAAA", "CDKD", "CAAK"]
    test_input2 = ["CDDD", "CAAK"]
    test_output = [(1,0,1), (1,2,1), (0,3,1),(1,4,0)]
    assert set_equal(symdel(test_input1, max_edits=1,
                               seqs2=test_input2), test_output)


@pytest.mark.skip(reason="very slow")
@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_bulk(algorithm):
    combinations1 = product("ACD", repeat=2)
    combinations2 = product("ACD", repeat=3)
    combinations3 = product("ACD", repeat=4)
    combinations4 = product("ACD", repeat=5)
    combinations5 = product("ACD", repeat=6)
    combinations6 = product("ACD", repeat=7)

    def to_string(x):
        return "C" + "".join(x)

    test_input = list(map(to_string, combinations1))
    test_input += list(map(to_string, combinations2))
    test_input += list(map(to_string, combinations3))
    test_input += list(map(to_string, combinations4))
    test_input += list(map(to_string, combinations5))
    test_input += list(map(to_string, combinations6))
    assert len(test_input) == 3276

    # cache it to save time
    global fallback_cache
    if fallback_cache is None:
        fallback_cache = fallback(test_input, max_edits=2)

    fuzzy_version = algorithm(test_input, max_edits=2, n_cpu=4)
    assert set_equal(fallback_cache, fuzzy_version)

def test_symdel_progress():
    test_input1 = ["CAAA", "CADA", "CAAA", "CDKD", "CAAK"]
    test_input2 = ["CDDD", "CAAK"]
    test_output = [(1,0,1), (1,2,1), (0,3,1),(1,4,0)]
    assert set_equal(symdel(test_input1, max_edits=1,
                               seqs2=test_input2, progress=True), test_output)
