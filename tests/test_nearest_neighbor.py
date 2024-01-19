import pytest
from pyrepseq.nn import hash_based, kdtree, symspell
from itertools import product
from Levenshtein import distance
import numpy as np
import pandas as pd

ALGORITHMS = [kdtree, hash_based, symspell]
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
    # normal set equallity is slow in python, so we implement one
    set_a, set_b = set(list_a), set(list_b)
    return len(set_a) == len(set_b) == len(set_a & set_b)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_basic(algorithm):
    test_input = ["CAAA", "CDDD", "CADA", "CAAK"]
    test_output = [(0, 2, 1), (0, 3, 1), (2, 0, 1), (3, 0, 1)]
    assert set_equal(algorithm(test_input, max_edits=1), test_output)

    fallback_version = fallback(test_input)
    assert set_equal(fallback_version, test_output)

    test_output = [(0, 2, 1), (2, 0, 1), (3, 0, 1)]

    assert set_equal(algorithm(test_input, max_edits=1,
                               max_returns=1), test_output)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_duplicate(algorithm):
    test_input = ['CAAA', 'CDDD', 'CADA', 'CAAA']
    test_output = [(0, 3, 0), (0, 2, 1), (2, 0, 1),
                   (2, 3, 1), (3, 0, 0), (3, 2, 1)]
    assert set_equal(algorithm(test_input, max_edits=1), test_output)

    fallback_version = fallback(test_input)
    assert set_equal(fallback_version, test_output)

    output = set(algorithm(test_input, max_edits=1, max_returns=1))
    assert len(output) == 3 == len(set(test_output) & output)


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

    output = set(algorithm(test_input, max_edits=1, max_returns=1,
                           custom_distance=distance))
    assert len(output) == 3 == len(output & set(test_output))

    test_output = []
    assert set_equal(algorithm(test_input, max_edits=1, max_returns=1,
                               custom_distance=distance,
                               max_custom_distance=0), test_output)


def test_seq2():
    test_input1 = ["CAAA", "CADA", "CAAA", "CDKD", "CAAK"]
    test_input2 = ["CDDD", "CAAK"]
    test_output = [(1,0,1), (1,2,1), (0,3,1),(1,4,0)]
    assert set_equal(symspell(test_input1, max_edits=1,
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
