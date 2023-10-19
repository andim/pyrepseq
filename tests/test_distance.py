from pyrepseq.distance import *
import numpy as np
import scipy.spatial.distance
import Levenshtein


def test_pdist():
    strings = ["abs", "asdf", "adfl"]
    distances = pdist(strings)
    distances_scipy = scipy.spatial.distance.pdist(
        np.array(strings, dtype=str)[:, np.newaxis],
        metric=lambda u, v: Levenshtein.distance(u[0], v[0]),
    )
    assert all(distances == distances_scipy)


def test_cdist():
    strings = ["abs", "asdf", "adfl"]
    distances = cdist(strings, strings)
    array = np.array(strings, dtype=str)[:, np.newaxis]
    distances_scipy = scipy.spatial.distance.cdist(
        array, array, metric=lambda u, v: Levenshtein.distance(u[0], v[0])
    )
    assert (distances == distances_scipy).all()


def test_nndist_hamming():
    reference = set(["ABC", "DEF"])
    assert nndist_hamming("ABC", reference) == 0
    assert nndist_hamming("ABD", reference) == 1
    assert nndist_hamming("EBD", reference) == 2
    assert nndist_hamming("EGD", reference) == 3


def test_levenshtein_neighbors():
    neighbors = list(levenshtein_neighbors("A"))
    # deletion
    assert "" in neighbors
    # replacement
    assert "C" in neighbors
    # insertions
    assert "CA" in neighbors
    assert "AC" in neighbors
    assert "AA" in neighbors
    # uniqueness
    assert len(neighbors) == len(set(neighbors))

    neighbors = list(levenshtein_neighbors("ACC"))
    assert len(neighbors) == len(set(neighbors))


def test_hamming_neighbors():
    neighbors = list(hamming_neighbors("A"))
    # replacement
    assert "C" in neighbors
    # correct length
    assert len(neighbors) == (len(aminoacids) - 1)
    neighbors = list(hamming_neighbors("AAA", variable_positions=[1]))
    assert len(neighbors) == (len(aminoacids) - 1)


def test_next_nearest_neighbors():
    neighborhood = lambda x: hamming_neighbors(x, alphabet="ABC")
    neighbors = next_nearest_neighbors("AAAA", neighborhood=neighborhood, maxdistance=2)
    assert "ABAC" in neighbors


def test_find_neighbor_pairs():
    pairs = find_neighbor_pairs(["AA", "AC"])
    assert (("AA", "AC") in pairs) or (("AC", "AA") in pairs)
    assert len(pairs) == 1


def test_load_pcDelta_background():
    back, bins = load_pcDelta_background()
