from pyrepseq.neighbors import *

def test_levenshtein():
    neighbors = list(levenshtein_neighbors('A'))
    # deletion
    assert '' in neighbors
    # replacement
    assert 'C' in neighbors
    # insertions
    assert 'CA' in neighbors
    assert 'AC' in neighbors
    assert 'AA' in neighbors
    # uniqueness
    assert len(neighbors) == len(set(neighbors))

    neighbors = list(levenshtein_neighbors('ACC'))
    assert len(neighbors) == len(set(neighbors))

def test_hamming():
    neighbors = list(hamming_neighbors('A'))
    # replacement
    assert 'C' in neighbors
    # correct length
    assert len(neighbors) == (len(aminoacids)-1)
    neighbors = list(hamming_neighbors('AAA', variable_positions=[1]))
    assert len(neighbors) == (len(aminoacids)-1)

def test_next_nearest_neighbors():
    neighborhood = lambda x: hamming_neighbors(x, alphabet='ABC')
    neighbors = next_nearest_neighbors('AAAA', neighborhood=neighborhood, maxdistance=2)
    assert 'ABAC' in neighbors

def test_find_neighbor_pairs():
    pairs = find_neighbor_pairs(['AA', 'AC'])
    assert (('AA', 'AC') in pairs) or (('AC', 'AA') in pairs)
    assert len(pairs) == 1
