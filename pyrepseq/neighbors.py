from .main import aminoacids
import numpy as np

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

    variable_positions: iterable of positions to be varied
    (default: all)
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

    neighborhood: neighborhood iterator
    maxdistance : go up to maxdistance nearest neighbor
    """
   
    neighbors = [list(neighborhood(x))]
    distance = 1
    while distance < maxdistance:
        neighbors_dist = []
        for x in neighbors[-1]:
            neighbors_dist.extend(neighborhood(x))
        neighbors.append(set(neighbors_dist))
        distance += 1
    return set(_flatten_list(neighbors))
 
def find_neighbor_pairs(seqs, neighborhood=hamming_neighbors):
    """Find neighboring sequences in a list of unique sequences.

    neighborhood: callable returning an iterable of neighbors

    returns: tuple (seq1, seq2)
    """
    reference = set(seqs)
    pairs = []
    for x in set(seqs):
        for y in (set(neighborhood(x)) & reference):
            pairs.append((x, y))
        reference.remove(x)
    return pairs

def find_neighbor_pairs_index(seqs, neighborhood=hamming_neighbors):
    """Find neighboring sequences in a list of unique sequences.

    neighborhood: callable returning an iterable of neighbors

    returns: tuple (index1, index2)
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

    seqs: list of sequences
    """
    reference = set(seqs)
    return np.array([len(set(neighborhood(seq)) & reference) for seq in seqs])
