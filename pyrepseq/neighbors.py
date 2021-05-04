from .main import aminoacids

def levenshtein_neighbors(x, alphabet=aminoacids):
    """Iterator over Levenshtein neighbors of a string x"""
    for i in range(len(x)):
        # deletion
        # only delete first amino acid when repeated
        if (i > 0) and (x[i] == x[i-1]):
            continue
        yield x[:i]+x[i+1:]
        # replacement
        for aa in alphabet:
            # do not replace same amino acid to avoid redundancy 
            if aa == x[i]:
                continue
            yield x[:i]+aa+x[i+1:]
    for i in range(len(x)+1):
        for aa in alphabet:
            if aa == x[i-1]:
                continue
            # insertion
            yield x[:i]+aa+x[i:]
    # insertion at end
#    for aa in alphabet:
#        yield x+aa

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
 
def find_neighbor_pairs(seqs, neighborhood=hamming_neighbors):
    """Find neighboring sequences in a list.

    neighborhood: callable returning an iterable of neighbors
    """
    reference = set(seqs)
    pairs = []
    for x in set(seqs):
        for y in neighborhood(x):
            if y in reference:
                pairs.append((x, y))
        reference.remove(x)
    return pairs
