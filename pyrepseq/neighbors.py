from .main import aminoacids

def levenshtein_neighbors(x, alphabet=aminoacids):
    """Iterator over Levenshtein neighbors of a string x"""
    for i in range(len(x)):
        # deletion
        yield x[:i]+x[i+1:]
        for aa in alphabet:
            if aa == x[i]:
                continue
            # insertion
            # do not insert same amino acid to avoid redundancy 
            yield x[:i]+aa+x[i:]
            # replacement
            yield x[:i]+aa+x[i+1:]
    # insertion at end
    for aa in alphabet:
        yield x+aa

def hamming_neighbors(x, alphabet=aminoacids):
    """Iterator over Hamming neighbors of a string x"""
    for i in range(len(x)):
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
