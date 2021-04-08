from .main import aminoacids

def levenshtein_neighbors(x, alphabet=aminoacids):
    """Iterator for Levenshtein neighbors of a sequence x"""
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
    """Iterator for Hamming neighbors of a sequence x"""
    for i in range(len(x)):
        for aa in alphabet:
            if aa == x[i]:
                continue
            yield x[:i]+aa+x[i+1:]
 
