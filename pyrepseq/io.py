from .main import isvalidaa

def isvalidcdr3(string):
    """
    returns True if string is a valid CDR3 sequence

    Checks the following:
        - first amino acid is a cysteine (C)
        - last amino acid is either phenylalanine (F) or tryptophan (W)
        - each amino acid is part of the standard amino acid alphabet
    See http://www.imgt.org/IMGTScientificChart/Numbering/IMGTIGVLsuperfamily.html
    """
    return isvalidaa(string) and (string[0] == 'C') and (string[-1] in ['F', 'W'])
