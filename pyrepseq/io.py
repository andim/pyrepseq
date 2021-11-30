from .main import isvalidaa

def isvalidcdr3(string):
    """
    returns True if string is a valid CDR3 alpha sequence

    Checks whether first aa is cysteine (C), and last aa is phenylalanine (F) or tryptophan (W).
    See http://www.imgt.org/IMGTScientificChart/Numbering/IMGTIGVLsuperfamily.html
    """
    return isvalidaa(string) and (string[0] == 'C') and (string[-1] in ['F', 'W'])
