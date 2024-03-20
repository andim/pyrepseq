from pyrepseq.util import *

def test_consensus():
    seqs = ['CAF', 'CAF', 'CSF']
    assert seqs_to_consensus(seqs, align=False) == 'CAF'
