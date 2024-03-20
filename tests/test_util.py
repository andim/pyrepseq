from pyrepseq.util import *

def test_consensus():
    seqs = ['CAF', 'CAF', 'CSF', 'CF']
    assert seqs_to_consensus(seqs) == 'CAF'
