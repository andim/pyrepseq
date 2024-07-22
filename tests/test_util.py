from pyrepseq.util import *

def test_consensus():
    seqs = ['CAF', 'CAF', 'CSF']
    assert seqs_to_consensus(seqs, align=False) == 'CAF'

def test_alignment():
    seqs = [
        'CATGGAAGNKLTF',
        'CATGGAAGNKLTF',
        'CATGGAAGNKLTF',
        'CATGGAAGNKLTF',
        'CAVGGAAGNKLTF',
        'CAGGGAAGNKLTF',
        'CATGGAAGNKLTF'
        ]
    # Error handling in case mafft-linsi is not installed at test time
    try:
        assert align_seqs(seqs) == seqs
    except OSError:
        pass
