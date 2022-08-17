from pyrepseq import pc

def test_coincidence():
    p = pc(['AA', 'AA'])
    assert p == 1.0
    p = pc(['AA', 'AB'])
    assert p == 0.0
    p = pc(['AA', 'AA', 'AB'])
    assert p == 1/3.
