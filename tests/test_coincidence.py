from pyrepseq import pC

def test_coincidence():
    p = pC(['AA', 'AA'])
    assert p == 1.0
    p = pC(['AA', 'AB'])
    assert p == 0.0
    p = pC(['AA', 'AA', 'AB'])
    assert p == 1/3.
