from pyrepseq import coincidence_probability

def test_coincidence():
    p = coincidence_probability(['AA', 'AA'])
    assert p == 1.0
    p = coincidence_probability(['AA', 'AB'])
    assert p == 0.0
    p = coincidence_probability(['AA', 'AA', 'AB'])
    assert p == 1/3.
