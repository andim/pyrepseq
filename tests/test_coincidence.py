from pyrepseq import pc, pcDelta


def test_coincidence():
    p = pc(["AA", "AA"])
    assert p == 1.0
    p = pcDelta(["AA", "AA"], bins=0)
    assert p == 1.0
    p = pc(["AA", "AB"])
    assert p == 0.0
    p = pc(["AA", "AA", "AB"])
    assert p == 1 / 3.0
    p = pc(["AA", "AB"], ["AB", "AC"])
    assert p == 1 / 4.0
    p = pcDelta(["AA", "AB"], ["AB", "AC"], bins=0)
    assert p == 1 / 4.0
