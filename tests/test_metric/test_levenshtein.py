import numpy as np
from pyrepseq.metric import Levenshtein


class TestLevenshtein:
    metric = Levenshtein()

    def test_cdist(self):
        anchors = ["abc","bcd"]
        comparisons = ["aaa", "bbb"]
        result = self.metric.calc_cdist_matrix(anchors,comparisons)
        expected = np.array([[2,2],[3,2]])

        assert np.array_equal(result, expected)

    def test_pdist(self):
        instances = ["abc","bcd"]
        result = self.metric.calc_pdist_vector(instances)
        expected = np.array([2])

        assert np.array_equal(result, expected)