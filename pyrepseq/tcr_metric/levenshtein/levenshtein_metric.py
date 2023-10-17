from abc import abstractmethod
from numpy import ndarray
from pandas import DataFrame
from pyrepseq.tcr_metric import TcrMetric
from rapidfuzz import process


class LevenshteinMetric(TcrMetric):
    @property
    @abstractmethod
    def columns_to_compare(self) -> str:
        pass

    def calc_cdist_matrix(
        self, anchor_tcrs: DataFrame, comparison_tcrs: DataFrame
    ) -> ndarray:
        cdists = [self._compute_cdist_for]
