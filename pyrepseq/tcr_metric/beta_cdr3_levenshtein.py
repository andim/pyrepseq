from numpy import ndarray
from pandas import DataFrame
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein
from scipy.spatial import distance

from pyrepseq.tcr_metric.tcr_metric import TcrMetric


class BetaCdr3Levenshtein(TcrMetric):
    name = "Beta CDR3 Levenshtein"
    distance_bins = range(26)

    def calc_cdist_matrix(
        self, anchor_tcrs: DataFrame, comparison_tcrs: DataFrame
    ) -> ndarray:
        anchor_tcr_cdr3bs = anchor_tcrs.CDR3B
        comparison_tcr_cdr3bs = comparison_tcrs.CDR3B
        cdist_matrix = process.cdist(
            anchor_tcr_cdr3bs, comparison_tcr_cdr3bs, scorer=Levenshtein.distance
        )
        return cdist_matrix

    def calc_pdist_vector(self, tcrs: DataFrame) -> ndarray:
        pdist_matrix = self.calc_cdist_matrix(tcrs, tcrs)
        pdist_vector = distance.squareform(pdist_matrix, checks=False)
        return pdist_vector
