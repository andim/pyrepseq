from numpy import ndarray
from pandas import DataFrame
from scipy.spatial import distance

from pyrepseq.tcr_metric import TcrMetric
from pyrepseq.tcr_metric.tcrdist.simplified_tcrdist_interface import TcrdistInterface


class BetaTcrdist(TcrMetric):
    name = "Beta tcrdist"
    distance_bins = range(0, 301, 5)

    _tcrdist_interface = TcrdistInterface()

    def calc_cdist_matrix(
        self, anchor_tcrs: DataFrame, comparison_tcrs: DataFrame
    ) -> ndarray:
        beta_cdist_matrices = self._tcrdist_interface.calc_beta_cdist_matrices(
            anchor_tcrs, comparison_tcrs
        )
        full_tcrdist_cdist = beta_cdist_matrices["tcrdist"]
        return full_tcrdist_cdist

    def calc_pdist_vector(self, tcrs: DataFrame) -> ndarray:
        pdist_matrix = self.calc_cdist_matrix(tcrs, tcrs)
        pdist_vector = distance.squareform(pdist_matrix, checks=False)
        return pdist_vector


class BetaCdr3Tcrdist(TcrMetric):
    name = "Beta CDR3 tcrdist"
    distance_bins = range(0, 81, 2)

    _tcrdist_interface = TcrdistInterface()

    def calc_cdist_matrix(
        self, anchor_tcrs: DataFrame, comparison_tcrs: DataFrame
    ) -> ndarray:
        beta_cdist_matrices = self._tcrdist_interface.calc_beta_cdist_matrices(
            anchor_tcrs, comparison_tcrs
        )
        cdr3_cdist = beta_cdist_matrices["cdr3_b_aa"]
        return cdr3_cdist

    def calc_pdist_vector(self, tcrs: DataFrame) -> ndarray:
        pdist_matrix = self.calc_cdist_matrix(tcrs, tcrs)
        pdist_vector = distance.squareform(pdist_matrix, checks=False)
        return pdist_vector
