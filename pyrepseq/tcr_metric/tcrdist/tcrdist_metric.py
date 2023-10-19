from abc import abstractmethod
from enum import Enum
from numpy import ndarray
from pandas import DataFrame
from pyrepseq.tcr_metric.tcr_metric import TcrMetric
from pyrepseq.tcr_metric.tcrdist.simplified_tcrdist_interface import TcrdistInterface
from scipy.spatial import distance
from typing import Iterable


tcrdist_interface = TcrdistInterface()


class TcrChain(Enum):
    ALPHA = 1
    BETA = 2


class TcrdistType(Enum):
    CDR3 = 1
    FULL = 2


class AbstractTcrdist(TcrMetric):
    @property
    @abstractmethod
    def _chains_to_compare(self) -> Iterable[TcrChain]:
        pass

    @property
    @abstractmethod
    def _tcrdist_type(self) -> TcrdistType:
        pass

    def calc_cdist_matrix(
        self, anchor_tcrs: DataFrame, comparison_tcrs: DataFrame
    ) -> ndarray:
        cdist_matrices = []

        if TcrChain.ALPHA in self._chains_to_compare:
            alpha_result = self._calc_alpha_cdist(anchor_tcrs, comparison_tcrs)
            cdist_matrices.append(alpha_result)
        if TcrChain.BETA in self._chains_to_compare:
            beta_result = self._calc_beta_cdist(anchor_tcrs, comparison_tcrs)
            cdist_matrices.append(beta_result)

        return sum(cdist_matrices)

    def _calc_alpha_cdist(
        self, anchor_tcrs: DataFrame, comparison_tcrs: DataFrame
    ) -> ndarray:
        result = tcrdist_interface.calc_alpha_cdist_matrices(
            anchor_tcrs, comparison_tcrs
        )
        if self._tcrdist_type is TcrdistType.CDR3:
            return result["cdr3_a_aa"]
        else:
            return result["tcrdist"]

    def _calc_beta_cdist(
        self, anchor_tcrs: DataFrame, comparison_tcrs: DataFrame
    ) -> ndarray:
        result = tcrdist_interface.calc_beta_cdist_matrices(
            anchor_tcrs, comparison_tcrs
        )
        if self._tcrdist_type is TcrdistType.CDR3:
            return result["cdr3_b_aa"]
        else:
            return result["tcrdist"]

    def calc_pdist_vector(self, tcrs: DataFrame) -> ndarray:
        pdist_matrix = self.calc_cdist_matrix(tcrs, tcrs)
        pdist_vector = distance.squareform(pdist_matrix, checks=False)
        return pdist_vector


class AlphaCdr3Tcrdist(AbstractTcrdist):
    name = "Alpha CDR3 tcrdist"
    distance_bins = range(0, 80 + 1, 2)
    _chains_to_compare = [TcrChain.ALPHA]
    _tcrdist_type = TcrdistType.CDR3


class BetaCdr3Tcrdist(AbstractTcrdist):
    name = "Beta CDR3 tcrdist"
    distance_bins = range(0, 80 + 1, 2)
    _chains_to_compare = [TcrChain.BETA]
    _tcrdist_type = TcrdistType.CDR3


class Cdr3Tcrdist(AbstractTcrdist):
    name = "CDR3 tcrdist"
    distance_bins = range(0, 160 + 1, 2)
    _chains_to_compare = [TcrChain.ALPHA, TcrChain.BETA]
    _tcrdist_type = TcrdistType.CDR3


class AlphaTcrdist(AbstractTcrdist):
    name = "Alpha tcrdist"
    distance_bins = range(0, 300 + 1, 5)
    _chains_to_compare = [TcrChain.ALPHA]
    _tcrdist_type = TcrdistType.FULL


class BetaTcrdist(AbstractTcrdist):
    name = "Beta tcrdist"
    distance_bins = range(0, 300 + 1, 5)
    _chains_to_compare = [TcrChain.BETA]
    _tcrdist_type = TcrdistType.FULL


class Tcrdist(AbstractTcrdist):
    name = "tcrdist"
    distance_bins = range(0, 600 + 1, 5)
    _chains_to_compare = [TcrChain.ALPHA, TcrChain.BETA]
    _tcrdist_type = TcrdistType.FULL
