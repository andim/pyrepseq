from abc import abstractmethod
from numpy import ndarray
from pandas import DataFrame, Series
from pyrepseq.tcr_metric.tcr_metric import TcrMetric
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein
from scipy.spatial import distance
from tidytcells import tr
from typing import Iterable, Optional, Tuple


class EditTypeWeights:
    def __init__(
        self, insertion_weight: int, deletion_weight: int, substitution_weight: int
    ) -> None:
        self.insertion_weight = insertion_weight
        self.deletion_weight = deletion_weight
        self.substitution_weight = substitution_weight

    def to_tuple(self) -> Tuple[int]:
        return (self.insertion_weight, self.deletion_weight, self.substitution_weight)


class ChainWeights:
    def __init__(self, alpha_weight: int, beta_weight: int) -> None:
        self.alpha_weight = alpha_weight
        self.beta_weight = beta_weight


class CdrWeights:
    def __init__(self, cdr1_weight: int, cdr2_weight: int, cdr3_weight: int) -> None:
        self.cdr1_weight = cdr1_weight
        self.cdr2_weight = cdr2_weight
        self.cdr3_weight = cdr3_weight


class LevenshteinMetric(TcrMetric):
    _edit_type_weights: EditTypeWeights
    _chain_weights: ChainWeights
    _cdr_weights: CdrWeights

    @property
    @abstractmethod
    def _columns_to_compare(self) -> Iterable[str]:
        pass

    def __init__(
        self,
        insertion_weight: int = 1,
        deletion_weight: int = 1,
        substitution_weight: int = 1,
        alpha_weight: int = 1,
        beta_weight: int = 1,
        cdr1_weight: int = 1,
        cdr2_weight: int = 1,
        cdr3_weight: int = 1,
    ) -> None:
        self._edit_type_weights = EditTypeWeights(
            insertion_weight, deletion_weight, substitution_weight
        )
        self._chain_weights = ChainWeights(alpha_weight, beta_weight)
        self._cdr_weights = CdrWeights(cdr1_weight, cdr2_weight, cdr3_weight)

    def calc_cdist_matrix(
        self, anchor_tcrs: DataFrame, comparison_tcrs: DataFrame
    ) -> ndarray:
        anchor_tcrs = self._expand_v_gene_cdrs(anchor_tcrs)
        comparison_tcrs = self._expand_v_gene_cdrs(comparison_tcrs)

        cdist_matrices = [
            self._calc_cdist_matrix_for_column(anchor_tcrs, comparison_tcrs, column)
            for column in self._columns_to_compare
        ]

        return sum(cdist_matrices)

    def _expand_v_gene_cdrs(self, df: DataFrame) -> DataFrame:
        df = df.copy()
        df[["CDR1A", "CDR2A"]] = self._get_cdrs_from_v_genes(df.TRAV)
        df[["CDR1B", "CDR2B"]] = self._get_cdrs_from_v_genes(df.TRBV)
        return df

    def _get_cdrs_from_v_genes(self, v_genes: Series) -> DataFrame:
        df = DataFrame(columns=["CDR1X", "CDR2X"])
        df.CDR1X = v_genes.map(self._get_cdr1_from_v_gene_if_possible)
        df.CDR2X = v_genes.map(self._get_cdr2_from_v_gene_if_possible)
        return df

    def _get_cdr1_from_v_gene_if_possible(self, v_gene: Optional[str]) -> Optional[str]:
        if not isinstance(v_gene, str):
            return None
        return tr.get_aa_sequence(v_gene)["CDR1-IMGT"]

    def _get_cdr2_from_v_gene_if_possible(self, v_gene: Optional[str]) -> Optional[str]:
        if not isinstance(v_gene, str):
            return None
        return tr.get_aa_sequence(v_gene)["CDR2-IMGT"]

    def _calc_cdist_matrix_for_column(
        self, anchor_tcrs: DataFrame, comparison_tcrs: DataFrame, column: str
    ) -> ndarray:
        anchors = anchor_tcrs[column]
        comparisons = comparison_tcrs[column]
        cdist = process.cdist(anchors, comparisons, scorer=self._levenshtein_scorer)

        if "A" in column:
            cdist *= self._chain_weights.alpha_weight
        elif "B" in column:
            cdist *= self._chain_weights.beta_weight

        if "1" in column:
            cdist *= self._cdr_weights.cdr1_weight
        elif "2" in column:
            cdist *= self._cdr_weights.cdr2_weight
        elif "3" in column:
            cdist *= self._cdr_weights.cdr3_weight

        return cdist

    def _levenshtein_scorer(self, *args, **kwargs) -> int:
        return Levenshtein.distance(
            *args, **kwargs, weights=self._edit_type_weights.to_tuple()
        )

    def calc_pdist_vector(self, tcrs: DataFrame) -> ndarray:
        pdist_matrix = self.calc_cdist_matrix(tcrs, tcrs)
        pdist_vector = distance.squareform(pdist_matrix, checks=False)
        return pdist_vector


class AlphaCdr3Levenshtein(LevenshteinMetric):
    name = "Alpha CDR3 Levenshtein"
    distance_bins = range(25 + 1)
    _columns_to_compare = ["CDR3A"]

    def __init__(
        self,
        insertion_weight: int = 1,
        deletion_weight: int = 1,
        substitution_weight: int = 1,
    ) -> None:
        super().__init__(
            insertion_weight=insertion_weight,
            deletion_weight=deletion_weight,
            substitution_weight=substitution_weight,
        )


class BetaCdr3Levenshtein(LevenshteinMetric):
    name = "Beta CDR3 Levenshtein"
    distance_bins = range(25 + 1)
    _columns_to_compare = ["CDR3B"]

    def __init__(
        self,
        insertion_weight: int = 1,
        deletion_weight: int = 1,
        substitution_weight: int = 1,
    ) -> None:
        super().__init__(
            insertion_weight=insertion_weight,
            deletion_weight=deletion_weight,
            substitution_weight=substitution_weight,
        )


class Cdr3Levenshtein(LevenshteinMetric):
    name = "CDR3 Levenshtein"
    distance_bins = range(50 + 1)
    _columns_to_compare = ["CDR3A", "CDR3B"]

    def __init__(
        self,
        insertion_weight: int = 1,
        deletion_weight: int = 1,
        substitution_weight: int = 1,
        alpha_weight: int = 1,
        beta_weight: int = 1,
    ) -> None:
        super().__init__(
            insertion_weight=insertion_weight,
            deletion_weight=deletion_weight,
            substitution_weight=substitution_weight,
            alpha_weight=alpha_weight,
            beta_weight=beta_weight,
        )


class AlphaCdrLevenshtein(LevenshteinMetric):
    name = "Alpha CDR Levenshtein"
    distance_bins = range(35 + 1)
    _columns_to_compare = ["CDR1A", "CDR2A", "CDR3A"]

    def __init__(
        self,
        insertion_weight: int = 1,
        deletion_weight: int = 1,
        substitution_weight: int = 1,
        cdr1_weight: int = 1,
        cdr2_weight: int = 1,
        cdr3_weight: int = 1,
    ) -> None:
        super().__init__(
            insertion_weight=insertion_weight,
            deletion_weight=deletion_weight,
            substitution_weight=substitution_weight,
            cdr1_weight=cdr1_weight,
            cdr2_weight=cdr2_weight,
            cdr3_weight=cdr3_weight,
        )


class BetaCdrLevenshtein(LevenshteinMetric):
    name = "Beta CDR Levenshtein"
    distance_bins = range(35 + 1)
    _columns_to_compare = ["CDR1B", "CDR2B", "CDR3B"]

    def __init__(
        self,
        insertion_weight: int = 1,
        deletion_weight: int = 1,
        substitution_weight: int = 1,
        cdr1_weight: int = 1,
        cdr2_weight: int = 1,
        cdr3_weight: int = 1,
    ) -> None:
        super().__init__(
            insertion_weight=insertion_weight,
            deletion_weight=deletion_weight,
            substitution_weight=substitution_weight,
            cdr1_weight=cdr1_weight,
            cdr2_weight=cdr2_weight,
            cdr3_weight=cdr3_weight,
        )


class CdrLevenshtein(LevenshteinMetric):
    name = "CDR Levenshtein"
    distance_bins = range(70 + 1)
    _columns_to_compare = ["CDR1A", "CDR2A", "CDR3A", "CDR1B", "CDR2B", "CDR3B"]
