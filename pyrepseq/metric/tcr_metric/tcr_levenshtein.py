__all__ = [
    "AlphaCdr3Levenshtein",
    "BetaCdr3Levenshtein",
    "Cdr3Levenshtein",
    "AlphaCdrLevenshtein",
    "BetaCdrLevenshtein",
    "CdrLevenshtein"
]

from abc import abstractmethod
from enum import Enum
import itertools
from numpy import ndarray
from pandas import DataFrame, Series
from pyrepseq.metric.tcr_metric import TcrMetric
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein
from scipy.spatial import distance
from tidytcells import tr
from typing import Literal, Tuple


class ChainScope(Enum):
    PAIRED = 1
    ALPHA = 2
    BETA = 3


class CdrScope(Enum):
    ALL = 1
    CDR3 = 2


class ChainWeights:
    def __init__(self, alpha_weight: int, beta_weight: int) -> None:
        self.alpha_weight = alpha_weight
        self.beta_weight = beta_weight


class CdrWeights:
    def __init__(self, cdr1_weight: int, cdr2_weight: int, cdr3_weight: int) -> None:
        self.cdr1_weight = cdr1_weight
        self.cdr2_weight = cdr2_weight
        self.cdr3_weight = cdr3_weight


class TcrLevenshtein(TcrMetric):
    _chain_weights: ChainWeights
    _cdr_weights: CdrWeights

    @property
    @abstractmethod
    def _chain_scope(self) -> ChainScope:
        pass

    @property
    @abstractmethod
    def _cdr_scope(self) -> CdrScope:
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
        if insertion_weight == 1 and deletion_weight == 1 and substitution_weight == 1:
            self._scorer = Levenshtein.distance
        else:
            self._scorer = lambda *args, **kwargs: Levenshtein.distance(
                *args, **kwargs, weights=(insertion_weight, deletion_weight, substitution_weight)
            )

        self._chain_weights = ChainWeights(alpha_weight, beta_weight)
        self._cdr_weights = CdrWeights(cdr1_weight, cdr2_weight, cdr3_weight)

    def calc_cdist_matrix(
        self, anchors: DataFrame, comparisons: DataFrame
    ) -> ndarray:
        super().calc_cdist_matrix(anchors, comparisons)

        if self._cdr_scope is CdrScope.ALL:
            anchors = self._expand_v_gene_cdrs(anchors)
            comparisons = self._expand_v_gene_cdrs(comparisons)

        cdist_matrices = [
            self._calc_cdist_matrix_for_column(anchors, comparisons, column)
            for column in self._get_columns_to_compare()
        ]

        return sum(cdist_matrices)

    def _expand_v_gene_cdrs(self, df: DataFrame) -> DataFrame:
        df = df.copy()
        df[["CDR1A", "CDR2A"]] = self._get_cdrs_from_v_genes(df.TRAV)
        df[["CDR1B", "CDR2B"]] = self._get_cdrs_from_v_genes(df.TRBV)
        return df

    def _get_cdrs_from_v_genes(self, v_genes: Series) -> DataFrame:
        df = DataFrame(columns=["CDR1X", "CDR2X"])
        df.CDR1X = v_genes.map(lambda v: self._get_cdr1_from_v_gene_if_possible(v, "CDR1-IMGT"))
        df.CDR2X = v_genes.map(lambda v: self._get_cdr1_from_v_gene_if_possible(v, "CDR2-IMGT"))
        return df

    @staticmethod
    def _get_cdr1_from_v_gene_if_possible(v_gene: str, cdr_loop: Literal["CDR1-IMGT", "CDR2-IMGT"]) -> str:
        v_gene_seq_data = tr.get_aa_sequence(v_gene)

        if cdr_loop not in v_gene_seq_data:
            return ""
        
        return v_gene_seq_data[cdr_loop]

    def _get_columns_to_compare(self) -> Tuple[str]:
        cdr_prefixes = ["CDR3"]
        if self._cdr_scope is CdrScope.ALL:
            cdr_prefixes.extend(["CDR1", "CDR2"])
        
        chain_suffixes = []
        if self._chain_scope in (ChainScope.PAIRED, ChainScope.ALPHA):
            chain_suffixes.append("A")
        if self._chain_scope in (ChainScope.PAIRED, ChainScope.BETA):
            chain_suffixes.append("B")

        columns_to_compare = [prefix + suffix for prefix, suffix in itertools.product(cdr_prefixes, chain_suffixes)]

        return columns_to_compare
            
    def _calc_cdist_matrix_for_column(
        self, anchors: DataFrame, comparisons: DataFrame, column: str
    ) -> ndarray:
        anchors = anchors[column]
        comparisons = comparisons[column]
        cdist = process.cdist(anchors, comparisons, scorer=self._scorer, workers=-1)

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

    def calc_pdist_vector(self, instances: DataFrame) -> ndarray:
        super().calc_pdist_vector(instances)
        pdist_matrix = self.calc_cdist_matrix(instances, instances)
        pdist_vector = distance.squareform(pdist_matrix, checks=False)
        return pdist_vector


class AlphaCdr3Levenshtein(TcrLevenshtein):
    """
    A TcrMetric that measures the Levenshtein distance between the alpha chain CDR3 sequences.

    Parameters
    ----------
    insertion_weight: int
        An integer multiplier for insertions
        Defaults to 1.
        
    deletion_weight: int
        An integer multiplier for deletions
        Defaults to 1.

    substitution_weight: int
        An integer multiplier for substitutions
        Defaults to 1.
    """

    name = "Alpha CDR3 Levenshtein"
    distance_bins = range(25 + 1)
    _chain_scope = ChainScope.ALPHA
    _cdr_scope = CdrScope.CDR3

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


class BetaCdr3Levenshtein(TcrLevenshtein):
    """
    A TcrMetric that measures the Levenshtein distance between the beta chain CDR3 sequences.

    Parameters
    ----------
    insertion_weight: int
        An integer multiplier for insertions
        Defaults to 1.
        
    deletion_weight: int
        An integer multiplier for deletions
        Defaults to 1.

    substitution_weight: int
        An integer multiplier for substitutions
        Defaults to 1.
    """

    name = "Beta CDR3 Levenshtein"
    distance_bins = range(25 + 1)
    _chain_scope = ChainScope.BETA
    _cdr_scope = CdrScope.CDR3

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


class Cdr3Levenshtein(TcrLevenshtein):
    """
    A TcrMetric that measures the Levenshtein distance between the alpha and beta chain CDR3 sequences.

    Parameters
    ----------
    insertion_weight: int
        An integer multiplier for insertions
        Defaults to 1.
        
    deletion_weight: int
        An integer multiplier for deletions
        Defaults to 1.

    substitution_weight: int
        An integer multiplier for substitutions
        Defaults to 1.

    alpha_weight: int
        An integer multiplier for edits on the alpha chain.
        Defaults to 1.

    beta_weight: int
        An integer multiplier for edits on the beta chain.
        Defaults to 1.
    """

    name = "CDR3 Levenshtein"
    distance_bins = range(50 + 1)
    _chain_scope = ChainScope.PAIRED
    _cdr_scope = CdrScope.CDR3

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


class AlphaCdrLevenshtein(TcrLevenshtein):
    """
    A TcrMetric that measures the Levenshtein distance between the alpha chain CDR1, CDR2, and CDR3 sequences.

    Parameters
    ----------
    insertion_weight: int
        An integer multiplier for insertions
        Defaults to 1.
        
    deletion_weight: int
        An integer multiplier for deletions
        Defaults to 1.

    substitution_weight: int
        An integer multiplier for substitutions
        Defaults to 1.

    cdr1_weight: int
        An integer multiplier for edits on the CDR1.
        Defaults to 1.

    cdr2_weight: int
        An integer multiplier for edits on the CDR2.
        Defaults to 1.

    cdr3_weight: int
        An integer multiplier for edits on the CDR3.
        Defaults to 1.
    """

    name = "Alpha CDR Levenshtein"
    distance_bins = range(35 + 1)
    _chain_scope = ChainScope.ALPHA
    _cdr_scope = CdrScope.ALL

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


class BetaCdrLevenshtein(TcrLevenshtein):
    """
    A TcrMetric that measures the Levenshtein distance between the beta chain CDR1, CDR2, and CDR3 sequences.

    Parameters
    ----------
    insertion_weight: int
        An integer multiplier for insertions
        Defaults to 1.
        
    deletion_weight: int
        An integer multiplier for deletions
        Defaults to 1.

    substitution_weight: int
        An integer multiplier for substitutions
        Defaults to 1.

    cdr1_weight: int
        An integer multiplier for edits on the CDR1.
        Defaults to 1.

    cdr2_weight: int
        An integer multiplier for edits on the CDR2.
        Defaults to 1.

    cdr3_weight: int
        An integer multiplier for edits on the CDR3.
        Defaults to 1.
    """

    name = "Beta CDR Levenshtein"
    distance_bins = range(35 + 1)
    _chain_scope = ChainScope.BETA
    _cdr_scope = CdrScope.ALL

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


class CdrLevenshtein(TcrLevenshtein):
    """
    A TcrMetric that measures the Levenshtein distance between the alpha and beta chain CDR1, CDR2, and CDR3 sequences.

    Parameters
    ----------
    insertion_weight: int
        An integer multiplier for insertions
        Defaults to 1.
        
    deletion_weight: int
        An integer multiplier for deletions
        Defaults to 1.

    substitution_weight: int
        An integer multiplier for substitutions
        Defaults to 1.

    cdr1_weight: int
        An integer multiplier for edits on the CDR1.
        Defaults to 1.

    cdr2_weight: int
        An integer multiplier for edits on the CDR2.
        Defaults to 1.

    cdr3_weight: int
        An integer multiplier for edits on the CDR3.
        Defaults to 1.

    alpha_weight: int
        An integer multiplier for edits on the alpha chain.
        Defaults to 1.

    beta_weight: int
        An integer multiplier for edits on the beta chain.
        Defaults to 1.
    """

    name = "CDR Levenshtein"
    distance_bins = range(70 + 1)
    _chain_scope = ChainScope.PAIRED
    _cdr_scope = CdrScope.ALL
