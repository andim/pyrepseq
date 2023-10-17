from abc import abstractmethod
from numpy import ndarray
from pandas import DataFrame, Series
from pyrepseq.tcr_metric.tcr_metric import TcrMetric
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein
from tidytcells import tr
from typing import Iterable, Optional


class LevenshteinMetric(TcrMetric):
    @property
    @abstractmethod
    def _columns_to_compare(self) -> Iterable[str]:
        pass

    def calc_cdist_matrix(
        self, anchor_tcrs: DataFrame, comparison_tcrs: DataFrame
    ) -> ndarray:
        anchor_tcrs = self._expand_v_gene_cdrs(anchor_tcrs)
        comparison_tcrs = self._expand_v_gene_cdrs(comparison_tcrs)

        cdist_matrices = [self._calc_cdist_matrix_for_column(anchor_tcrs, comparison_tcrs, column) for column in self._columns_to_compare]

        return sum(cdist_matrices)

    def _expand_v_gene_cdrs(self, df: DataFrame) -> DataFrame:
        df[["CDR1A", "CDR2A"]] = self._get_cdrs_from_v_genes(df.TRAV)
        df[["CDR1B", "CDR2B"]] = self._get_cdrs_from_v_genes(df.TRBV)
        return df

    def _get_cdrs_from_v_genes(self, v_genes: Series) -> DataFrame:
        df = DataFrame(columns=["CDR1X", "CDR2X"])
        df.CDR1X = v_genes.map(lambda v_gene: self._get_cdr1_from_v_gene_if_possible)
        df.CDR2X = v_genes.map(lambda v_gene: self._get_cdr2_from_v_gene_if_possible)
        return df

    def _get_cdr1_from_v_gene_if_possible(self, v_gene: Optional[str]) -> Optional[str]:
        if not isinstance(v_gene, str):
            return None
        return tr.get_aa_sequence(v_gene)["CDR1-IMGT"]

    def _get_cdr2_from_v_gene_if_possible(self, v_gene: Optional[str]) -> Optional[str]:
        if not isinstance(v_gene, str):
            return None
        return tr.get_aa_sequence(v_gene)["CDR2-IMGT"]

    def _calc_cdist_matrix_for_column(self, anchor_tcrs: DataFrame, comparison_tcrs: DataFrame, column: str) -> ndarray:
        anchors = anchor_tcrs[column]
        comparisons = comparison_tcrs[column]
        return process.cdist(anchors, comparisons, scorer=Levenshtein.distance)


class AlphaCdr3Levenshtein(LevenshteinMetric):
    name = "Alpha CDR3 Levenshtein"
    distance_bins = range(25 + 1)
    _columns_to_compare = ["CDR3A"]


class BetaCdr3Levenshtein(LevenshteinMetric):
    name = "Beta CDR3 Levenshtein"
    distance_bins = range(25 + 1)
    _columns_to_compare = ["CDR3B"]


class Cdr3Levenshtein(LevenshteinMetric):
    name = "CDR3 Levenshtein"
    distance_bins = range(50 + 1)
    _columns_to_compare = ["CDR3A", "CDR3B"]


class AlphaCdrLevenshtein(LevenshteinMetric):
    name = "Alpha CDR Levenshtein"
    distance_bins = range(35 + 1)
    _columns_to_compare = ["CDR1A", "CDR2A", "CDR3A"]


class BetaCdrLevenshtein(LevenshteinMetric):
    name = "Beta CDR Levenshtein"
    distance_bins = range(35 + 1)
    _columns_to_compare = ["CDR1B", "CDR2B", "CDR3B"]


class CdrLevenshtein(LevenshteinMetric):
    name = "CDR Levenshtein"
    distance_bins = range(70 + 1)
    _columns_to_compare = ["CDR1A", "CDR2A", "CDR3A", "CDR1B", "CDR2B", "CDR3B"]
