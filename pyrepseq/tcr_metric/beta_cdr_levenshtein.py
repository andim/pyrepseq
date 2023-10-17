from numpy import ndarray
from pandas import DataFrame, Series
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein
from scipy.spatial import distance
from tidytcells import tcr

from pyrepseq.tcr_metric.tcr_metric import TcrMetric


class BetaCdrLevenshtein(TcrMetric):
    name = "Beta CDR Levenshtein"
    distance_bins = range(36)

    def calc_cdist_matrix(
        self, anchor_tcrs: DataFrame, comparison_tcrs: DataFrame
    ) -> ndarray:
        anchor_trbvs = anchor_tcrs.TRBV
        comparison_trbvs = comparison_tcrs.TRBV

        anchor_trbvs = self._add_first_allele_designation_if_none_present(anchor_trbvs)
        comparison_trbvs = self._add_first_allele_designation_if_none_present(
            comparison_trbvs
        )

        anchor_vcdrs = self._get_vcdrs(anchor_trbvs)
        comparison_vcdrs = self._get_vcdrs(comparison_trbvs)

        cdist_cdr1 = process.cdist(
            anchor_vcdrs.CDR1, comparison_vcdrs.CDR1, scorer=Levenshtein.distance
        )
        cdist_cdr2 = process.cdist(
            anchor_vcdrs.CDR2, comparison_vcdrs.CDR2, scorer=Levenshtein.distance
        )
        cdist_cdr3 = process.cdist(
            anchor_tcrs.CDR3B, comparison_tcrs.CDR3B, scorer=Levenshtein.distance
        )

        return cdist_cdr1 + cdist_cdr2 + cdist_cdr3

    def _add_first_allele_designation_if_none_present(self, trbvs: Series) -> Series:
        return trbvs.map(
            lambda x: x + "*01" if isinstance(x, str) and "*" not in x else x
        )

    def _get_vcdrs(self, trbvs: Series) -> DataFrame:
        df = DataFrame()
        df["CDR1"] = trbvs.map(lambda trbv: tcr.get_aa_sequence(trbv)["CDR1-IMGT"])
        df["CDR2"] = trbvs.map(lambda trbv: tcr.get_aa_sequence(trbv)["CDR2-IMGT"])
        return df

    def calc_pdist_vector(self, tcrs: DataFrame) -> ndarray:
        pdist_matrix = self.calc_cdist_matrix(tcrs, tcrs)
        pdist_vector = distance.squareform(pdist_matrix, checks=False)
        return pdist_vector
