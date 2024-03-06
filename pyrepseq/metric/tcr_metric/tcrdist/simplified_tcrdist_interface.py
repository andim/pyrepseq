from numpy import ndarray
from pandas import DataFrame
from tcrdist import rep_funcs
from tcrdist import repertoire_db
import pwseqdist as pw
import warnings
from typing import Dict, Literal


class TcrdistInterface:
    _all_genes = repertoire_db.RefGeneSet("alphabeta_gammadelta_db.tsv").all_genes

    def calc_alpha_cdist_matrices(
        self, anchor_tcrs: DataFrame, comparison_tcrs: DataFrame
    ) -> Dict[str, ndarray]:
        return self._calc_cdist_matrices(anchor_tcrs, comparison_tcrs, chain="alpha")

    def calc_beta_cdist_matrices(
        self, anchor_tcrs: DataFrame, comparison_tcrs: DataFrame
    ) -> Dict[str, ndarray]:
        return self._calc_cdist_matrices(anchor_tcrs, comparison_tcrs, chain="beta")

    def _calc_cdist_matrices(
        self,
        anchor_tcrs: DataFrame,
        comparison_tcrs: DataFrame,
        chain: Literal["alpha", "beta"],
    ) -> Dict[str, ndarray]:
        anchor_tcrs = self._convert_df_to_tcrdist_form(anchor_tcrs)
        comparison_tcrs = self._convert_df_to_tcrdist_form(comparison_tcrs)

        anchor_tcrs = self._infer_cdrs_from_v_gene(anchor_tcrs, chain)
        comparison_tcrs = self._infer_cdrs_from_v_gene(comparison_tcrs, chain)

        pws_kwargs = self._get_pws_kwargs(chain)

        return rep_funcs._pws(
            df=anchor_tcrs, df2=comparison_tcrs, store=True, **pws_kwargs
        )

    def _convert_df_to_tcrdist_form(self, df: DataFrame) -> DataFrame:
        df = df.rename(
            columns={
                "TRAV": "v_a_gene",
                "CDR3A": "cdr3_a_aa",
                "TRBV": "v_b_gene",
                "CDR3B": "cdr3_b_aa",
                "duplicate_count": "count",
            }
        )

        for column in ["v_a_gene", "v_b_gene"]:
            if column in df:
                df[column] = df[column].map(
                    lambda x: x if not type(x) == str or "*" in x else x + "*01"
                )

        if not "count" in df:
            df["count"] = 1

        return df

    def _infer_cdrs_from_v_gene(
        self, cell_df, chain, organism="human", imgt_aligned=True
    ):
        """
        Taken and modified from tcrdist.repertoire.TCRrep
        """

        if not imgt_aligned:
            f0 = lambda v: self._map_gene_to_reference_seq2(
                gene=v, cdr=0, organism=organism, attr="cdrs_no_gaps"
            )
            f1 = lambda v: self._map_gene_to_reference_seq2(
                gene=v, cdr=1, organism=organism, attr="cdrs_no_gaps"
            )
            f2 = lambda v: self._map_gene_to_reference_seq2(
                gene=v, cdr=2, organism=organism, attr="cdrs_no_gaps"
            )
        else:
            imgt_aligned_status = True
            f0 = lambda v: self._map_gene_to_reference_seq2(
                gene=v, cdr=0, organism=organism, attr="cdrs"
            )
            f1 = lambda v: self._map_gene_to_reference_seq2(
                gene=v, cdr=1, organism=organism, attr="cdrs"
            )
            f2 = lambda v: self._map_gene_to_reference_seq2(
                gene=v, cdr=2, organism=organism, attr="cdrs"
            )

        if chain == "alpha":
            cell_df = cell_df.assign(
                cdr1_a_aa=list(map(f0, cell_df.v_a_gene)),
                cdr2_a_aa=list(map(f1, cell_df.v_a_gene)),
                pmhc_a_aa=list(map(f2, cell_df.v_a_gene)),
            )
        if chain == "beta":
            cell_df = cell_df.assign(
                cdr1_b_aa=list(map(f0, cell_df.v_b_gene)),
                cdr2_b_aa=list(map(f1, cell_df.v_b_gene)),
                pmhc_b_aa=list(map(f2, cell_df.v_b_gene)),
            )
        if chain == "gamma":
            cell_df = cell_df.assign(
                cdr1_g_aa=list(map(f0, cell_df.v_g_gene)),
                cdr2_g_aa=list(map(f1, cell_df.v_g_gene)),
                pmhc_g_aa=list(map(f2, cell_df.v_g_gene)),
            )
        if chain == "delta":
            cell_df = cell_df.assign(
                cdr1_d_aa=list(map(f0, cell_df.v_d_gene)),
                cdr2_d_aa=list(map(f1, cell_df.v_d_gene)),
                pmhc_d_aa=list(map(f2, cell_df.v_d_gene)),
            )

        return cell_df

    def _map_gene_to_reference_seq2(self, organism, gene, cdr, attr="cdrs_no_gaps"):
        """
        Taken and modified from tcrdist.repertoire.TCRrep
        """

        try:
            aa_string = self._all_genes[organism][gene].__dict__[attr][cdr]
        except KeyError:
            aa_string = None
            warnings.warn(
                "{} gene was not recognized in reference db no cdr seq could be inferred".format(
                    gene
                ),
                stacklevel=2,
            )
        except IndexError:
            aa_string = None
            warnings.warn(
                "{} gene was not found in index, no cdr seq could be inferred".format(
                    gene
                ),
                stacklevel=2,
            )
        return aa_string

    def _get_pws_kwargs(self, chain: Literal["alpha", "beta"]) -> dict:
        if chain == "alpha":
            chain_code = "a"
        else:
            chain_code = "b"

        metrics = {
            f"cdr3_{chain_code}_aa": pw.metrics.nb_vector_tcrdist,
            f"pmhc_{chain_code}_aa": pw.metrics.nb_vector_tcrdist,
            f"cdr2_{chain_code}_aa": pw.metrics.nb_vector_tcrdist,
            f"cdr1_{chain_code}_aa": pw.metrics.nb_vector_tcrdist,
        }
        weights = {
            f"cdr3_{chain_code}_aa": 3,
            f"pmhc_{chain_code}_aa": 1,
            f"cdr2_{chain_code}_aa": 1,
            f"cdr1_{chain_code}_aa": 1,
        }
        kargs = {
            f"cdr3_{chain_code}_aa": {
                "use_numba": True,
                "distance_matrix": pw.matrices.tcr_nb_distance_matrix,
                "dist_weight": 1,
                "gap_penalty": 4,
                "ntrim": 3,
                "ctrim": 2,
                "fixed_gappos": False,
            },
            f"pmhc_{chain_code}_aa": {
                "use_numba": True,
                "distance_matrix": pw.matrices.tcr_nb_distance_matrix,
                "dist_weight": 1,
                "gap_penalty": 4,
                "ntrim": 0,
                "ctrim": 0,
                "fixed_gappos": True,
            },
            f"cdr2_{chain_code}_aa": {
                "use_numba": True,
                "distance_matrix": pw.matrices.tcr_nb_distance_matrix,
                "dist_weight": 1,
                "gap_penalty": 4,
                "ntrim": 0,
                "ctrim": 0,
                "fixed_gappos": True,
            },
            f"cdr1_{chain_code}_aa": {
                "use_numba": True,
                "distance_matrix": pw.matrices.tcr_nb_distance_matrix,
                "dist_weight": 1,
                "gap_penalty": 4,
                "ntrim": 0,
                "ctrim": 0,
                "fixed_gappos": True,
            },
        }
        return {"metrics": metrics, "weights": weights, "kargs": kargs}
