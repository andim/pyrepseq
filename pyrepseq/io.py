from functools import reduce

import pandas as pd
from pandas import DataFrame
import tidytcells as tt
from typing import Mapping

aminoacids = "ACDEFGHIKLMNPQRSTVWY"
_aminoacids_set = set(aminoacids)


def standardize_dataframe(
    df_old: DataFrame,
    col_mapper: Mapping,
    standardize: bool = True,
    species: str = "HomoSapiens",
    tcr_enforce_functional: bool = True,
    tcr_precision: str = "gene",
    mhc_precision: str = "gene",
    strict_cdr3_standardization: bool = False,
    suppress_warnings: bool = False,
):
    """
    Utility function to organise TCR data into a standardized format.

    If standardization is enabled (True by default), the function will additionally attempt to standardize the TCR and MHC gene symbols to be IMGT-compliant, and CDR3/Epitope amino acid sequences to be valid.
    During standardization, most non-standardizable/nonsensical values will be removed, replaced with `None`.
    However, since epitopes are not necessarily always amino acid sequences, values in the Epitope column that fail standardization will be kept as their original value.
    The appropriate standardization procedures will be applied for columns with the following names:
        - TRAV / TRBV
        - TRAJ / TRBJ
        - CDR3A / CDR3B
        - MHCA / MHCB
        - Epitope

    Parameters
    ----------

    df_old: pandas.DataFrame
        Source ``DataFrame`` from which to pull data.

    col_mapper: Mapping
        A mapping object, such as a dictionary, which maps the old column names to the new column names.

    standardize: bool
        When set to ``False``, gene name standardisation is not attempted.
        Defaults to ``True``.

    species: str
        Name of the species from which the TCR data is derived, in their binomial nomenclature, camel-cased.
        Defaults to ``'HomoSapiens'``.

    tcr_enforce_functional: bool
        When set to ``True``, TCR genes that are not functional (i.e. ORF or pseudogene) are removed, and replaced with ``None``.
        Defaults to ``True``.

    tcr_precision: str
        Level of precision to trim the TCR gene data to (``'gene'`` or ``'allele'``).
        Defaults to ``'gene'``.

    mhc_precision: str
        Level of precision to trim the MHC gene data to (``'gene'``, ``'protein'`` or ``'allele'``).
        Defaults to ``'gene'``.

    strict_cdr3_standardization: bool
        If True, any string that does not look like a CDR3 sequence is rejected.
        If False, any inputs that are valid amino acid sequences but do not start with C and end with F/W are not rejected and instead are corrected by having a C appended to the beginning and an F appended at the end.
        Defaults to False.

    suppress_warnings: bool
        If ``True``, suppresses warnings that are emitted when the standardisation of certain values fails.
        Defaults to ``False``.

    Returns
    -------
    pandas.DataFrame
        Standardized ``DataFrame`` containing the original data, cleaned.
    """
    df = df_old[list(col_mapper.keys())]
    df.rename(columns=col_mapper, inplace=True)

    # Standardize TCR genes and MHC genes
    if standardize:
        for chain in ("A", "B"):
            cdr3 = f"CDR3{chain}"
            if cdr3 in df.columns:
                df[cdr3] = df[cdr3].map(
                    lambda x: None
                    if pd.isna(x)
                    else tt.junction.standardize(
                        seq=x,
                        strict=strict_cdr3_standardization,
                        suppress_warnings=suppress_warnings,
                    )
                )

            for gene in ("V", "J"):
                col = f"TR{chain}{gene}"
                if col in df.columns:
                    df[col] = df[col].map(
                        lambda x: None
                        if pd.isna(x)
                        else tt.tr.standardize(
                            gene=x,
                            species=species,
                            enforce_functional=tcr_enforce_functional,
                            precision=tcr_precision,
                            suppress_warnings=suppress_warnings,
                        )
                    )

            mhc = f"MHC{chain}"
            if mhc in df.columns:
                df[mhc] = df[mhc].map(
                    lambda x: None
                    if pd.isna(x)
                    else tt.mh.standardize(
                        gene=x,
                        species=species,
                        precision=mhc_precision,
                        suppress_warnings=suppress_warnings,
                    )
                )

            if "Epitope" in df.columns:
                df["Epitope"] = df["Epitope"].map(
                    lambda x: None
                    if pd.isna(x)
                    else tt.aa.standardize(
                        seq=x, on_fail="keep", suppress_warnings=suppress_warnings
                    )
                )

    return df


def isvalidaa(string):
    "returns true if string is composed only of characters from the standard amino acid alphabet"
    try:
        return all(c in _aminoacids_set for c in string)
    except TypeError:
        return False


def isvalidcdr3(string):
    """
    returns True if string is a valid CDR3 sequence

    Checks the following:
        - first amino acid is a cysteine (C)
        - last amino acid is either phenylalanine (F), tryptophan (W), or cysteine (C)
        - each amino acid is part of the standard amino acid alphabet

    See http://www.imgt.org/IMGTScientificChart/Numbering/IMGTIGVLsuperfamily.html
    and also https://doi.org/10.1093/nar/gkac190
    """
    try:
        return (
            isvalidaa(string) and (string[0] == "C") and (string[-1] in ["F", "W", "C"])
        )
    # if 'string' is not of string type (e.g. nan) it is not valid
    except TypeError:
        return False


def multimerge(dfs, on, suffixes=None, **kwargs):
    """Merge multiple dataframes on a common column.

    Provides support for custom suffixes.

    Parameters
    ----------
    on: 'index' or column name
    suffixes: [list-like | None]
        list of suffixes to append to the data
    **kwargs:  keyword arguments passed along to `pd.merge`

    Returns
    -------
    merged dataframe
    """

    merge_kwargs = dict(how="outer")
    merge_kwargs.update(kwargs)
    if suffixes:
        dfs_new = []
        for df, suffix in zip(dfs, suffixes):
            if not on == "index":
                df = df.set_index(on)
            dfs_new.append(df.add_suffix("_" + suffix))
        return reduce(
            lambda left, right: pd.merge(
                left, right, right_index=True, left_index=True, **merge_kwargs
            ),
            dfs_new,
        )
    if on == "index":
        return reduce(
            lambda left, right: pd.merge(
                left, right, right_index=True, left_index=True, **merge_kwargs
            ),
            dfs,
        )
    return reduce(lambda left, right: pd.merge(left, right, on, **merge_kwargs), dfs)
