from functools import reduce
import pandas as pd
from pandas import DataFrame
import tidytcells as tt
from typing import Mapping
from warnings import warn

aminoacids = "ACDEFGHIKLMNPQRSTVWY"
_aminoacids_set = set(aminoacids)


def standardize_dataframe(
    df: DataFrame = None,
    col_mapper: Mapping = None,
    standardize: bool = True,
    species: str = "HomoSapiens",
    tcr_enforce_functional: bool = True,
    tcr_precision: str = "gene",
    mhc_precision: str = "gene",
    strict_cdr3_standardization: bool = False,
    suppress_warnings: bool = False,
    df_old: DataFrame = None
):
    """
    This is a utility function to organise a table of TCR-pMHC data into the standard pyrepseq format and perform data cleaning/standardization to ensure that the TCR/MHC gene symbols are IMGT-compliant, the epitopes are all valid amino acid strings, and the CDR3s look valid.
    For further notes on data standardization, see below.
    The standard format is a table with some or all of the following columns (not necessarily in order):

    +-----------------+------------------------------------------+-----------+
    | Column Name     | Column should contain                    | Data type |
    +=================+==========================================+===========+
    | TRAV            | TRAV gene symbol                         | `str`     |
    +-----------------+------------------------------------------+-----------+
    | CDR3A           | TCR alpha chain CDR3 amino acid sequence | `str`     |
    +-----------------+------------------------------------------+-----------+
    | TRAJ            | TRAJ gene symbol                         | `str`     |
    +-----------------+------------------------------------------+-----------+
    | TRBV            | TRBV gene symbol                         | `str`     |
    +-----------------+------------------------------------------+-----------+
    | CDR3B           | TCR beta chain CDR3 amino acid sequence  | `str`     |
    +-----------------+------------------------------------------+-----------+
    | TRBJ            | TRBJ gene symbol                         | `str`     |
    +-----------------+------------------------------------------+-----------+
    | Epitope         | Epitope amino acid sequence              | `str`     |
    +-----------------+------------------------------------------+-----------+
    | MHCA            | MHC alpha chain gene symbol              | `str`     |
    +-----------------+------------------------------------------+-----------+
    | MHCB            | MHC beta chain gene symbol               | `str`     |
    +-----------------+------------------------------------------+-----------+

    If the input DataFrame contains the necessary data in columns that are named differently, this can be resolved by providing the mapping to the col_mapper argument (see parameters and examples).

    If standardization is enabled (True by default), the function will additionally attempt to standardize the TCR and MHC gene symbols to be IMGT-compliant, and CDR3/Epitope amino acid sequences to be valid.
    However, for the standardization to happen, the columns with the relevant data must either be correctly named, or the necessary re-naming scheme must be specified by supplying an argument to the `col_mapper` parameter.
    During standardization, most non-standardizable/nonsensical values will be removed, replaced with `None`.
    However, since epitopes are not necessarily always amino acid sequences, values in the Epitope column that fail standardization will be kept as their original value.

    .. deprecated:: 1.4
        `df_old` will be removed in pyrepseq 2.0, with the more simply named `df` parameter.

    Parameters
    ----------

    df: pandas.DataFrame
        Source ``DataFrame`` from which to pull data.

    df_old: pandas.DataFrame
        Alias for ``df``.
        Now deprecated and will be removed in version 2.0.

    col_mapper: Mapping
        A mapping object, such as a dictionary, which maps the old column names to the new column names.
        This should not be set if no column re-naming is necessary.
        Defaults to ``None``.

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

    Examples
    --------
    If you already have a DataFrame in the standard format, `standardize_dataframe` can perform data standardization for you.
    In the examples shown here, we omit any standardization warnings for ease of reading.

    Say you have the following DataFrame:

    >>> from pyrepseq import io
    >>> import pandas as pd
    >>> df = pd.DataFrame(
    ...     data=[
    ...         ["av26.1*1",  "CIVRAPGRADMRF", "aj43*1",    "bv13*1",      "CASSYLPGQGDHYSNQPQHF","bj1.5*1",    "FLKEKGGL",       "b8",         "b2m"],
    ...         ["TCRAV20*01","CAVPSGAGSYQLTF","TCRAJ28*01","TCRBV28S1*01","CASSLGQSGANVLTF",     "TCRBJ2S6*01","LQPFPQPELPYPQPQ","HLA-DQA1*05","HLA-DQB1*02"],
    ...         ["unknown",   "unknown",       "unknown",   "TRBV7-2*01",  "CASSDWGSQNTLYF",      "TRBJ2-4*01", "YMPYFFTLL",      "HLA-A*02",   "B2M"]
    ...     ],
    ...     columns=["TRAV","CDR3A","TRAJ","TRBV","CDR3B","TRBJ","Epitope","MHCA","MHCB"]
    ... )
    >>> df
             TRAV           CDR3A        TRAJ          TRBV                 CDR3B         TRBJ          Epitope         MHCA         MHCB  
    0    av26.1*1   CIVRAPGRADMRF      aj43*1        bv13*1  CASSYLPGQGDHYSNQPQHF      bj1.5*1         FLKEKGGL           b8          b2m  
    1  TCRAV20*01  CAVPSGAGSYQLTF  TCRAJ28*01  TCRBV28S1*01       CASSLGQSGANVLTF  TCRBJ2S6*01  LQPFPQPELPYPQPQ  HLA-DQA1*05  HLA-DQB1*02  
    2     unknown         unknown     unknown    TRBV7-2*01        CASSDWGSQNTLYF   TRBJ2-4*01        YMPYFFTLL     HLA-A*02          B2M

    By passing this to `standardize_dataframe, you will get a cleaned version of the data.

    >>> io.standardize_dataframe(df, suppress_warnings=True)
           TRAV           CDR3A    TRAJ     TRBV                 CDR3B     TRBJ          Epitope      MHCA      MHCB  
    0  TRAV26-1   CIVRAPGRADMRF  TRAJ43   TRBV13  CASSYLPGQGDHYSNQPQHF  TRBJ1-5         FLKEKGGL     HLA-B       B2M  
    1    TRAV20  CAVPSGAGSYQLTF  TRAJ28   TRBV28       CASSLGQSGANVLTF  TRBJ2-6  LQPFPQPELPYPQPQ  HLA-DQA1  HLA-DQB1  
    2      None            None    None  TRBV7-2        CASSDWGSQNTLYF  TRBJ2-4        YMPYFFTLL     HLA-A       B2M

    If you want to have extra columns on the DataFrame, that is allowed.

    >>> extended_df = df.copy()
    >>> extended_df["clone_count"] = [1,2,3]
    >>> io.standardize_dataframe(extended_df, suppress_warnings=True)
           TRAV           CDR3A    TRAJ     TRBV                 CDR3B     TRBJ          Epitope      MHCA      MHCB  clone_count  
    0  TRAV26-1   CIVRAPGRADMRF  TRAJ43   TRBV13  CASSYLPGQGDHYSNQPQHF  TRBJ1-5         FLKEKGGL     HLA-B       B2M            1
    1    TRAV20  CAVPSGAGSYQLTF  TRAJ28   TRBV28       CASSLGQSGANVLTF  TRBJ2-6  LQPFPQPELPYPQPQ  HLA-DQA1  HLA-DQB1            2
    2      None            None    None  TRBV7-2        CASSDWGSQNTLYF  TRBJ2-4        YMPYFFTLL     HLA-A       B2M            3

    Having only a subset of the standard columns is also allowed.

    >>> beta_only_df = df.copy()
    >>> beta_only_df = beta_only_df[["TRBV","CDR3B","TRBJ"]]
    >>> io.standardize_dataframe(beta_only_df, suppress_warnings=True)
          TRBV                 CDR3B     TRBJ
    0   TRBV13  CASSYLPGQGDHYSNQPQHF  TRBJ1-5
    1   TRBV28       CASSLGQSGANVLTF  TRBJ2-6
    2  TRBV7-2        CASSDWGSQNTLYF  TRBJ2-4

    Columns can be renamed by suppling a mapping to the `col_mapper` parameter.

    >>> beta_only_misnamed = beta_only_df.copy()
    >>> beta_only_misnamed.columns = ["foo", "bar", "baz"]
    >>> beta_only_misnamed
                foo                   bar          baz
    0        bv13*1  CASSYLPGQGDHYSNQPQHF      bj1.5*1
    1  TCRBV28S1*01       CASSLGQSGANVLTF  TCRBJ2S6*01
    2    TRBV7-2*01        CASSDWGSQNTLYF   TRBJ2-4*01
    >>> col_mapper = {
    ...     "foo": "TRBV",
    ...     "bar": "CDR3B",
    ...     "baz": "TRBJ"
    ... }
    >>> io.standardize_dataframe(beta_only_misnamed, col_mapper=col_mapper)
          TRBV                 CDR3B     TRBJ
    0   TRBV13  CASSYLPGQGDHYSNQPQHF  TRBJ1-5
    1   TRBV28       CASSLGQSGANVLTF  TRBJ2-6
    2  TRBV7-2        CASSDWGSQNTLYF  TRBJ2-4
    """
    if df_old is not None:
        if df is not None:
            raise ValueError("`df` and `df_old` are mutually exclusive.")
        warn("The parameter df_old is now deprecated in favour of df. This will be removed in version 2.0.")
        df = df_old
    
    if df is None:
        raise ValueError("Missing argument for parameter `df`.")
    
    df_standardized = df.copy()

    if col_mapper is not None:
        df_standardized = df_standardized.rename(columns=col_mapper)

    # Standardize TCR genes and MHC genes
    if standardize:
        for chain in ("A", "B"):
            cdr3 = f"CDR3{chain}"
            if cdr3 in df_standardized.columns:
                df_standardized[cdr3] = df_standardized[cdr3].map(
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
                if col in df_standardized.columns:
                    df_standardized[col] = df_standardized[col].map(
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
            if mhc in df_standardized.columns:
                df_standardized[mhc] = df_standardized[mhc].map(
                    lambda x: None
                    if pd.isna(x)
                    else tt.mh.standardize(
                        gene=x,
                        species=species,
                        precision=mhc_precision,
                        suppress_warnings=suppress_warnings,
                    )
                )

        if "Epitope" in df_standardized.columns:
            df_standardized["Epitope"] = df_standardized["Epitope"].map(
                lambda x: None
                if pd.isna(x)
                else tt.aa.standardize(
                    seq=x, on_fail="keep", suppress_warnings=suppress_warnings
                )
            )

    return df_standardized


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
