from functools import reduce

import numpy as np
import pandas as pd
import tidytcells as tt
from warnings import warn

aminoacids = 'ACDEFGHIKLMNPQRSTVWY'
_aminoacids_set = set(aminoacids)

def standardize_dataframe(df_old, from_columns,
                          to_columns = ["TRAV", "CDR3A","TRAJ",
                                        "TRBV", "CDR3B", "TRBJ",
                                        "Epitope", "MHCA", "MHCB",
                                        "clonal_counts"],
                          species = 'HomoSapiens',
                          tcr_enforce_functional = True,
                          tcr_precision = 'gene',
                          mhc_precision = 'gene'):
    '''
    Utility function to organise TCR data into a standardised format.

    Parameters
    ----------

    df_old: pandas.DataFrame
        Source ``DataFrame`` from which to pull data.
        
    from_columns: Iterable
        Iterable of old table column names to be mapped to the standardised columns, in their respective order.
        
    to_columns: Iterable
        List of columns to map the old ``from_columns`` to.
        Defaults to ``["TRAV", "CDR3A","TRAJ", "TRBV", "CDR3B", "TRBJ", "Epitope", "MHCA", "MHCB", "clonal_counts"]``.
        
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
        
    Returns
    -------
    pandas.DataFrame
        Standardised ``DataFrame`` containing the original data, cleaned.
    '''


    df = pd.DataFrame()
    for from_column, to_column in zip(from_columns, to_columns):
        try:
            df[to_column] = df_old[from_column]
        except:
            df[to_column] = np.full(len(df_old), np.nan)

    if df["CDR3A"].isnull().all():
        df = df[df['CDR3B'].apply(isvalidcdr3)]

    elif df["CDR3B"].isnull().all():
        df = df[df['CDR3A'].apply(isvalidcdr3)]

    else:
        df = df[df['CDR3A'].apply(isvalidcdr3)]
        df = df[df['CDR3B'].apply(isvalidcdr3)]
    

    # Standardise TCR genes and MHC genes
    for col in ('TRAV', 'TRAJ', 'TRBV', 'TRBJ'):
        if not col in to_columns:
            warn(f'No column identified for {col}. Skipping gene standardisation...')
            continue
        
        df[col] = df[col].map(
            lambda x: None if pd.isna(x) else tt.tcr.standardise(
                gene_name=x,
                species=species,
                enforce_functional=tcr_enforce_functional,
                precision=tcr_precision
            )
        )
    
    for col in ('MHCA', 'MHCB'):
        if not col in to_columns:
            warn(f'No column identified for {col}. Skipping gene standardisation...')
            continue
        
        df[col] = df[col].map(
            lambda x: None if pd.isna(x) else tt.mhc.standardise(
                gene_name=x,
                species=species,
                precision=mhc_precision
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
        return (isvalidaa(string)
            and (string[0] == 'C')
            and (string[-1] in ['F', 'W', 'C']))
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

    merge_kwargs = dict(how='outer')
    merge_kwargs.update(kwargs)
    if suffixes:
        dfs_new = []
        for df, suffix in zip(dfs, suffixes):
            if not on == 'index':
                df = df.set_index(on)
            dfs_new.append(df.add_suffix('_'+suffix))
        return reduce(lambda left, right: pd.merge(left, right,
                                                   right_index=True, left_index=True,
                                                   **merge_kwargs),
                      dfs_new)
    if on == 'index':
        return reduce(lambda left, right: pd.merge(left, right,
                                                   right_index=True, left_index=True,
                                                   **merge_kwargs),
                      dfs)
    return reduce(lambda left, right: pd.merge(left, right, on,
                                               **merge_kwargs), dfs)
