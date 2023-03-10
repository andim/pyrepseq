from functools import reduce

import numpy as np
import pandas as pd
import tidytcells as tt
from typing import Iterable, Mapping, Optional
from warnings import warn

aminoacids = 'ACDEFGHIKLMNPQRSTVWY'
_aminoacids_set = set(aminoacids)

def standardize_dataframe(df_old,
                          col_mapper: Mapping,
                          standardize: bool = True,
                          species = 'HomoSapiens',
                          tcr_enforce_functional = True,
                          tcr_precision = 'gene',
                          mhc_precision = 'gene',
                          suppress_warnings = False):
    '''
    Utility function to organise TCR data into a standardised format.

    If standardization is enabled (True by default), the function will additionally attempt to standardise the TCR and MHC gene symbols to be IMGT-compliant, and CDR3 sequences to be valid.
    The appropriate standardization procedures will be applied for columns with the following names:
        - TRAV / TRBV
        - TRAJ / TRBJ
        - CDR3A / CDR3B
        - MHCA / MHCB
    
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

    suppress_warnings: bool
        If ``True``, suppresses warnings that are emitted when the standardisation of certain values fails.
        Defaults to ``False``.
        
    Returns
    -------
    pandas.DataFrame
        Standardised ``DataFrame`` containing the original data, cleaned.
    '''
    df = df_old[list(col_mapper.keys())]
    df.rename(columns=col_mapper, inplace=True)
    
    # Standardise TCR genes and MHC genes
    if standardize:
        for chain in ('A', 'B'):

            cdr3 = f'CDR3{chain}'
            if cdr3 in df.columns:
                df[cdr3] = df[cdr3].map(
                    lambda x: x if isvalidcdr3(x) else pd.NA
                )

            for gene in ('V', 'J'):
                col = f'TR{chain}{gene}'
                if col in df.columns:
                    df[col] = df[col].map(
                        lambda x: None if pd.isna(x) else tt.tcr.standardise(
                            gene=x,
                            species=species,
                            enforce_functional=tcr_enforce_functional,
                            precision=tcr_precision,
                            suppress_warnings=suppress_warnings
                        )
                    )
            
            mhc = f'MHC{chain}'
            if mhc in df.columns:
                df[mhc] = df[mhc].map(
                    lambda x: None if pd.isna(x) else tt.mhc.standardise(
                        gene=x,
                        species=species,
                        precision=mhc_precision,
                        suppress_warnings=suppress_warnings
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
