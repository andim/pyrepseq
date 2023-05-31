from .io import aminoacids
from .stats import pc, pc_joint, pc_conditional
import os.path

import numpy as np
import pandas as pd


def renyi2_entropy(df, features, by=None, base=2.0):
    """Compute Renyi-Simpson entropies.
    """

    if base is not None and base <= 0:
        raise ValueError("`base` must be a positive number or `None`.")

    # Check if multiple features are given or a single feature
    if type(features) == str:
        features = [features]
    
    # Check for single term groupby
    if by is not None and len(by) == 1:
        by = by[0]
    
    if not by:
        entropy = -np.log(pc_joint(df, features))
    else:
        entropy = -np.log(pc_conditional(df, by, features))

    if base is not None:
        entropy /= np.log(base) 
    return entropy


def feature_relevance_dict(df_spc, df_back, features, by=[], spc_group_column=None, *args, **kwargs):
    """ Calculate a dictionary of feature relevance scores.
    """
    
    if not spc_group_column:
        by_spc = by

    else:
        by_spc = by + [spc_group_column]
        
    relevance_dict = {}
    for i, feature in enumerate(features):
        # Compute relevance
        relevance_dict[feature] = (renyi2_entropy(df_back, feature, by) - renyi2_entropy(df_spc, feature, by_spc))
        for feature_2 in np.delete(features,i):
            # Compute conditional relevance
            relevance_dict[feature+"|"+feature_2] =  (renyi2_entropy(df_back, feature, by + [feature_2])
                                                        -  renyi2_entropy(df_spc, feature, by_spc + [feature_2]))
            # Compute joint relevance
            relevance_dict[feature+"+"+feature_2] =  (renyi2_entropy(df_back, [feature, feature_2], by)
                                                        -  renyi2_entropy(df_spc, [feature, feature_2], by_spc))
    
    return relevance_dict
