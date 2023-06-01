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
    if type(features) == str or type(features) == np.str_:
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

def relevance(df_spc, df_back, features, by=[], spc_group_column=None):
    """ Calculate relevance score/joint relevance score of provided features
    """
    
    if spc_group_column is None:
        by_spc = by

    else:
        by_spc = by + [spc_group_column]
    
    return (renyi2_entropy(df_back, features, by) - renyi2_entropy(df_spc, features, by_spc))


def feature_relevance_dict(df_spc, df_back, features, by=[], spc_group_column=None):
    """ Calculate a dictionary of feature relevance scores.
    """
        
    relevance_dict = {"features":features}
    for i, feature in enumerate(features):
        # Compute relevance
        relevance_dict[feature] = relevance(df_spc, df_back, feature, by, spc_group_column)
        
        for feature_2 in np.delete(features,i):
            # Compute joint relevance
            relevance_dict[feature+"+"+feature_2] =  relevance(df_spc, df_back, [feature, feature_2], by, spc_group_column)
            
    return relevance_dict