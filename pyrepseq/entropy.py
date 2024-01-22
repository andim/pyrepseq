import pandas as pd
import numpy as np
from .stats import pc, pc_joint, pc_conditional, stdpc, stdpc_joint, stdpc_conditional


def shannon_entropy(df, features, by=None, base=2.0):
    """Compute Shannon entropies
    
    Parameters
    ----------
    df : pandas DataFrame
    features: list
    by: string/list of strings
    base: float

    Returns
    ----------: 
    float
    """
    
    if base is not None and base <= 0:
        raise ValueError("`base` must be a positive number or `None`.")
    
    if type(features) != list:
        features = [features]
        
    if by is None:
        probabilities = df[features].value_counts(normalize=True)
        entropy  = -(probabilities*np.log(probabilities)).sum()
    
    else:
        if type(by) == list and len(by) == 1:
            by = by[0]
            
        marginal_probabilities = df[by].value_counts(normalize=True)
        
        if type(by) != list:
            joint_probabilities = df[features+[by]].value_counts(normalize=True)
        else:
            joint_probabilities = df[features+by].value_counts(normalize=True)

        entropy = -(joint_probabilities*np.log(joint_probabilities/marginal_probabilities)).sum()
        
    if base is not None:
        entropy /= np.log(base) 
        
    return entropy 

def renyi2_entropy(df, features, by=None, base=2.0, *args, **kwargs):
    """Compute Renyi-Simpson entropies
    
    Parameters
    ----------
    df : pandas DataFrame
    features: list
    by: string/list of strings
    base: float

    Returns
    ----------: 
    float
    """
    
    if base is not None and base <= 0:
        raise ValueError("`base` must be a positive number or `None`.")
    
    if type(features) != list:
        features = [features]
            
    if not by:
        entropy = -np.log(pc_joint(df, features))
    else:
        entropy = -np.log(pc_conditional(df, by, features, *args, **kwargs))
    
    if base is not None:
        entropy /= np.log(base) 
    
    return entropy

def stdrenyi2_entropy(df, features, by=None, base=2.0, *args, **kwargs):
    
    if base is not None and base <= 0:
        raise ValueError("`base` must be a positive number or `None`.")
    
    if type(features) != list:
        features = [features]
            
    if not by:
        stdentropy = stdpc_joint(df, features)/pc_joint(df, features)
        
    else:
        stdentropy = stdpc_conditional(df, by, features, *args, **kwargs)/pc_conditional(df, by, features, *args, **kwargs)
    
    if base is not None:
        stdentropy /= np.log(base) 
    
    return stdentropy