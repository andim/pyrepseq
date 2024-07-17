import pandas as pd
import numpy as np
from .stats import pc, pc_joint, pc_conditional, stdpc, stdpc_joint


def renyi2_entropy(df, features, by=None, base=2.0, **kwargs):
    """Compute Renyi-Simpson entropies as the negative-log of estimated coincidence probabilities.
    
    Parameters
    ----------
    df : pandas DataFrame
    features: list or string
        if list, features are concatenated to calculate joint entropy
    by: string/list of strings
        to compute conditional entropy
    base: float (default: 2)

    Returns
    ----------: 
    float
    """
        
    if base is not None and base <= 0:
        raise ValueError("`base` must be a positive number or `None`.")
        
    if not by:
        if type(features) != list:
            entropy = -np.log(pc(df[features]))
            
        else:
            entropy = -np.log(pc_joint(df, features))
        
    else:
        entropy = -np.log(pc_conditional(df, by, features, **kwargs))
    
    if base is not None:
        entropy /= np.log(base) 
    
    return entropy

def stdrenyi2_entropy(df, features, base=2.0, **kwargs):
    """Compute standard deviation of Renyi-Simpson entropies.

    Uses linear error propagation from coincidence probability variances.
    
    Parameters
    ----------
    df : pandas DataFrame
    features: list or string
        if list, features are concatenated to calculate joined entropy
    base: float (default: 2)

    Returns
    ----------: 
    float
    """
    
    if base is not None and base <= 0:
        raise ValueError("`base` must be a positive number or `None`.")
          
    if type(features) != list:
        stdentropy = stdpc(df[features])/pc(df[features])
        
    else:
        stdentropy = stdpc_joint(df, features, **kwargs)/pc_joint(df, features)
    
    
    if base is not None:
        stdentropy /= np.log(base) 
    
    return stdentropy
