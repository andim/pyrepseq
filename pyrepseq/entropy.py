import pandas as pd
import numpy as np
from .stats import pc, pc_joint, pc_conditional, stdpc, stdpc_joint, stdpc_conditional


def renyi2_entropy(df, features, by=None, base=2.0, **kwargs):
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
        entropy = -np.log(pc_conditional(df, by, features, **kwargs))
    
    if base is not None:
        entropy /= np.log(base) 
    
    return entropy

def stdrenyi2_entropy(df, features, by=None, base=2.0, **kwargs):
    
    if base is not None and base <= 0:
        raise ValueError("`base` must be a positive number or `None`.")
    
    if type(features) != list:
        features = [features]
            
    if not by:
        stdentropy = stdpc_joint(df, features, **kwargs)/pc_joint(df, features, **kwargs)
        
    else:
        stdentropy = stdpc_conditional(df, by, features, **kwargs)/pc_conditional(df, by, features, **kwargs)
    
    if base is not None:
        stdentropy /= np.log(base) 
    
    return stdentropy