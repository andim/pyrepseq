from .io import aminoacids
from .stats import pc, pc_joint, pc_conditional
import os.path

import numpy as np
import pandas as pd


#Compute entropy of a feature from coincidence probability
def renyi2_entropy(df, features, by = [], log_fun = np.log2):

    "Estimator for Renyi entropy of order 2"
    
    #Check if multiple features are given or a single feature
    if type(features) == str:
        
        features = [features]
    
    #Check for single term groupby
    if len(by) == 1:
        
        by = by[0]
    
    #Check to see if conditioning is required
    if not by:
        
        entropy = -log_fun(pc_joint(df, features))
        
    else:
        
        entropy = -log_fun(pc_conditional(df, by, features))
        
    
    return entropy


def feature_relevance_dict(df_spc, df_back, features, by = [], epitope_column="Epitope", *args, **kwargs):
    
    if not epitope_column:
        by_spc = by

    else:
        by_spc = by + [epitope_column]
        
    relevance_dict = {}
    #Iterate through the features
    for i, feature in enumerate(features):
        
        #Compute lone relevance
        relevance_dict[feature] = (renyi2_entropy(df_back, feature, by) - renyi2_entropy(df_spc, feature, by_spc))
        
           
        ##Compute conditional and joint relevance
        for feature_2 in np.delete(features,i):
            relevance_dict[feature+"|"+feature_2] =  (renyi2_entropy(df_back, feature, by + [feature_2])
                                                        -  renyi2_entropy(df_spc, feature, by_spc + [feature_2]))
      
      
            relevance_dict[feature+"+"+feature_2] =  (renyi2_entropy(df_back, [feature, feature_2], by)
                                                        -  renyi2_entropy(df_spc, [feature, feature_2], by_spc))
    
    return relevance_dict