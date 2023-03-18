import numpy as np

def levenshtein_dist(seq, seq1=None):
    from rapidfuzz.process import cdist
    from rapidfuzz.distance import Levenshtein

    if seq1==None: seq1=seq
    D = cdist(seq, seq1, scorer=Levenshtein.distance)
    return(np.array(D, dtype=float))

def weighted_levenshtein_dist(seq, seq1=None, 
                              insertion=1+np.log(4), deletion = 1+np.log(4), substitution=1):
    from rapidfuzz.process import cdist
    from rapidfuzz.distance import Levenshtein

    def mydist(*args, **kwargs):
        # this adds weights to indels vs subs according to James's pC analysis
        # note weights must me integers, so multiplying by 10^3 to get 2-point precision
        return(Levenshtein.distance(*args, **kwargs, weights=(insertion*10**3,deletion*10**3,substitution*10**3))/10**3) 

    if seq1==None: seq1=seq
    D = cdist(seq, seq1, scorer=mydist)
    return(np.array(D, dtype=float))

def triplet_similarity(seq, seq1=None):
    import rpy2.robjects.packages as rpackages
    from rpy2.robjects.vectors import StrVector
    import rpy2.robjects as robjects
    # import R's utility package - this allows you to install the packages you don't already have
    utils = rpackages.importr('utils')
    # select a mirror for R packages
    utils.chooseCRANmirror(ind=1)
    packnames = ['kernlab']
    names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
    if len(names_to_install) > 0:
        utils.install_packages(StrVector(names_to_install))
    r = robjects.r
    r['source']('pyrepseq/TripletKernel.R')
    triplet_similarity = robjects.globalenv['triplet_similarity']

    if seq1==None: seq1=seq
    D = np.array(triplet_similarity(seq, seq1))
    return(np.array(D, dtype=float))

def triplet_diversity(seq, seq1=None):
    D = 1-triplet_similarity(seq, seq1)
    return(np.array(D, dtype=float))

def tcrdist_cdr3s(seq, seq1=None, type='cdr3_b_aa'):
    # not tested because I am trusting pwseqdist implementation
    assert type in ['cdr3_b_aa', 'cdr3_a_aa']
    import pwseqdist as pw
    if seq1==None: seq1=seq

    kargs ={"cdr3_a_aa" : {'use_numba': True, 'distance_matrix': pw.matrices.tcr_nb_distance_matrix, 'dist_weight': 1, 'gap_penalty':4, 'ntrim':3, 'ctrim':2, 'fixed_gappos':False},
            "pmhc_a_aa" : {'use_numba': True, 'distance_matrix': pw.matrices.tcr_nb_distance_matrix, 'dist_weight': 1, 'gap_penalty':4, 'ntrim':0, 'ctrim':0, 'fixed_gappos':True},
            "cdr2_a_aa" : {'use_numba': True, 'distance_matrix': pw.matrices.tcr_nb_distance_matrix, 'dist_weight': 1, 'gap_penalty':4, 'ntrim':0, 'ctrim':0, 'fixed_gappos':True},
            "cdr1_a_aa" : {'use_numba': True, 'distance_matrix': pw.matrices.tcr_nb_distance_matrix, 'dist_weight': 1, 'gap_penalty':4, 'ntrim':0, 'ctrim':0, 'fixed_gappos':True},
            "cdr3_b_aa" : {'use_numba': True, 'distance_matrix': pw.matrices.tcr_nb_distance_matrix, 'dist_weight': 1, 'gap_penalty':4, 'ntrim':3, 'ctrim':2, 'fixed_gappos':False},
            "pmhc_b_aa" : {'use_numba': True, 'distance_matrix': pw.matrices.tcr_nb_distance_matrix, 'dist_weight': 1, 'gap_penalty':4, 'ntrim':0, 'ctrim':0, 'fixed_gappos':True},
            "cdr2_b_aa" : {'use_numba': True, 'distance_matrix': pw.matrices.tcr_nb_distance_matrix, 'dist_weight': 1, 'gap_penalty':4, 'ntrim':0, 'ctrim':0, 'fixed_gappos':True},
            "cdr1_b_aa" : {'use_numba': True, 'distance_matrix': pw.matrices.tcr_nb_distance_matrix, 'dist_weight': 1, 'gap_penalty':4, 'ntrim':0, 'ctrim':0, 'fixed_gappos':True}}
    D= pw.apply_pairwise_rect(metric = pw.metrics.nb_vector_tcrdist, seqs1 = seq, seqs2 = seq1, **kargs[type])
    return(np.array(D, dtype=float))
