import numpy as np
import pandas as pd
import scipy

import igraph
import sklearn.cluster

def graph_clustering(adjacency_matrix, seqs, clustering='cc', **kwargs):
    """
    Identify clusters in an unweighted sequence adjacency graph.

    adjacency_matrix: array_like of (i, j, dist)
    seqs: labels for the nodes in the graph
    clustering: choice of clustering algorithm
                'cc' or one of 'fastgreedy', 'multilevel', 'leiden'
                see igraph.community_* for documentation of these clustering algorithms
    **kwargs: passed on to community finding algorithm

    Returns: DataFrame with columns seq, cluster assignment 
    """
    edges = np.array(adjacency_matrix)[:, :2]

    if clustering == 'DBSCAN':
        a = adjacency_matrix
        neighbors_sparse = scipy.sparse.coo_array((a[:, 2], (a[:, 0], a[:, 1])),
                                                  shape=(len(seqs), len(seqs)))

        opt = sklearn.cluster.DBSCAN(metric='precomputed', min_samples=2,
                                     eps=np.amax(adjacency_matrix[:, 2]))
        labels = opt.fit_predict(neighbors_sparse)
        cluster_df = pd.DataFrame(dict(junction_aa=seqs, cluster=labels))
        cluster_df = cluster_df[cluster_df != -1]
    else:
        g = igraph.Graph(edges, n=len(seqs))
        if clustering == 'cc':
            components = g.connected_components(mode='weak')
        else:
            g.simplify()
            components = eval(f'g.community_{clustering}')(**kwargs)
            try:
                components = components.as_clustering()
            except AttributeError:
                pass
        cluster_df = pd.DataFrame(dict(junction_aa=seqs, cluster=components.membership))

    cluster_counts = cluster_df['cluster'].value_counts()
    expanded_cluster = set(cluster_counts[cluster_counts>1].index)
    cluster_df = cluster_df[cluster_df['cluster'].isin(expanded_cluster)]
    return cluster_df
