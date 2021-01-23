"""Utility functions for SLaQ"""
import numpy as np 
from igraph import Graph
import scipy.sparse
from scipy.sparse.base import spmatrix

def laplacian(adjacency, normalized = True):
    """Computes the sparse Laplacian matrix given sparse adjacency matrix as input.

    Args:
        adjacency (spmatrix): Input adjacency matrix of a graph.
        normalized (bool): If True, return the normalized version of the Laplacian.

    Returns:
        spmatrix: Sparse Laplacian matrix of the graph.
    """
    degree = np.squeeze(np.asarray(adjacency.sum(axis=1)))
    if not normalized:
        return scipy.sparse.diags(degree) - adjacency
    with np.errstate(divide='ignore'): # Ignore the warning for divide by 0 case.
        degree = 1. / np.sqrt(degree)
    degree[degree == np.inf] = 0
    degree = scipy.sparse.diags(degree)
    return scipy.sparse.eye(adjacency.shape[0], dtype=np.float32) - degree @ adjacency @ degree

def get_adjacency(graph):
    vcount = graph.vcount()
    adjacency = scipy.sparse.dok_matrix((vcount, vcount), dtype=np.float32)
    for e in graph.get_edgelist():
        adjacency[e[0], e[1]] = 1
        adjacency[e[1], e[0]] = 1
    # Convert to the compressed sparse row format for efficiency.
    adjacency = adjacency.tocsr()
    adjacency.data = np.ones(adjacency.data.shape, dtype=np.float32) # Set all elements to one in case of duplicate rows.

    return adjacency