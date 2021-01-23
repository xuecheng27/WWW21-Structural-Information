from igraph import Graph
import numpy as np 
from scipy.sparse import csr_matrix
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh

def __get_adjacency_sparse(G):
    """Get the adjacency matrix of the graph G in sparse form

    Arguments:
        G (igraph.Graph): the unweighted, undirected graph
    
    Returns:
        A (csr_matrix): a sparse adjacency matrix
    """
    edges = G.get_edgelist()
    weights = [1] * len(edges)
    N = G.vcount()
    sparse_matrix = csr_matrix((weights, list(zip(*edges))), shape=(N, N), dtype=np.float64)
    A = sparse_matrix + sparse_matrix.T 
    di = np.diag_indices(N)
    A[di] /= 2
    return A

def __compute_Q(G):
    """Compute the quadratic approximation Q of the von Neumann graph entropy

    Arguments:
        G (igraph.Graph): the unweighted, undirected graph to be analyzed

    Returns:
        Q (float): the quadratic approximation
    """
    volume = 2 * G.ecount()
    
    volume_square = 0
    for deg in G.degree():
        volume_square += deg * deg
    
    W = G.ecount()

    Q = 1 - (volume_square + 2 * W) / (volume ** 2)
    return Q

def finger_hat_entropy(G):
    """Compute an approximation of von Neumann graph entropy using Q and maximum eigenvalue

    Arguments:
        G (igraph.Graph): the unweighted, undirected graph

    Returns:
        hat_H (float): an approximation of the von Neumann graph entropy
    """
    volume = 2 * G.ecount()
    Q = __compute_Q(G)

    adjacency = __get_adjacency_sparse(G)
    laplacian = csgraph.laplacian(adjacency, normed=False)
    eigmax = eigsh(laplacian, 1, return_eigenvectors=False)[0]

    hat_H = -Q * np.log2(eigmax / volume)

    return hat_H

def finger_tilde_entropy(G):
    """Compute an approximation of von Neumann graph entropy using Q and max_degree

    Arguments:
        G (igraph.Graph): the unweighted, undirected graph to be analyzed

    Returns:
        tilde_H (float): an approximation of the von Neumann graph entropy
    """
    volume = 2 * G.ecount()
    max_degree = np.array(G.degree()).max()

    Q = __compute_Q(G)
    tilde_H = -Q * np.log2(2 * max_degree / volume)

    return tilde_H