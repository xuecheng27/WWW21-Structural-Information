from math import log
import numpy as np 
from scipy.sparse import csr_matrix
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh

def __xlogx(x):
    epsilon = 10 ** (-8)
    if x <= epsilon:
        return 0
    else:
        return x * log(x, 2)

def __get_adjacency_sparse(graph):
    # return the adjacency matrix of input graph in sparse matrix form
    # 'Graph.get_adjacency_sparse()' has some bug dealing with undirected graph in igraph version 0.8.0
    edges = graph.get_edgelist()
    weights = [1] * len(edges)
    N = graph.vcount()
    sparse_matrix = csr_matrix((weights, list(zip(*edges))), shape=(N, N), dtype=np.float64)
    sparse_matrix = sparse_matrix + sparse_matrix.T 
    di = np.diag_indices(N)
    sparse_matrix[di] /= 2
    return sparse_matrix

def one_dimensional_structural_entropy(graph):
    degree_seq = graph.vs.degree()
    vol = sum(degree_seq)

    entropy = 0
    for deg in degree_seq:
        entropy -= __xlogx(deg / vol)
    return entropy

def von_Neumann_entropy(graph, mode='laplacian'):
    # compute the exact von Neumann entropy for small size graphs
    adjacency = __get_adjacency_sparse(graph)
    if mode == 'laplacian' or mode == 'Laplacian':
        # laplacian = np.array(graph.laplacian(normalized=False))
        laplacian = csgraph.laplacian(adjacency, normed=False)
    elif mode == 'normalized laplacian' or mode == 'Normalized Laplacian':
        # laplacian = np.array(graph.laplacian(normalized=True))
        laplacian = csgraph.laplacian(adjacency, normed=True)
    else:
        raise Exception

    eigenvalues = eigsh(laplacian, graph.vcount() - 1, return_eigenvectors=False)
    eigsum = eigenvalues.sum()

    entropy = 0
    for eigval in eigenvalues:
        entropy -= __xlogx(eigval / eigsum)
    return entropy