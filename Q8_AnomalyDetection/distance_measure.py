import numpy as np 
from igraph import Graph
from scipy.sparse import csr_matrix
from Q7_IncreSim.deltaCon.DeltaCon import deltaCon
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh

def von_Neumann_distance(G1, G2):
    A1 = __get_adjacency_sparse(G1)
    A2 = __get_adjacency_sparse(G2)
    L1 = csgraph.laplacian(A1, normed=False)
    L2 = csgraph.laplacian(A2, normed=False)

    L = L1 / L1.diagonal().sum() + L2 / L2.diagonal().sum()

    S1 = __von_Neumann_entropy(L1)
    S2 = __von_Neumann_entropy(L2)
    S12 = __von_Neumann_entropy(L)

    return np.sqrt(S12 - (S1 + S2) / 2)

def delta_con(G1, G2):
    A1 = __get_adjacency_sparse(G1)
    A2 = __get_adjacency_sparse(G2)

    return 1 - deltaCon(A1, A2, 30)

def veo_score(G1, G2):
    N, M1, M2 = G1.vcount(), G1.ecount(), G2.ecount()

    M12 = 0
    for e in G1.get_edgelist():
        if G2.are_connected(e[0], e[1]):
            M12 += 1

    return 1 - 2 * (N + M12) / (2 * N + M1 + M2)

def structural_information_distance(G1, G2):
    N = G1.vcount()
    degree_seq1 = np.array(G1.degree())
    degree_seq2 = np.array(G2.degree())
    vol1 = degree_seq1.sum()
    vol2 = degree_seq2.sum()

    S1 = 0
    for deg in degree_seq1:
        S1 -= __xlog2x(deg / vol1)

    S2 = 0
    for deg in degree_seq2:
        S2 -= __xlog2x(deg / vol2)

    S12 = 0
    for i in range(N):
        deg1 = degree_seq1[i]
        deg2 = degree_seq2[i]
        S12 -= __xlog2x(deg1 / (2 * vol1) + deg2 / (2 * vol2))

    return np.sqrt(S12 - (S1 + S2) / 2)

def __xlog2x(x):
    if x <= 1e-9:
        return 0
    else:
        return x * np.log2(x)

def __get_adjacency_sparse(g):
    N, M = g.vcount(), g.ecount()
    data = np.zeros(2 * M, dtype=np.float64)
    row = np.zeros(2 * M, dtype=int)
    col = np.zeros(2 * M, dtype=int)
    for idx, e in enumerate(g.get_edgelist()):
        data[2 * idx] = 1
        row[2 * idx] = e[0]
        col[2 * idx] = e[1]
        data[2 * idx + 1] = 1
        row[2 * idx + 1] = e[1]
        col[2 * idx + 1] = e[0]
    adjacency_matrix_sparse = csr_matrix((data, (row, col)), shape=(N, N), dtype=np.float64)
    return adjacency_matrix_sparse

def __von_Neumann_entropy(L):
    eigenvalues = eigsh(L, L.shape[0] - 1, return_eigenvectors=False)
    eigsum = eigenvalues.sum()

    entropy = 0
    for eigval in eigenvalues:
        entropy -= __xlog2x(eigval / eigsum)

    return entropy