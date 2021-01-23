import numpy as np
from scipy.sparse import identity
from scipy.sparse import diags

def fastBeliefPropagation(A, y):
    """A fast power method used to solve the equation (I+aD-cA)x=y

    Arguments:
        A (csr_matrix): Symmetric adjacency matrix of an undirected graph
        y (array): Vector of length n

    Returns:
        x (array): Vector of length n
    
    Reference:
        D. Koutra et al., 2011, in PKDD,
        "Unifying guilt-by-association approaches: theorems and fast algorithms"
    """
    # I = identity(A.shape[0]) # Identity matrix (dia_matrix)
    D = diags(sum(A).toarray(), [0]) # Diagonal matrix (dia_matrix)

    # Pick h to achieve convergence
    h1 = 1 / (2 + 2 * D.diagonal().max())
    c1 = 2 + D.diagonal().sum() # D.diagonal() (array)
    c2 = np.power(D.diagonal(), 2).sum() - 1
    h2 = np.sqrt((-c1 + np.sqrt(c1 * c1 + 4 * c2)) / (8 * c2))
    h = max(h1, h2)

    # Compute a and c using h
    a = 4 * (h * h) / (1 - 4 * (h * h))
    c = 2 * h / (1 - 4 * (h * h))

    W = c * A - a * D # csr_matrix
    x = np.array(y)
    delta_x = np.array(y)
    if W.max() <= 1e-9:
        return x

    for i in range(10):
        delta_x = W.dot(delta_x)
        x += delta_x
    
    return x