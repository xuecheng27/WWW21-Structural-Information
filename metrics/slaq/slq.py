"""Core Stochastic Lanczos Quadrature routines for SLaQ."""
from typing import Callable
from typing import List
from typing import Union
import numpy as np 
from scipy.sparse.base import spmatrix

def lanczos_m(matrix, lanczos_steps, nvectors):
    """Implementation of Lanczos algorithm for sparse matrices.

    Lanczos algorithm computes symmetric m x m tridiagonal matrix T
    and matrix V with orthogonal rows constituting the basis of the 
    Krylov subspace K_m(matrix, x), where x is an arbitrary starting unit vector.
    This implementation parallelizes 'nvectors' starting vectors.
    The notation follows https://en.wikipedia.org/wiki/Lanczos_algorithm.

    Arguments:
        matrix (spmatrix): Sparse input matrix.
        lanczos_steps (int): Number of Lanczos steps.
        nvectors (int): Number of random vectors.
    
    Returns:
        T (np.ndarray): A (nvectors x m x m) tensor, T[i, :, :] is the i-th symmetric
        tridiagonal matrix.
        V (np.ndarray): A (n x m x nvectors) tensor, V[:, :, i] is the i-th matrix
        with orthogonal rows.
    """
    start_vectors = np.random.randn(matrix.shape[0], nvectors).astype(np.float32) # Initialize random vectors in columns (n x nvectors).
    V = np.zeros((start_vectors.shape[0], lanczos_steps, nvectors), dtype=np.float32)
    T = np.zeros((nvectors, lanczos_steps, lanczos_steps), dtype=np.float32)

    np.divide(start_vectors, np.linalg.norm(start_vectors, axis=0), out=start_vectors) # Normalize each column.
    V[:, 0, :] = start_vectors

    # First Lanczos step.
    w = matrix @ start_vectors
    alpha = np.einsum('ij,ij->j', w, start_vectors)
    w -= alpha[None, :] * start_vectors
    beta = np.einsum('ij,ij->j', w, w)
    np.sqrt(beta, beta)

    T[:, 0, 0] = alpha
    T[:, 0, 1] = beta
    T[:, 1, 0] = beta

    np.divide(w, beta[None, :], out=w)
    V[:, 1, :] = w

    t = np.zeros((lanczos_steps, nvectors), dtype=np.float32)

    # Further Lanczos steps.
    for i in range(1, lanczos_steps):
        old_vectors = V[:, i - 1, :]
        start_vectors = V[:, i, :]

        w = matrix @ start_vectors
        w -= beta[None, :] * old_vectors
        np.einsum('ij,ij->j', w, start_vectors, out=alpha)
        T[:, i, i] = alpha

        if i < lanczos_steps - 1:
            w -= alpha[None, :] * start_vectors
            # Orthogonalize columns of V.
            np.einsum('ijk,ik->jk', V, w, out=t)
            w -= np.einsum('ijk,jk->ik', V, t)
            np.einsum('ij,ij->j', w, w, out=beta)
            np.sqrt(beta, beta)
            np.divide(w, beta[None, :], out=w)

            T[:, i, i + 1] = beta
            T[:, i + 1, i] = beta
            V[:, i + 1, :] = w

            if (np.abs(beta) > 1e-6).sum() == 0:
                break
    
    return T, V

def slq(matrix, m, nvectors, functions, scales = np.ones(1)):
    """Stochastic Lanczos Quadrature approximation of given matrix functions.

    Arguments:
        matrix (spmatrix): Sparse input matrix.
        m (int): Number of Lanczos steps.
        nvectors(int): Number of random vectors.
        functions (List[Callable[np.ndarray, np.ndarray]]): A list of functions over the matrix spectrum.
        scales (np.ndarray): An array of scales to parametrize the functions. By default no scaling of the spectrum is used.
    
    Returns:
        traces (np.ndarray): a (nvectors x m x m) tensor, T[i, :, :] is the i-th symmetric tridiagonal matrix.
    """
    T, _ = lanczos_m(matrix, m, nvectors)
    eigenvalues, eigenvectors = np.linalg.eigh(T)
    traces = np.zeros((len(functions), len(scales)))
    for i, function in enumerate(functions):
        expeig = function(np.outer(scales, eigenvalues)).reshape(len(scales), nvectors, m)
        sqeigv1 = np.power(eigenvectors[:, 0, :], 2)
        traces[i, :] = matrix.shape[-1] * (expeig * sqeigv1).sum(axis=-1).mean(axis=-1)
    
    return traces