"""Main SLaQ interface for approximating graph descriptors NetLSD and VNGE."""
import numpy as np 
from scipy.sparse.base import spmatrix
from metrics.slaq.slq import slq
from metrics.slaq.util import laplacian
from metrics.slaq.util import get_adjacency
from igraph import Graph

def _slq_red_var_vnge(matrix, lanczos_steps, nvectors):
    """Approximates von Neumann graph entropy (VNGE) of a given matrix.

    Uses the control variates method to reduce the variance of VNGE estimation.

    Args:
        matrix (sparse matrix): Input adjacency matrix of a graph.
        lanczos_steps (int): Number of Lanczos steps.
        nvectors (int): Number of random vectors for stochastic estimation.

    Returns:
        float: Approximated von Neumann graph entropy.
    """
    functions = [lambda x: np.where(x > 0, -x * np.log(x), 0), lambda x: x]
    traces = slq(matrix, lanczos_steps, nvectors, functions).ravel()
    return (traces[0] - traces[1] + 1) / np.log(2) # base-2 entropy

def vnge(graph, lanczos_steps=10, nvectors=100):
    """Computes von Neumann graph entropy (VNGE) using SLaQ.

    Args:
        adjacency (scipy.sparse.base.spmatrix): Input adjacency matrix of a graph.
        lanczos_steps (int): Number of Lanczos steps. Setting lanczos_steps=10 is the default from SLaQ.
        nvectors (int): Number of random vectors for stochastic estimation. Setting nvectors=100 is the default values from the SLaQ paper.

    Returns:
        float: Approximated VNGE.
    """
    adjacency = get_adjacency(graph)
    if adjacency.nnz == 0: # By convention, if x=0, x*log(x)=0.
        return 0
    density = laplacian(adjacency, False)
    density.data /= np.sum(density.diagonal()).astype(np.float32)
    return _slq_red_var_vnge(density, lanczos_steps, nvectors)


def _slq_red_var_netlsd(matrix, lanczos_steps, nvectors, timescales):
    """Computes unnormalized NetLSD signatures of a given matrix.

    Uses the control variates method to reduce the variance of NetLSD estimation.

    Args:
        matrix (sparse matrix): Input adjacency matrix of a graph.
        lanczos_steps (int): Number of Lanczos steps.
        nvectors (int): Number of random vectors for stochastic estimation.
        timescales (np.ndarray): Timescale parameter for NetLSD computation. Default value is the one used in both NetLSD and SLaQ papers.

    Returns:
        np.ndarray: Approximated NetLSD descriptors.
    """
    functions = [np.exp, lambda x: x]
    traces = slq(matrix, lanczos_steps, nvectors, functions, -timescales)
    subee = traces[0, :] - traces[1, :] / np.exp(timescales)
    sub = -timescales * matrix.shape[0] / np.exp(timescales)
    return np.array(subee + sub)

def netlsd(graph, timescales=np.logspace(-2, 2, 256), lanczos_steps=10, nvectors=100, normalization=None):
    """Computes NetLSD descriptors using SLaQ.
    
    Args:
        adjacency (sparse matrix): Input adjacency matrix of a graph.
        timescales (np.ndarray): Timescale parameter for NetLSD computation. Default value is the one used in both NetLSD and SLaQ papers.
        lanczos_steps (int): Number of Lanczos steps. Setting lanczos_steps=10 is the default from SLaQ.
        nvectors (int): Number of random vectors for stochastic estimation. Setting nvectors=100 is the default values from the SLaQ paper.
        normalization (str): Normalization type for NetLSD.

    Returns:
        np.ndarray: Approximated NetLSD descriptors.
    """
    lap = laplacian(get_adjacency(graph), True)
    hkt = _slq_red_var_netlsd(lap, lanczos_steps, nvectors, timescales)
    if normalization is None:
        return hkt
    n = lap.shape[0]
    if normalization == 'empty':
        return hkt / n 
    elif normalization == 'complete':
        return hkt / (1 + (n - 1) * np.exp(-timescales))
    elif normalization is None:
        return hkt
    else:
        raise ValueError("Unknown normalization type: expected one of [None, 'empty', 'complete'], got", normalization)