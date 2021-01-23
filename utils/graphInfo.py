import numpy as np

def get_volume(g):
    """Compute the volume of a graph.

    Arguments:
        g (igraph.Graph): a weighted, undirected graph without self-loops and multiple edges

    Returns:
        vol (float): the volume of a graph 
    """
    vol = 2 * np.array(g.es['weight']).sum()
    return vol

def get_edgeinfo(g, idx1, idx2):
    """Get the basic info of an edge in a graph.

    Arguments:
        g (igraph.Graph): a weighted, undirected graph without self-loops and multiple edges
        idx1 (int): the index of an endpoint
        idx2 (int): the index of another endpoint

    Returns:
        u (str): the name of an endpoint
        v (str): the name of another endpoint
        w (float): the weight of the edge
    """
    u = g.vs[idx1]['name']
    v = g.vs[idx2]['name']
    eid = g.get_eid(idx1, idx2)
    w = g.es[eid]['weight']

    return u, v, w