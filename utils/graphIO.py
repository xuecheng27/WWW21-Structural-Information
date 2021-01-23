import pandas as pd 
from igraph import Graph
import numpy as np 

def read_edgelist(filepath, heads=['from', 'to', 'weight']):
    """Read a graph from its edgelist.

    Arguments:
        filepath (str): filepath of the edgelist
        heads (List[str]): titles of each column in the edgelist

    Returns:
        g (igraph.Graph): a weighted, undirected graph without self-loops and multiple edges
    """
    df = pd.read_csv(filepath, sep='\s+', header=None, names=heads, dtype={'from':'str', 'to':'str', 'weight':np.float64})
    g = Graph.TupleList(df.values, directed=False, edge_attrs=heads[2:])
    g.simplify(combine_edges={'weight':sum})
    return g
