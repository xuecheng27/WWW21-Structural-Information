from Q7_IncreSim.deltaCon.fastBP import fastBeliefPropagation
import numpy as np

def deltaCon(A1, A2, g):
    """A principled, intuitive, and scalable algorithm that assesses the 
       similarity between two graphs on the same nodes.

    Arguments:
        A1 (csr_matrix): Symmetric adjacency matrix of an undirected graph G1
        A2 (csr_matrix): Symmetric adjacency matrix of an undirected graph G2
        g (int): number of node partitions

    Returns:
        similarity (float): a value between 0 and 1

    Reference:
        D. Koutra et al., 2016, in TKDD,
        "DeltaCon: Principled massive-graph similarity function with attribution"
    """
    vcount = A1.shape[0] # the number of nodes

    # random partition
    partitions = dict()
    for i in range(g):
        partitions[i] = np.zeros(vcount)
    for idx, membership in enumerate(np.random.choice(range(g), vcount)):
        partitions[membership][idx] += 1
    
    for i in range(g):
        s0 = partitions[i]
        s1 = fastBeliefPropagation(A1, s0)
        s2 = fastBeliefPropagation(A2, s0)
        if i == 0:
            S1 = s1.reshape((-1, 1))
            S2 = s1.reshape((-1, 1))
        else:
            S1 = np.concatenate((S1, s1.reshape((-1, 1))), axis=1)
            S2 = np.concatenate((S2, s2.reshape((-1, 1))), axis=1)


    similarity = 1 / (1 + __rootED(S1, S2))
    return similarity

    
def __rootED(S1, S2):
    S = np.power(S1, 1 / 2) - np.power(S2, 1 / 2)
    return np.sqrt(np.power(S, 2).sum())