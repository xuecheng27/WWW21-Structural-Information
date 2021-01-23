import numpy as np
from collections import defaultdict
from utils.graphIO import read_edgelist
from utils.graphInfo import get_edgeinfo
from Q7_IncreSim.deltaCon.DeltaCon import deltaCon
from scipy.sparse import csr_matrix
import logging.config
import settings

logging.config.dictConfig(settings.LOGGING_SETTINGS)
logger = logging.getLogger('normal')

class GraphSimilarity(object):
    """Compute the similarity between adjacent graphs in a graph stream
       using deltaCon algorithm.

    Arguments:
        filepath_list (List[str]): a list of filepath for the edges
        head_name_list (List[str]): a list of string specifying the name of each data column

    Returns:
        similarity_list (List[float]): a list of scores
    """
    def __init__(self, filepath_list, head_name_list):
        self.__filepath_list = filepath_list
        self.__head_name_list = head_name_list

        self.__node_indices = dict()
        self.__adjacency_matrix_sparse = None
        self.__vcount = None
        self.__similarity = np.zeros(len(self.__filepath_list) - 1)

    def __get_adjacency_sparse(self, g):
        data = np.zeros(2 * g.ecount(), dtype=np.float64)
        row = np.zeros(2 * g.ecount(), dtype=int)
        col = np.zeros(2 * g.ecount(), dtype=int)
        for idx, e in enumerate(g.get_edgelist()):
            u, v, w = get_edgeinfo(g, e[0], e[1])
            data[2 * idx] = w
            row[2 * idx] = self.__node_indices[u]
            col[2 * idx] = self.__node_indices[v]
            data[2 * idx + 1] = w
            row[2 * idx + 1] = self.__node_indices[v]
            col[2 * idx + 1] = self.__node_indices[u]
        
        adjacency_matrix_sparse = csr_matrix((data, (row, col)), shape=(self.__vcount, self.__vcount), dtype=np.float64)
        return adjacency_matrix_sparse

    def __preprocess(self):
        g = read_edgelist(self.__filepath_list[0], self.__head_name_list)
        self.__vcount = g.vcount()
        
        for i in range(g.vcount()):
            self.__node_indices[g.vs[i]['name']] = i

        self.__adjacency_matrix_sparse = self.__get_adjacency_sparse(g)

    def __increment(self, g):
        # update node indices and vcount
        for v in g.vs['name']:
            if v not in self.__node_indices:
                self.__node_indices[v] = self.__vcount
                self.__vcount += 1
        
        self.__adjacency_matrix_sparse.resize((self.__vcount, self.__vcount))

        g_adjacency_matrix = self.__get_adjacency_sparse(g)

        similarity = 1 - deltaCon(self.__adjacency_matrix_sparse, self.__adjacency_matrix_sparse + g_adjacency_matrix, 30)

        self.__adjacency_matrix_sparse += g_adjacency_matrix

        return similarity

    def run(self):
        logger.info("=" * 60)
        logger.info('Algorithm: deltaCon')
        logger.info("=" * 60)
        self.__preprocess()

        for i in range(len(self.__filepath_list) - 1):
            g = read_edgelist(self.__filepath_list[i + 1], self.__head_name_list)
            self.__similarity[i] = self.__increment(g)
            logger.info(f'({self.__similarity[i]:8.7f})')
        
        logger.info("=" * 60)
        logger.info("\n\n")

        return self.__similarity
