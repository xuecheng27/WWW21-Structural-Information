from igraph import Graph
import logging.config
import settings
import numpy as np
from scipy.linalg import eigh

logging.config.dictConfig(settings.LOGGING_SETTINGS)
logger = logging.getLogger('normal')

class ExtensibilityTest(object):
    """Evaluate the impact of edge weights on the entropy gap.

    Arguments:
        graph_path (str): the filepath of the simple real-world graph
        weight_range (List[int]): the list of maximum weights
    """
    def __init__(self, graph_path, weight_range):
        self.__graph_path = graph_path
        self.__weight_range = weight_range

    def __xlogx(self, x):
        if x <= 10 ** (-8):
            return 0
        else:
            return x * np.log2(x)

    def __compute_structural_information(self, laplacian_matrix):
        structural_information = 0
        degree_sum = np.trace(laplacian_matrix)
        for d in np.diagonal(laplacian_matrix):
            structural_information -= self.__xlogx(d / degree_sum)

        return structural_information

    def __compute_von_Neumann_entropy(self, laplacian_matrix):
        von_Neumann_entropy = 0
        degree_sum = np.trace(laplacian_matrix)

        for eigval in eigh(laplacian_matrix, eigvals_only=True):
            von_Neumann_entropy -= self.__xlogx(eigval / degree_sum)

        return von_Neumann_entropy

    def __start(self):
        logger.info("=" * 60)
        logger.info(f'Graph: {self.__graph_path}')
        logger.info("=" * 60)

    def __quit(self):
        logger.info("=" * 60)
        logger.info('The End')
        logger.info("=" * 60)
        logger.info("\n\n")

    def run(self):
        self.__start()

        for w in self.__weight_range:
            g = Graph.Read_GML(self.__graph_path)
            g.es['weight'] = 1.0
            for e in g.get_edgelist():
                g[e[0], e[1]] = np.random.uniform(1, w)
            
            laplacian = np.asarray(g.laplacian(weights='weight'))
            structural_information = self.__compute_structural_information(laplacian)
            von_Neumann_entropy = self.__compute_von_Neumann_entropy(laplacian)
            entropy_gap = structural_information - von_Neumann_entropy
            logger.info(f'structural information: ({structural_information:8.7f}), von Neumann entropy: ({von_Neumann_entropy:8.7f}), entropy gap: ({entropy_gap:8.7f})')

        self.__quit()