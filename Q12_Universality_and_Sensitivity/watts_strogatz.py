from igraph import Graph
import logging.config
import settings
from utils.timer import time_mark
import time
from metrics.entropy import one_dimensional_structural_entropy
from metrics.entropy import von_Neumann_entropy
from metrics.entropy_bound import sharpened_lower_bound
from metrics.entropy_bound import sharpened_upper_bound

logging.config.dictConfig(settings.LOGGING_SETTINGS)
logger = logging.getLogger('normal')

class WattsStrogatzTest(object):
    """Compute the structural information, von Neumann entropy, entropy gap, lower bound, upper bound
       of synthetic graphs generated from Watts Strogatz random graph model.

    Arguments:
        n (int): the number of nodes in the ring graph
        k (int): the average degree of the ring graph, must be even
        prange (List[float]): the edge rewiring probability
    """
    def __init__(self, n, k, prange):
        self.__vcount = n 
        self.__average_degree = k 
        self.__rewire_probability_list = prange

        self.__start_time = time.time()
        self.__graph = None
        self.__end_time = None

    def __start(self):
        logger.info("=" * 60)
        logger.info(f"WattsStrogatz{self.__average_degree}")
        logger.info(f'Time: {time_mark(self.__start_time)}')
        logger.info(f'Vcount: {self.__vcount}')
        logger.info(f'Average degree: {self.__average_degree}')
        logger.info(f'Rewire probability: {self.__rewire_probability_list}')
        logger.info("=" * 60)

    def __quit(self):
        self.__end_time = time.time()
        logger.info("=" * 60)
        logger.info(f'Time: {time_mark(self.__end_time)}')
        logger.info(f'Total: {(self.__end_time - self.__start_time): 10.4f} s')
        logger.info("=" * 60)
        logger.info("\n\n")

    def __analyze(self):
        structural_entropy = one_dimensional_structural_entropy(self.__graph)
        exact_von_neumann = von_Neumann_entropy(self.__graph)
        entropy_gap = structural_entropy - exact_von_neumann
        gap_lower_bound = sharpened_lower_bound(self.__graph)
        gap_upper_bound = sharpened_upper_bound(self.__graph)
        
        logger.info(f"structural entropy: ({structural_entropy:8.7f}), von Neumann entropy: ({exact_von_neumann:8.7f}), entropy gap: ({entropy_gap:8.7f}), "
                    f"lower bound: ({gap_lower_bound:8.7f}), upper bound: ({gap_upper_bound:8.7f})")

    def __experiment(self):
        for p in self.__rewire_probability_list:
            self.__graph = Graph.Watts_Strogatz(1, self.__vcount, int(self.__average_degree / 2), p)
            self.__analyze()

    def run(self):
        self.__start()
        self.__experiment()
        self.__quit()