from igraph import Graph
import numpy as np 
import logging.config
import settings
from metrics.entropy_bound import sharpened_lower_bound
from metrics.entropy_bound import sharpened_upper_bound
from metrics.entropy import one_dimensional_structural_entropy
from metrics.entropy import von_Neumann_entropy
from metrics.finger.finger import finger_hat_entropy
from metrics.finger.finger import finger_tilde_entropy
from metrics.slaq import slaq

import time
from utils.timer import time_mark

logging.config.dictConfig(settings.LOGGING_SETTINGS)
logger = logging.getLogger('normal')

class RealGraphTest(object):
    """Compute structural information, von Neumann entropy, SLaQ, FINGER for a real-world graph.
       Record the running time at the same time.

    Arguments:
        graph_path (str): the filepath of the graph to be analyzed
        return_vnge (bool): whether or not compute the exact von Neumann entropy
    """
    def __init__(self, graph_path, return_vnge=True):
        self.__graph_path = graph_path
        self.__return_vnge = return_vnge

        self.__graph = None
        self.__vcount = None
        self.__ecount = None
        self.__von_Neumann_entropy = None
        self.__one_dimensional_structural_entropy = None
        self.__time_structural_information = None
        
        self.__approx_entropy_by_finger_hat = None
        self.__time_finger_hat = None
        self.__approx_entropy_by_finger_tilde = None
        self.__time_finger_tilde = None
        self.__approx_entropy_by_slaq = None
        self.__time_slaq = None

        self.__start_time = time.time()
        self.__end_time = None

    def __start(self):
        logger.info("=" * 60)
        logger.info(f'Graph: {self.__graph_path}')
        logger.info(f'Time: {time_mark(self.__start_time)}')
        logger.info(f'Vcount: {self.__vcount}')
        logger.info(f'Ecount: {self.__ecount}')
        logger.info("=" * 60)

    def __quit(self):
        self.__end_time = time.time()
        logger.info("=" * 60)
        logger.info(f'Time: {time_mark(self.__end_time)}')
        logger.info(f'Total: {(self.__end_time - self.__start_time): 10.4f} s')
        logger.info("=" * 60)
        logger.info("\n\n")

    def __preprocess(self):
        self.__graph = Graph.Read_GML(self.__graph_path)
        self.__vcount = self.__graph.vcount()
        self.__ecount = self.__graph.ecount()
        if self.__return_vnge:
            self.__von_Neumann_entropy = von_Neumann_entropy(self.__graph)

        tik = time.time()
        self.__one_dimensional_structural_entropy = one_dimensional_structural_entropy(self.__graph)
        tok = time.time()
        self.__time_structural_information = tok - tik

        tik = time.time()
        self.__approx_entropy_by_finger_hat = finger_hat_entropy(self.__graph)
        tok = time.time()
        self.__time_finger_hat = tok - tik

        tik = time.time()
        self.__approx_entropy_by_finger_tilde = finger_tilde_entropy(self.__graph)
        tok = time.time()
        self.__time_finger_tilde = tok - tik

        entropy_by_slaq = np.zeros(10)
        tik = time.time()
        for i in range(10):
            entropy_by_slaq[i] = slaq.vnge(self.__graph)
        tok = time.time()
        self.__time_slaq = (tok - tik) / 10
        self.__approx_entropy_by_slaq = entropy_by_slaq.mean()

    def __show(self):
        if self.__return_vnge:
            logger.info(f'von Neumann entropy: ({self.__von_Neumann_entropy:8.7f})')
        logger.info(f'structural information: ({self.__one_dimensional_structural_entropy:8.7f})')
        logger.info(f'time for structural information: ({self.__time_structural_information:10.7f}) s')
        logger.info(f'approximated entropy by slaq: ({self.__approx_entropy_by_slaq:8.7f})')
        logger.info(f'time for slaq: ({self.__time_slaq:10.7f}) s')
        logger.info(f'approximated entropy by finger hat: ({self.__approx_entropy_by_finger_hat:8.7f})')
        logger.info(f'time for finger hat: ({self.__time_finger_hat:10.7f}) s')
        logger.info(f'approximated entropy by finger tilde: ({self.__approx_entropy_by_finger_tilde:8.7f})')
        logger.info(f'time for finger tilde: ({self.__time_finger_tilde:10.7f}) s')

    def run(self):
        self.__preprocess()
        self.__start()
        self.__show()
        self.__quit()