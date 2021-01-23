from igraph import Graph
from math import log
from scipy.sparse.linalg import eigsh
import numpy as np 
import logging.config
import settings
import time
import sys
import random
from utils.timer import time_mark
from metrics.entropy import von_Neumann_entropy

logging.config.dictConfig(settings.LOGGING_SETTINGS)
logger = logging.getLogger('normal')

class MaxEntropy(object):
    """Maximize the entropy of a graph via adding edges

    Arguments:
        graph_path (str): the filepath of the graph
        budget (int): the maximum number of edges that are allowed to add
        interval (int): the time interval to monitor the dynamics of graph entropy
        method (str): the algorithm to be used, including 'greedy', 'random', and 'algebraic connectivity'
    """
    def __init__(self, graph_path, budget, interval, method):
        self.__graph_path = graph_path
        self.__budget = budget
        self.__interval = interval
        self.__method = method

        self.__graph = None
        self.__ecount = None
        self.__vcount = None
        self.__sorted_degree_sequence = list()
        self.__sum_dlogd = 0
        self.__von_Neumann_entropy = None
        self.__structural_information = None

        self.__should_stop = False
        self.__start_time = time.time()
        self.__end_time = None
        self.__count = 1

    def __xlogx(self, x):
        if x <= 10 ** (-8):
            return 0
        else:
            return x * log(x, 2)

    def __preprocess(self):
        self.__graph = Graph.Read_GML(self.__graph_path)
        self.__ecount = self.__graph.ecount()
        self.__vcount = self.__graph.vcount()
        self.__sorted_degree_sequence = list(range(self.__vcount))
        self.__von_Neumann_entropy = von_Neumann_entropy(self.__graph)

        if self.__method == 'greedy':
            self.__sorted_degree_sequence.sort(key=lambda x: self.__graph.degree(x))
            for deg in self.__graph.vs.degree():
                self.__sum_dlogd += self.__xlogx(deg)
            self.__structural_information = log(2 * self.__ecount, 2) - self.__sum_dlogd / (2 * self.__ecount)

    def __start(self):
        logger.info("=" * 60)
        logger.info(self.__method)
        logger.info(f'Time: {time_mark(self.__start_time)}')
        logger.info(f'Graph: {self.__graph_path}')
        logger.info(f'Vcount: {self.__vcount}')
        logger.info(f'Ecount: {self.__ecount}')
        logger.info(f'Budget: {self.__budget}')
        logger.info(f'Interval: {self.__interval}')
        logger.info("=" * 60)
        
        # self.__tiktok()

        if self.__method == 'greedy':
            logger.info(f'structural information: ({self.__structural_information:8.7f}), von Neumann entropy: ({self.__von_Neumann_entropy:8.7f})')
        else:
            logger.info(f'von Neumann entropy: ({self.__von_Neumann_entropy:8.7f})')


    def __quit(self):
        self.__end_time = time.time()

        logger.info("=" * 60)
        logger.info(f'Ecount: {self.__ecount}')
        logger.info(f'Time: {time_mark(self.__end_time)}')
        logger.info(f'Total: {(self.__end_time - self.__start_time): 10.4f} s')
        logger.info("=" * 60)
        logger.info("\n\n")

    def __should_count(self, count):
        return divmod(count, self.__interval)[1]

    def __random(self):
        u = random.choice(self.__sorted_degree_sequence)
        v = random.choice(self.__sorted_degree_sequence)
        while u == v or self.__graph.are_connected(u, v):
            u = random.choice(self.__sorted_degree_sequence)
            v = random.choice(self.__sorted_degree_sequence)
        
        self.__graph.add_edge(u, v)
        self.__ecount += 1

    def __max_algebraic_connectivity(self):
        laplacian = self.__graph.laplacian(normalized=False)
        laplacian = np.array(laplacian, dtype=float)
        eigenvalues, eigenvectors = eigsh(laplacian, k=2, which='SA', tol=0.001)
        fiedler_vector = eigenvectors[:, 1]
        sorted_vertices = list(range(self.__vcount))
        sorted_vertices.sort(key=lambda x: fiedler_vector[x])

        for i in range(len(sorted_vertices) - 1):
            for j in range(len(sorted_vertices) - 1, i, -1):
                if not self.__graph.are_connected(sorted_vertices[i], sorted_vertices[j]):
                    break
            if not self.__graph.are_connected(sorted_vertices[i], sorted_vertices[j]):
                break
        u, v = sorted_vertices[i], sorted_vertices[j]
        head, tail = i, j
        threshold = (fiedler_vector[u] - fiedler_vector[v]) ** 2
        for j in range(tail + 1, len(sorted_vertices)):
            for i in range(head + 1, j):
                delta = (fiedler_vector[sorted_vertices[i]] - fiedler_vector[sorted_vertices[j]]) ** 2
                if delta <= threshold:
                    break
                if not self.__graph.are_connected(sorted_vertices[i], sorted_vertices[j]):
                    u, v = sorted_vertices[i], sorted_vertices[j]
                    threshold = delta
                    break
        
        self.__graph.add_edge(u, v)
        self.__ecount += 1

    def __greedy(self):
        head, tail = 0, len(self.__sorted_degree_sequence) - 1
        threshold = sys.maxsize

        while head < tail:
            d_head = self.__graph.degree(self.__sorted_degree_sequence[head])
            for i in range(head+1, tail+1):
                d_i = self.__graph.degree(self.__sorted_degree_sequence[i])
                delta = self.__xlogx(d_head + 1) - self.__xlogx(d_head) + self.__xlogx(d_i + 1) - self.__xlogx(d_i)
                if delta >= threshold:
                    tail = i - 1
                    break
                if not self.__graph.are_connected(self.__sorted_degree_sequence[head], self.__sorted_degree_sequence[i]):
                    u = self.__sorted_degree_sequence[head]
                    v = self.__sorted_degree_sequence[i]
                    threshold = delta
                    tail = i - 1
                    break
            head += 1
        
        new_structural_information = log(2 * self.__ecount + 2, 2) - (self.__sum_dlogd + threshold) / (2 * self.__ecount + 2)
        if np.abs(self.__structural_information - np.log2(self.__vcount)) <= 1e-9:
            self.__should_stop = True
        else:
            self.__structural_information = new_structural_information
            self.__sum_dlogd += threshold
            self.__ecount += 1
            self.__graph.add_edge(u, v)
            self.__sorted_degree_sequence.sort(key=lambda x: self.__graph.degree(x))

    def __analyze(self):
        self.__von_Neumann_entropy = von_Neumann_entropy(self.__graph)
        if self.__method == 'greedy':
            logger.info(f'structural information: ({self.__structural_information:8.7f}), von Neumann entropy: ({self.__von_Neumann_entropy:8.7f})')
        else:
            logger.info(f'von Neumann entropy: ({self.__von_Neumann_entropy:8.7f})')

    def __tiktok(self):
        current_time = time.time() - self.__start_time
        logger.info(f'current time: ({current_time:10.4f})')

    def run(self):
        self.__preprocess()
        self.__start()

        while self.__count <= self.__budget and not self.__should_stop:
            if self.__method == 'greedy':
                self.__greedy()
            elif self.__method == 'random':
                self.__random()
            elif self.__method == 'algebraic connectivity':
                self.__max_algebraic_connectivity()
            
            if not self.__should_count(self.__count) and not self.__should_stop:
                self.__analyze()
                # self.__tiktok()
            self.__count += 1

        self.__quit()
