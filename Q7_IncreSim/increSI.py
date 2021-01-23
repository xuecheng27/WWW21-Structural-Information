import numpy as np
from utils.graphIO import read_edgelist
from utils.graphInfo import get_volume
import logging.config
import settings

logging.config.dictConfig(settings.LOGGING_SETTINGS)
logger = logging.getLogger('normal')

class GraphSimilarity(object):
    """Compute the similarity between adjacent graphs in a graph stream
       using structural information.

    Arguments:
        filepath_list (List[string]): a list of filepath for the edges
        head_name_list (List[str]): a list of string specifying the names of each data column

    Returns:
        similarity_list (List[float]): a list of scores 
    """
    def __init__(self, filepath_list, head_name_list=['from', 'to', 'weight']):
        self.__filepath_list = filepath_list
        self.__head_name_list = head_name_list

        self.__degree_dict = dict()
        self.__volume = None
        self.__si = None
        self.__similarity = np.zeros(len(self.__filepath_list) - 1)

    def __xlog2x(self, x):
        if x <= 1e-8:
            return 0
        else:
            return x * np.log2(x)

    def __preprocess(self):
        g = read_edgelist(self.__filepath_list[0], self.__head_name_list)
        self.__volume = get_volume(g)
        for i in range(g.vcount()):
            self.__degree_dict[g.vs[i]['name']] = g.strength(i, weights='weight')

        si = 0
        for d in self.__degree_dict.values():
            si -= self.__xlog2x(d)
        si /= self.__volume
        si += np.log2(self.__volume)
        self.__si = si

    def __increSim(self, g):
        g_volume = 2 * np.array(g.es['weight']).sum()
        g_degree_dict = dict()
        for i in range(g.vcount()):
            g_degree_dict[g.vs[i]['name']] = g.strength(i, weights='weight')
        
        c = (self.__volume + g_volume / 2) / (self.__volume * (self.__volume + g_volume))
        a, b, y, z = 0, 0, 0, 0
        for v in g_degree_dict.keys():
            g_deg = g_degree_dict[v]
            deg = self.__degree_dict[v] if v in self.__degree_dict else 0

            a += self.__xlog2x(deg + g_deg) - self.__xlog2x(deg)
            b += self.__xlog2x(deg / (2 * self.__volume) + (deg + g_deg) / (2 * (self.__volume + g_volume)))
            y += deg
            z += self.__xlog2x(deg)
        
        new_si = (self.__xlog2x(self.__volume + g_volume) - a - self.__xlog2x(self.__volume) + self.__volume * self.__si) / (self.__volume + g_volume)
        average_si = -b - (self.__volume - y) * self.__xlog2x(c) - c * (self.__xlog2x(self.__volume)- self.__volume * self.__si - z)
        similarity = np.sqrt(average_si - (new_si + self.__si) / 2)

        self.__volume += g_volume
        self.__si = new_si
        for v in g_degree_dict.keys():
            if v in self.__degree_dict:
                self.__degree_dict[v] += g_degree_dict[v]
            else:
                self.__degree_dict[v] = 0

        return similarity

    def run(self):
        logger.info("=" * 60)
        logger.info('Algorithm: structural information')
        logger.info("=" * 60)
        self.__preprocess()

        for i in range(len(self.__filepath_list) - 1):
            g = read_edgelist(self.__filepath_list[i + 1], self.__head_name_list)
            self.__similarity[i] = self.__increSim(g)
            logger.info(f'({self.__similarity[i]:8.7f})')
        
        logger.info("=" * 60)
        logger.info("\n\n")

        return self.__similarity