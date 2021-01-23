from utils.graphIO import read_edgelist
from utils.graphInfo import get_edgeinfo
import numpy as np
import logging.config
import settings

logging.config.dictConfig(settings.LOGGING_SETTINGS)
logger = logging.getLogger('normal')

class GraphSimilarity(object):
    """Compute the similarity between adjacent graphs in a graph stream
       using veoScore algorithm.

    Arguments:
        filepath_list (List[str]): a list of filepath for the edges
        head_name_list (List[str]): a list of string specifying the name of each data column

    Returns:
        similarity_list (List[float]): a list of scores
    """
    def __init__(self, filepath_list, head_name_list):
        self.__filepath_list = filepath_list
        self.__head_name_list = head_name_list

        self.__graph = None
        self.__vcount = None
        self.__ecount = None
        self.__similarity = np.zeros(len(self.__filepath_list) - 1)

    def __preprocess(self):
        self.__graph = read_edgelist(self.__filepath_list[0], self.__head_name_list)
        self.__vcount = self.__graph.vcount()
        self.__ecount = self.__graph.ecount()

    def __increment(self, g):
        new_vertices = [0] * g.vcount()
        incre_vcount = 0
        for v in g.vs['name']:
            try:
                self.__graph.vs.find(v)
            except ValueError:
                new_vertices[incre_vcount] = v
                incre_vcount += 1
        self.__graph.add_vertices(new_vertices[:incre_vcount])

        incre_ecount = 0
        for e in g.get_edgelist():
            u, v, w = get_edgeinfo(g, e[0], e[1])
            if not self.__graph.are_connected(u, v) and w > 0:
                incre_ecount += 1
            if self.__graph.are_connected(u, v):
                eid = self.__graph.get_eid(u, v)
                curr_w = self.__graph.es[eid]['weight']
                if w > 0 and curr_w <= 0 and w + curr_w > 0:
                    incre_ecount += 1
                if w < 0 and curr_w + w <= 0:
                    incre_ecount -= 1

        # update self.__graph
        new_edges = [0] * g.ecount()
        new_weights = [0] * g.ecount()
        num_new_edges = 0
        for e in g.get_edgelist():
            u, v, w = get_edgeinfo(g, e[0], e[1])
            if not self.__graph.are_connected(u, v):
                new_edges[num_new_edges] = (u, v)
                new_weights[num_new_edges] = w
                num_new_edges += 1
            else:
                eid = self.__graph.get_eid(u, v)
                self.__graph.es[eid]['weight'] += w
        self.__graph.add_edges(new_edges[:num_new_edges])
        for i in range(num_new_edges):
            e = new_edges[i]
            w = new_weights[i]
            eid = self.__graph.get_eid(e[0], e[1])
            self.__graph.es[eid]['weight'] = w

        similarity = (incre_vcount + incre_ecount) / (incre_vcount + incre_ecount + 2 * (self.__vcount + self.__ecount))
        self.__vcount += incre_vcount
        self.__ecount += incre_ecount

        return similarity

    def run(self):
        logger.info("=" * 60)
        logger.info('Algorithm: veoScore')
        logger.info("=" * 60)
        self.__preprocess()

        for i in range(len(self.__filepath_list) - 1):
            g = read_edgelist(self.__filepath_list[i + 1], self.__head_name_list)
            self.__similarity[i] = self.__increment(g)
            logger.info(f'({self.__similarity[i]:8.7f})')
        
        logger.info("=" * 60)
        logger.info("\n\n")
        
        return self.__similarity