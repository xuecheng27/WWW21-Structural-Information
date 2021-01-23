import matplotlib
matplotlib.rcParams['text.usetex'] = True
from igraph import Graph
import numpy as np 
from Q8_AnomalyDetection.distance_measure import *
import matplotlib.pyplot as plt
import logging.config
import settings

logging.config.dictConfig(settings.LOGGING_SETTINGS)
logger = logging.getLogger('normal')

class AnomalyDetection4BAGraph(object):
    """Model the DDoS attacks and detect the anomalous graph.

    Arguments:
        ddos_param (int): the number of malicious servers that are going to attack a target server
        num_exp (int): the number of repeated experiments
    """
    def __init__(self, ddos_param, num_exp):
        self.__ddos_param = ddos_param
        self.__num_experiments = num_exp

        self.__num_graph = 10
        self.__graph_stream = dict()

        self.__structural_information_distances = None
        self.__von_Neumann_distances = None
        self.__veo_distances = None
        self.__delta_con_distances = None

    def __preprocess(self):
        for i in range(self.__num_graph):
            self.__graph_stream[i] = Graph.Read_GML(f'datasets/synthetic/anomaly-BA-{i}.gml')

    def __generate_anomalous_graph(self, idx):
        anomalous_graph = Graph.Read_GML(f'datasets/synthetic/anomaly-BA-{idx}.gml')

        N = anomalous_graph.vcount()
        anomalous_source = np.random.choice(N)

        possible_targets = [i for i in range(N) if not anomalous_graph.are_connected(i, anomalous_source)]
        anomalous_targets = np.random.choice(possible_targets, self.__ddos_param)

        new_edges = [(anomalous_source, t) for t in anomalous_targets]
        anomalous_graph.add_edges(new_edges)

        return anomalous_graph

    def __analysis(self, idx, anomalous_graph):
        self.__structural_information_distances = np.zeros(self.__num_graph - 1)
        self.__von_Neumann_distances = np.zeros(self.__num_graph - 1)
        self.__veo_distances = np.zeros(self.__num_graph - 1)
        self.__delta_con_distances = np.zeros(self.__num_graph - 1)

        for i in range(self.__num_graph - 1):
            if idx == i:
                self.__structural_information_distances[i] = structural_information_distance(anomalous_graph, self.__graph_stream[i + 1])
                self.__von_Neumann_distances[i] = von_Neumann_distance(anomalous_graph, self.__graph_stream[i + 1])
                self.__veo_distances[i] = veo_score(anomalous_graph, self.__graph_stream[i + 1])
                self.__delta_con_distances[i] = delta_con(anomalous_graph, self.__graph_stream[i + 1])
            elif idx == i + 1:
                self.__structural_information_distances[i] = structural_information_distance(self.__graph_stream[i], anomalous_graph)
                self.__von_Neumann_distances[i] = von_Neumann_distance(self.__graph_stream[i], anomalous_graph)
                self.__veo_distances[i] = veo_score(self.__graph_stream[i], anomalous_graph)
                self.__delta_con_distances[i] = delta_con(self.__graph_stream[i], anomalous_graph)
            else:
                self.__structural_information_distances[i] = structural_information_distance(self.__graph_stream[i], self.__graph_stream[i + 1])
                self.__von_Neumann_distances[i] = von_Neumann_distance(self.__graph_stream[i], self.__graph_stream[i + 1])
                self.__veo_distances[i] = veo_score(self.__graph_stream[i], self.__graph_stream[i + 1])
                self.__delta_con_distances[i] = delta_con(self.__graph_stream[i], self.__graph_stream[i + 1])

    def __anomaly_detection(self, idx):
        anomaly_scores_si = np.zeros(self.__num_graph)
        anomaly_scores_si[0] = self.__structural_information_distances[0]
        for i in range(1, self.__num_graph - 1):
            anomaly_scores_si[i] = (self.__structural_information_distances[i] + self.__structural_information_distances[i - 1]) / 2
        anomaly_scores_si[self.__num_graph - 1] = self.__structural_information_distances[self.__num_graph - 2]
        sorted_si = sorted(range(self.__num_graph), reverse=True, key=lambda x: anomaly_scores_si[x])
        rank_si = sorted_si.index(idx)

        anomaly_scores_vn = np.zeros(self.__num_graph)
        anomaly_scores_vn[0] = self.__von_Neumann_distances[0]
        for i in range(1, self.__num_graph - 1):
            anomaly_scores_vn[i] = (self.__von_Neumann_distances[i] + self.__von_Neumann_distances[i - 1]) / 2
        anomaly_scores_vn[self.__num_graph - 1] = self.__von_Neumann_distances[self.__num_graph - 2]
        sorted_vn = sorted(range(self.__num_graph), reverse=True, key=lambda x: anomaly_scores_vn[x])
        rank_vn = sorted_vn.index(idx)

        anomaly_scores_veo = np.zeros(self.__num_graph)
        anomaly_scores_veo[0] = self.__veo_distances[0]
        for i in range(1, self.__num_graph - 1):
            anomaly_scores_veo[i] = (self.__veo_distances[i] + self.__veo_distances[i - 1]) / 2
        anomaly_scores_veo[self.__num_graph - 1] = self.__veo_distances[self.__num_graph - 2]
        sorted_veo = sorted(range(self.__num_graph), reverse=True, key=lambda x: anomaly_scores_veo[x])
        rank_veo = sorted_veo.index(idx)

        anomaly_scores_deltacon = np.zeros(self.__num_graph)
        anomaly_scores_deltacon[0] = self.__delta_con_distances[0]
        for i in range(1, self.__num_graph - 1):
            anomaly_scores_deltacon[i] = (self.__delta_con_distances[i] + self.__delta_con_distances[i - 1]) / 2
        anomaly_scores_deltacon[self.__num_graph - 1] = self.__delta_con_distances[self.__num_graph - 2]
        sorted_deltacon = sorted(range(self.__num_graph), reverse=True, key=lambda x: anomaly_scores_deltacon[x])
        rank_deltacon = sorted_deltacon.index(idx)

        logger.info(f'Structural information: ({rank_si}), von Neumann entropy: ({rank_vn}), VEO score: ({rank_veo}), deltaCon: ({rank_deltacon})')

    def __start(self):
        logger.info("=" * 60)
        logger.info(f'DDOS parameter: {self.__ddos_param}')
        logger.info(f'The number of experiments: {self.__num_experiments}')
        logger.info("=" * 60)

    def __quit(self):
        logger.info("=" * 60)
        logger.info('The END!')
        logger.info("=" * 60)
        logger.info("\n\n")

    def visualize(self):
        self.__preprocess()
        
        self.__structural_information_distances = np.zeros(self.__num_graph - 1)
        self.__von_Neumann_distances = np.zeros(self.__num_graph - 1)
        self.__veo_distances = np.zeros(self.__num_graph - 1)
        self.__delta_con_distances = np.zeros(self.__num_graph - 1)

        for i in range(self.__num_graph - 1):
            self.__structural_information_distances[i] = structural_information_distance(self.__graph_stream[i], self.__graph_stream[i + 1])
            self.__von_Neumann_distances[i] = von_Neumann_distance(self.__graph_stream[i], self.__graph_stream[i + 1])
            self.__veo_distances[i] = veo_score(self.__graph_stream[i], self.__graph_stream[i + 1])
            self.__delta_con_distances[i] = delta_con(self.__graph_stream[i], self.__graph_stream[i + 1])
        
        fig = plt.figure(figsize=(5.5, 3.5), constrained_layout=True)
        gs = fig.add_gridspec(ncols=1, nrows=1, wspace=0, hspace=0)
        ax = fig.add_subplot(gs[0, :])
        ax.plot(range(1, self.__num_graph), self.__structural_information_distances, marker='o', markerfacecolor='none', markeredgewidth=1.5, linewidth=1.5, linestyle='dashed', color='xkcd:red', label=r'$\mathcal{D}_{\rm SI}$')
        ax.plot(range(1, self.__num_graph), self.__von_Neumann_distances, marker='s', markerfacecolor='none', markeredgewidth=1.5, linewidth=1.5, linestyle='dashed', color='xkcd:black', label=r'$\mathcal{D}_{\rm QJS}$')
        ax.plot(range(1, self.__num_graph), self.__delta_con_distances, marker='^', markerfacecolor='none', markeredgewidth=1.5, linewidth=1.5, linestyle='dashed', label='deltaCon')
        ax.plot(range(1, self.__num_graph), self.__veo_distances, marker='v', markerfacecolor='none', markeredgewidth=1.5, linewidth=1.5, linestyle='dashed', label='VEO')

        ax.tick_params(labelsize=16)
        ax.set_xticks(range(1, self.__num_graph))
        X_labels = ['(1,2)','(2,3)','(3,4)','(4,5)','(5,6)','(6,7)','(7,8)','(8,9)','(9,10)']
        # ax.set_xticklabels(range(1, self.__num_graph))
        ax.set_xticklabels(X_labels)
        ax.set_xlabel('Index of Adjacent Graph Pair', fontsize=20)
        ax.set_ylabel('Distance', fontsize=20)
        ax.legend(ncol=4, fontsize=16, frameon=False, fancybox=False, edgecolor='inherit', handletextpad=0.2, borderpad=0.2, borderaxespad=0, columnspacing=0.6, labelspacing=0.2)

        plt.savefig('ddos_base.pdf')
        plt.show()

    def run(self):
        self.__preprocess()
        self.__start()

        for i in range(self.__num_experiments):
            idx = np.random.choice(self.__num_graph)
            g = self.__generate_anomalous_graph(idx)
            self.__analysis(idx, g)
            self.__anomaly_detection(idx)

        self.__quit()



class BarabasiGraphStream(object):
    """Generate synthetic graph streams according to Barabasi Albert random graph model.

    Arguments:
        n (int): the number of nodes
        m (int): the out degree of each vertex
        k (int): the number of graphs in a graph stream
    """
    def __init__(self, n=100, m=2, k=10):
        self.__graph_size = n
        self.__out_degree = m
        self.__num_graph = k

        self.__graph_stream = dict()
        self.__structural_information_distances = None
        self.__von_Neumann_distances = None
        self.__veo_distances = None
        self.__delta_con_distances = None

    def __preprocess(self):
        for i in range(self.__num_graph):
            self.__graph_stream[i] = Graph.Barabasi(self.__graph_size, self.__out_degree)

        self.__structural_information_distances = np.zeros(self.__num_graph - 1)
        self.__von_Neumann_distances = np.zeros(self.__num_graph - 1)
        self.__veo_distances = np.zeros(self.__num_graph - 1)
        self.__delta_con_distances = np.zeros(self.__num_graph - 1)
    
    def __analyze(self):
        for i in range(self.__num_graph - 1):
            self.__structural_information_distances[i] = structural_information_distance(self.__graph_stream[i], self.__graph_stream[i + 1])
            self.__von_Neumann_distances[i] = von_Neumann_distance(self.__graph_stream[i], self.__graph_stream[i + 1])
            self.__veo_distances[i] = veo_score(self.__graph_stream[i], self.__graph_stream[i + 1])
            self.__delta_con_distances[i] = delta_con(self.__graph_stream[i], self.__graph_stream[i + 1])

    def __export(self):
        for i in range(self.__num_graph):
            self.__graph_stream[i].write_gml(f'datasets/synthetic/anomaly-BA-{i}.gml')

    def __visualize(self):
        fig = plt.figure(figsize=(6, 3), constrained_layout=True)
        gs = fig.add_gridspec(ncols=1, nrows=1, wspace=0.0, hspace=0.0)
        ax = fig.add_subplot(gs[0, 0])
        ax.plot(range(1, self.__num_graph), self.__structural_information_distances, marker='o', markerfacecolor='none', markeredgewidth=1.5, linewidth=1.5, linestyle='dashed')
        ax.plot(range(1, self.__num_graph), self.__von_Neumann_distances, marker='s', markerfacecolor='none', markeredgewidth=1.5, linewidth=1.5, linestyle='dashed')
        ax.plot(range(1, self.__num_graph), self.__veo_distances, marker='v', markerfacecolor='none', markeredgewidth=1.5, linewidth=1.5, linestyle='dashed')
        ax.plot(range(1, self.__num_graph), self.__delta_con_distances, marker='^', markerfacecolor='none', markeredgewidth=1.5, linewidth=1.5, linestyle='dashed')
        ax.tick_params(labelsize=16)
        ax.set_xlabel('Index', fontsize=20)
        ax.set_ylabel('Distance', fontsize=20)
        ax.legend(fontsize=16, ncol=4, frameon=False, fancybox=False, edgecolor='inherit', handletextpad=0.2, borderpad=0.2, borderaxespad=0, columnspacing=0.6)

        plt.show()
    
    def run(self):
        self.__preprocess()
        self.__analyze()
        self.__visualize()
        self.__export()