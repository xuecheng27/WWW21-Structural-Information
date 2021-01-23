import numpy as np 

from Q12_Universality_and_Sensitivity.erdos_renyi import ErdosRenyiTest
from Q12_Universality_and_Sensitivity.barabasi_albert import BarabasiAlbertTest
from Q12_Universality_and_Sensitivity.watts_strogatz import WattsStrogatzTest

if __name__ == "__main__":
    average_degree_list = [6, 10, 20, 50]
    for k in range(10):
        for avg_deg in average_degree_list:
            t3 = WattsStrogatzTest(2000, avg_deg, np.linspace(0, 1, 21))
            t3.run()
    for k in range(10):
        t2 = BarabasiAlbertTest(2000, np.linspace(1, 100, 100, dtype=int))
        t2.run()
    for k in range(10):
        t1 = ErdosRenyiTest(2000, np.linspace(0.001, 0.1, 100))
        t1.run()