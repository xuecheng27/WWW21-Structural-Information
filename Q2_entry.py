import numpy as np 

from Q12_Universality_and_Sensitivity.erdos_renyi import ErdosRenyiTest
from Q12_Universality_and_Sensitivity.barabasi_albert import BarabasiAlbertTest
from Q12_Universality_and_Sensitivity.watts_strogatz import WattsStrogatzTest


if __name__ == "__main__":
    n_range = np.linspace(500, 5000, 10, dtype=int)
    for n in n_range:
        for k in range(10):
            t = WattsStrogatzTest(n, 20, [0, 0.1, 0.2, 0.4, 0.8, 1])
            t.run()
    
    n_range = np.linspace(500, 5000, 10, dtype=int)
    d_range = np.array([2, 6, 10, 20, 50, 100])
    m_range = (d_range / 2).astype(int)
    for n in n_range:
        for k in range(10):
            t = BarabasiAlbertTest(n, m_range)
            t.run()

    n_range = np.linspace(500, 5000, 10, dtype=int)
    d_range = np.array([2, 5, 10, 20, 50, 100])
    for n in n_range:
        p_range = d_range / (n - 1)
        for k in range(10):
            t = ErdosRenyiTest(n, p_range)
            t.run()