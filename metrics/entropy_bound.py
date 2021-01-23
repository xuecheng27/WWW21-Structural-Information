from math import log
from math import exp
import numpy as np 

def __func_f(x):
    epsilon = 10 ** (-8)
    if x <= epsilon:
        return 0
    else:
        return x * log(x, 2)

def sharpened_lower_bound(graph):
    m = graph.ecount()
    if m == 0:
        lower_bound = 0
    else:
        degree_seq = graph.vs.degree()
        if 0 in degree_seq:
            degree_seq.remove(0)
        max_degree = max(degree_seq)
        min_degree = min(degree_seq)
        lower_bound = __func_f(max_degree + 1) - __func_f(max_degree) + __func_f(min_degree - 1) - __func_f(min_degree)
        lower_bound /= 2 * m
    
    return lower_bound

def sharpened_upper_bound(graph):
    m = graph.ecount()
    n = graph.vcount()
    if m == 0:
        upper_bound = 0
    else:
        degree_seq = graph.vs.degree()
        if 0 in degree_seq:
            degree_seq.remove(0)

        sum_degree_log_degree = 0
        sum_square_degree = 0
        conjugate_degree_seq = np.zeros(n, dtype=int)
        for deg in degree_seq:
            sum_degree_log_degree += __func_f(deg)
            sum_square_degree += deg * deg
            conjugate_degree_seq[:deg] += 1

        conjugate_sum_degree_log_degree = 0
        for conj_deg in conjugate_degree_seq:
            conjugate_sum_degree_log_degree += __func_f(conj_deg)

        upper_bound = min(log(exp(1), 2), (conjugate_sum_degree_log_degree - sum_degree_log_degree) / (2 * m), log(1 + sum_square_degree / (2 * m), 2) - sum_degree_log_degree / (2 * m))

    return upper_bound