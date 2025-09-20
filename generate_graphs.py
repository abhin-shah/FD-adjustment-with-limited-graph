"""
Reference: A Shah, K Shanmugam, M Kocaoglu
"Front-door Adjustment Beyond Markov Equivalence with Limited Graph Knowledge,"
In 37th Conference on Neural Information Processing Systems (NeurIPS), 2023

Last updated: December 15, 2023
Code author: Abhin Shah

File name: generate_graphs.py

Description: Generate random semi-Markovian causal models (SMCMs) and corresponding 
PAGs for Tables 1 and 2. Creates random graphs with specified parameters and saves
the treatment, outcome, children, and adjacency matrices.
"""

import time
import pickle
import argparse
import numpy as np
import pandas as pd
import networkx as nx

from itertools import chain, combinations

from causallearn.graph.Dag import Dag
from causallearn.utils.DAG2PAG import dag2pag
from causallearn.graph.GraphNode import GraphNode
from causallearn.utils.GraphUtils import GraphUtils

from supporting_func import get_ER_adjacency_matrix, get_possible_t, get_easy_possible_t, get_tby, get_pag_adj, get_pag_R

def main(args):
    """Generate random SMCMs and PAGs for experimental evaluation of front-door criteria."""
    number_iterations = args.niter
    number_observed_list = args.nobs
    degree = args.d
    qq = args.qq
    table = args.table  # 1 for Table 1 (strict), 2 for Table 2 (relaxed)
    
    all_t = np.zeros((len(number_observed_list),number_iterations))
    all_y = np.zeros((len(number_observed_list),number_iterations))
    all_num_con = np.zeros((len(number_observed_list),number_iterations))
    all_b = []
    all_A = []
    all_A_pag = []
    starting_time = time.perf_counter()
    for n_count, number_observed in enumerate(number_observed_list):
        print(n_count, number_observed)
        all_current_b = []
        all_current_A = []
        all_current_A_pag = []
        for iter in range(number_iterations):
            if (iter + 1) % 5 == 0:
                print(iter + 1, number_observed)
            possible_t = []

            while len(possible_t) == 0:
                A_observed = get_ER_adjacency_matrix(number_observed,degree)
                G_dummy = nx.Graph(incoming_graph_data=A_observed)
                while nx.is_connected(G_dummy) == False:
                    A_observed = get_ER_adjacency_matrix(number_observed,degree)
                    G_dummy = nx.Graph(incoming_graph_data=A_observed)
                G_directed = nx.DiGraph(incoming_graph_data=A_observed)
                # Table 1: ancestor but not parent/grandparent, Table 2: ancestor but not parent  
                if table == 1:
                    possible_t = get_possible_t(G_directed)  # Strict: excludes parent and grandparent
                else:
                    possible_t = get_easy_possible_t(G_directed)  # Relaxed: excludes only parent
            
            t,b,y = get_tby(A_observed, possible_t)

            q = qq
            G_confounding = nx.erdos_renyi_graph(number_observed, q)
            A_confounding = nx.adjacency_matrix(G_confounding)
            if A_confounding[t,y] != 1:
                A_confounding[t,y] = 1
                A_confounding[y,t] = 1
            G_confounding = nx.Graph(incoming_graph_data=A_confounding)
            number_confounders = G_confounding.number_of_edges()
            
            A_total = np.zeros((number_observed + number_confounders, number_observed + number_confounders))
            A_total[:number_observed, :number_observed] = A_observed
            for count, edge in enumerate(G_confounding.edges()):
                A_total[count + number_observed, edge[0]] = 1
                A_total[count + number_observed, edge[1]] = 1
            A_pag = get_pag_adj(number_observed, number_confounders, A_total)
            A_pag_R = get_pag_R(A_pag)

            all_t[n_count, iter] = t
            all_y[n_count, iter] = y
            all_num_con[n_count, iter] = number_confounders
            all_current_b.append(b)
            all_current_A.append(A_total)
            all_current_A_pag.append(A_pag_R)
        all_b.append(all_current_b)
        all_A.append(all_current_A)
        all_A_pag.append(all_current_A_pag) 
        
    print('------------------------------------------------------------')
    print('Run time to create DAGs and PAGs: ' + str(time.perf_counter() - starting_time))
    print('------------------------------------------------------------')
        
    output_path = 'table' + str(table) + '_iter' + str(number_iterations) + '_q' + str(qq) + '/' + str(number_observed_list) + '_d_' + str(degree)
    np.save(output_path + '_t.npy', all_t)
    np.save(output_path + '_y.npy', all_y)
    np.save(output_path + '_num_con.npy', all_num_con)
    with open(output_path + '_b.bin', 'wb') as f:
        pickle.dump(all_b, f)
    with open(output_path + '_A.bin', 'wb') as f:
        pickle.dump(all_A, f)
    with open(output_path + '_A_pag.bin', 'wb') as f:
        pickle.dump(all_A_pag, f)
    np.save(output_path + '_time_graph.npy', np.array(time.perf_counter() - starting_time))
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--niter', help='number_iterations', default=100, type=int)
    parser.add_argument('--nobs', action='store', nargs='*', dest='nobs', help='number_observed_list', type=int)
    parser.add_argument('--d', help='degree', default=2, type=int)
    parser.add_argument('--qq', help='qq', default=0.1, type=float)
    parser.add_argument('--table', help='table number (1 or 2)', default=1, type=int)
    args = parser.parse_args()
    main(args)