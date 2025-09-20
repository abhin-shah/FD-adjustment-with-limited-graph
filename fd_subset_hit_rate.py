"""
Reference: A Shah, K Shanmugam, M Kocaoglu
"Front-door Adjustment Beyond Markov Equivalence with Limited Graph Knowledge,"
In 37th Conference on Neural Information Processing Systems (NeurIPS), 2023

Last updated: December 15, 2023
Code author: Abhin Shah

File name: fd_subset_hit_rate.py

Description: Compute hit rates for front-door adjustment using bounded search.
Tests proposed method from Theorem 3.2 on random graphs for Table 1.
Searches subsets z with cardinality ≤ K for computational efficiency.
"""

import sys
import time
import pickle
import argparse
import numpy as np
import pandas as pd
import networkx as nx

from supporting_func import degree_bounded_set, check_additional_CI

def main(args):
    """Test bounded search method for finding valid front-door adjustments on random graphs."""
    number_iterations = args.niter
    number_observed_list = args.nobs
    degree = args.d
    K = args.K
    qq = args.qq
    table = args.table  # 1 for Table 1 (strict), 2 for Table 2 (relaxed)
    
    output_path = 'table' + str(table) + '_iter' + str(number_iterations) + '_q' + str(qq) + '/' + str(number_observed_list) + '_d_' + str(degree)
    
    all_t = np.load(output_path + '_t.npy', allow_pickle=True)
    all_y = np.load(output_path + '_y.npy', allow_pickle=True)
    all_b = np.load(output_path + '_b.bin', allow_pickle=True)
    all_A = np.load(output_path + '_A.bin', allow_pickle=True)

    ours_subset = np.zeros((len(number_observed_list),number_iterations))
    starting_time = time.perf_counter()
    for n_count, number_observed in enumerate(number_observed_list):
        print(n_count)
        for iter in range(number_iterations):
            if (iter + 1) % 5 == 0:
                print(iter + 1, number_observed)
            t = all_t[n_count, iter]
            y = all_y[n_count, iter]
            b = all_b[n_count][iter]
            A_total = all_A[n_count][iter]
            G_total = nx.DiGraph(incoming_graph_data=A_total)

            possible_z = set(list(np.arange(number_observed))) - {t} - set(b) - {y}
            for candidate_z in degree_bounded_set(possible_z, K):
                if nx.d_separated(G_total, {y}, set(b), set(candidate_z + (t,))):
                    parity, candidate_zo, candidate_zi = check_additional_CI(G_total, t, b, candidate_z)
                    if parity:
                        ours_subset[n_count, iter] = 1
                        print("Candidate zo")
                        print(candidate_zo)
                        print("Candidate zi")
                        print(candidate_zi)
                        break

    print('------------------------------------------------------------')
    print('Run time for our subset method: ' + str(time.perf_counter() - starting_time))
    print(np.sum(ours_subset, axis = 1)/number_iterations)
    print('------------------------------------------------------------')
        
    with open(output_path + '_ours.txt', 'a') as sys.stdout:
        print("Our subset method with d = " + str(degree) + " for Table " + str(table))
        print(np.sum(ours_subset, axis = 1)/number_iterations)
        
    np.save(output_path + '_ours_subset.npy', ours_subset)
    np.save(output_path + '_time_ours_subset.npy', np.array(time.perf_counter() - starting_time))
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--niter', help='number_iterations', default=100, type=int)
    parser.add_argument('--nobs', action='store', nargs='*', dest='nobs', help='number_observed_list', type=int)
    parser.add_argument('--d', help='degree', default=2, type=int)
    parser.add_argument('--K', help='maximum subset size', default=5, type=int)
    parser.add_argument('--qq', help='qq', default=0.1, type=float)
    parser.add_argument('--table', help='table number (1 or 2)', default=1, type=int)
    args = parser.parse_args()
    main(args)