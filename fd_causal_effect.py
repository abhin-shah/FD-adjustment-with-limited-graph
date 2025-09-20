"""
Reference: A Shah, K Shanmugam, M Kocaoglu
"Front-door Adjustment Beyond Markov Equivalence with Limited Graph Knowledge,"
In 37th Conference on Neural Information Processing Systems (NeurIPS), 2023

Last updated: December 15, 2023
Code author: Abhin Shah

File name: fd_causal_effect.py

Description: Estimate average treatment effects for Figure 5. Compares proposed
front-door adjustment method against baseline across different sample sizes.
Uses synthetic data generated from valid random graphs.
"""

import time
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from sklearn.model_selection import train_test_split

from supporting_func import get_data, get_effect, worker, worker_zizo, get_errors

from causallearn.graph.Dag import Dag
from causallearn.utils.DAG2PAG import dag2pag
from causallearn.graph.GraphNode import GraphNode
from causallearn.utils.GraphUtils import GraphUtils

def get_dag(number_observed, number_confounders, A):
    """Convert adjacency matrix to DAG visualization format."""
    dag_nodes = []
    for node_name in range(number_observed + number_confounders):
        dag_nodes.append(GraphNode(node_name))
    dag = Dag(dag_nodes)
    for i in range(number_observed + number_confounders):
        for j in range(number_observed + number_confounders):
            if A[i,j] == 1:
                dag.add_directed_edge(dag_nodes[i],dag_nodes[j])
    return GraphUtils.to_pydot(dag)

def save_dag_visualizations(output_path, number_observed_list, number_iterations, all_t, all_y, all_b, all_num_con, all_A, ours):
    """Save DAG visualizations as PNG files for valid graphs."""
    for n_count, number_observed in enumerate(number_observed_list):
        store = 0
        for iter in range(number_iterations):
            if ours[n_count, iter] == True:
                store += 1
                print("Generating DAG visualization for Graph = " + str(iter) +  " out of " + str(int(np.sum(ours, axis = 1)[0])))
                t = int(all_t[n_count, iter])
                y = int(all_y[n_count, iter])
                b = all_b[n_count][iter]
                number_confounders = int(all_num_con[n_count, iter])
                A_total = all_A[n_count][iter]
                print(f"t={t}, y={y}, b={b}, number_confounders={number_confounders}")
                pyd = get_dag(number_observed, number_confounders, A_total)
                pyd.write_png(output_path + '_dag_' + str(store) + '.png')

def plot_ate_results(output_path, samples_list, number_observed_list, mean_base_ate_error, mean_zozi_ate_error, mean_bzi_ate_error, std_base_ate_error, std_zozi_ate_error, std_bzi_ate_error):
    """Generate Figure 5: ATE vs Sample Size plot."""
    for n_count, number_observed in enumerate(number_observed_list):
        print("Generating Figure 5 for Number observed = " + str(number_observed))
        
        current_mean_base_ate_error = mean_base_ate_error[n_count]
        current_mean_zozi_ate_error = mean_zozi_ate_error[n_count]
        current_mean_bzi_ate_error = mean_bzi_ate_error[n_count]
        
        current_std_base_ate_error = std_base_ate_error[n_count]
        current_std_zozi_ate_error = std_zozi_ate_error[n_count]
        current_std_bzi_ate_error = std_bzi_ate_error[n_count]
        
        samples_base_ate_error = np.nanmean(np.array(current_mean_base_ate_error), axis = 0)
        samples_zozi_ate_error = np.nanmean(np.array(current_mean_zozi_ate_error), axis = 0)
        samples_bzi_ate_error = np.nanmean(np.array(current_mean_bzi_ate_error), axis = 0)
        
        std_samples_base_ate_error = np.nanmean(np.array(current_std_base_ate_error), axis = 0)
        std_samples_zozi_ate_error = np.nanmean(np.array(current_std_zozi_ate_error), axis = 0)
        std_samples_bzi_ate_error = np.nanmean(np.array(current_std_bzi_ate_error), axis = 0)
        
        # Configure matplotlib for publication quality
        matplotlib.rcParams['ps.useafm'] = True
        matplotlib.rcParams['pdf.use14corefonts'] = True
        matplotlib.rcParams['text.usetex'] = True
        params = {'mathtext.default': 'regular'}          
        plt.rcParams.update(params)
        
        # Create the plot
        plt.figure(figsize = (10,6))
        plt.errorbar(samples_list, samples_base_ate_error, yerr = std_samples_base_ate_error, label = 'Baseline')
        plt.errorbar(samples_list, samples_zozi_ate_error, yerr = std_samples_zozi_ate_error, label = 'Exhaustive$_z$')
        plt.errorbar(samples_list, samples_bzi_ate_error, yerr = std_samples_bzi_ate_error, label = 'Exhaustive$_s$')
        plt.xscale('log')
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.tick_params(axis='both', which='major', labelsize = 27)
        plt.xlabel(r'Number of samples',fontsize = 27)
        plt.ylabel("Average ATE error",fontsize = 27)
        plt.grid(True, alpha=0.4)
        plt.yticks([0.00, 0.08, 0.16, 0.24, 0.32])
        plt.legend(loc='center right', handletextpad=0.5,fontsize = 24)
        plt.savefig(output_path + str(samples_list) + '_ATE_vs_samples.pdf',bbox_inches='tight')
        plt.close()
        
        print("Figure 5 saved as: " + output_path + str(samples_list) + '_ATE_vs_samples.pdf')

def main(args):
    """Estimate ATE errors for valid graphs across different sample sizes and p-value thresholds."""
    number_iterations = args.niter
    number_observed_list = args.nobs
    degree = args.d
    all_samples = args.all_samples
    number_repetitions = args.nr
    samples_list = args.samples_list
    p_thresholds = [0.1,0.3,0.5]
    qq = args.qq
    table = args.table  # 1 for Table 1 (strict), 2 for Table 2 (relaxed)
    l = 0
    
    output_path = 'table' + str(table) + '_iter' + str(number_iterations) + '_q' + str(qq) + '/' + str(number_observed_list) + '_d_' + str(degree)
    
    all_t = np.load(output_path + '_t.npy', allow_pickle=True)
    all_y = np.load(output_path + '_y.npy', allow_pickle=True)
    all_b = np.load(output_path + '_b.bin', allow_pickle=True)
    all_num_con = np.load(output_path + '_num_con.npy', allow_pickle=True)
    all_A = np.load(output_path + '_A.bin', allow_pickle=True)
    ours = np.load(output_path + '_ours.npy', allow_pickle=True)

    flag = True
    starting_time = time.perf_counter()

    all_base_ate_error = []
    all_zozi_ate_error = []
    all_bzi_ate_error = []

    mean_base_ate_error = []
    mean_zozi_ate_error = []
    mean_bzi_ate_error = []

    std_base_ate_error = []
    std_zozi_ate_error = []
    std_bzi_ate_error = []

    for n_count, number_observed in enumerate(number_observed_list):
        print("Number observed = " + str(number_observed))

        current_base_ate_error = []
        current_zozi_ate_error = []
        current_bzi_ate_error = []

        current_mean_base_ate_error = []
        current_mean_zozi_ate_error = []
        current_mean_bzi_ate_error = []

        current_std_base_ate_error = []
        current_std_zozi_ate_error = []
        current_std_bzi_ate_error = []

        for iter in range(number_iterations):
            if ours[n_count, iter] == True:
                print("Graph = " + str(iter) +  " out of " + str(int(np.sum(ours, axis = 1)[0])))
                t = int(all_t[n_count, iter])
                y = int(all_y[n_count, iter])
                b = all_b[n_count][iter]
                number_confounders = int(all_num_con[n_count, iter])
                A_total = all_A[n_count][iter]
                print(t,y,b,number_confounders)

                base_ate_error = np.zeros((number_repetitions,len(samples_list)))
                zozi_ate_error = np.zeros((len(p_thresholds),number_repetitions,len(samples_list)))
                bzi_ate_error = np.zeros((len(p_thresholds),number_repetitions,len(samples_list)))

                for r_iter in range(number_repetitions):
                    print("Repetition = " + str(r_iter+1) +  " out of " + str(number_repetitions))
                    data, data_t, data_y, data_b, data_x, true_ate = get_data(number_observed, number_confounders, all_samples, A_total, t, y, b)
                    c_train_all, c_test_all, t_train_all, t_test_all, b_train_all, b_test_all, y_train_all, y_test_all = train_test_split(data_x, data_t, data_b, data_y, test_size=0.2)
                    for d_iter, samples in enumerate(samples_list):
                        print("Number samples = " + str(samples) + "for repetition = " + str(r_iter+1))
                        base_error, zozi_error, bzi_error = get_errors(p_thresholds, c_train_all, c_test_all, t_train_all, t_test_all, b_train_all, b_test_all, y_train_all, y_test_all, samples, true_ate)
                        base_ate_error[r_iter, d_iter] = base_error
                        zozi_ate_error[:, r_iter, d_iter] = zozi_error
                        bzi_ate_error[:, r_iter, d_iter] = bzi_error
                        print('Run time so far (samples): ' + str(time.perf_counter() - starting_time))
                    
                    print('Run time so far (reps): ' + str(time.perf_counter() - starting_time))

                current_base_ate_error.append(base_ate_error)
                current_zozi_ate_error.append(zozi_ate_error)
                current_bzi_ate_error.append(bzi_ate_error)
                
                current_mean_base_ate_error.append((np.mean(base_ate_error, axis=0)))
                current_mean_zozi_ate_error.append((np.nanmean(zozi_ate_error, axis=1))[l,:])
                current_mean_bzi_ate_error.append((np.nanmean(bzi_ate_error, axis=1))[l,:])            

                current_std_base_ate_error.append((np.std(base_ate_error, axis=0))/np.sqrt(number_repetitions))
                current_std_zozi_ate_error.append((np.nanstd(zozi_ate_error, axis=1)/np.sqrt(np.count_nonzero(~np.isnan(zozi_ate_error), axis=1)))[l,:])
                current_std_bzi_ate_error.append((np.nanstd(bzi_ate_error, axis=1)/np.sqrt(np.count_nonzero(~np.isnan(bzi_ate_error), axis=1)))[l,:])
                
                print('Run time so far (graphs): ' + str(time.perf_counter() - starting_time))

        all_base_ate_error.append(current_base_ate_error)
        all_zozi_ate_error.append(current_zozi_ate_error)
        all_bzi_ate_error.append(current_bzi_ate_error)

        mean_base_ate_error.append(current_mean_base_ate_error)
        mean_zozi_ate_error.append(current_mean_zozi_ate_error)
        mean_bzi_ate_error.append(current_mean_bzi_ate_error)

        std_base_ate_error.append(current_std_base_ate_error)
        std_zozi_ate_error.append(current_std_zozi_ate_error)
        std_bzi_ate_error.append(current_std_bzi_ate_error)

    print('------------------------------------------------------------')
    print('Run time to get causal effects: ' + str(time.perf_counter() - starting_time))
    print('------------------------------------------------------------')
    np.save(output_path + str(samples_list) + '_time_effect.npy', np.array(time.perf_counter() - starting_time))
    
    print(mean_base_ate_error)
    print(mean_zozi_ate_error)
    print(mean_bzi_ate_error)
    
    # Generate DAG visualizations for valid graphs
    save_dag_visualizations(output_path, number_observed_list, number_iterations, all_t, all_y, all_b, all_num_con, all_A, ours)
    
    # Generate Figure 5: ATE vs Sample Size plot
    plot_ate_results(output_path, samples_list, number_observed_list, mean_base_ate_error, mean_zozi_ate_error, mean_bzi_ate_error, std_base_ate_error, std_zozi_ate_error, std_bzi_ate_error)
    
    with open(output_path + str(samples_list) + '_all_base.bin', 'wb') as f:
        pickle.dump(all_base_ate_error, f)
    with open(output_path + str(samples_list) + '_all_zozi.bin', 'wb') as f:
        pickle.dump(all_zozi_ate_error, f)
    with open(output_path + str(samples_list) + '_all_bzi.bin', 'wb') as f:
        pickle.dump(all_bzi_ate_error, f)
    with open(output_path + str(samples_list) + '_mean_base.bin', 'wb') as f:
        pickle.dump(mean_base_ate_error, f)
    with open(output_path + str(samples_list) + '_mean_zozi.bin', 'wb') as f:
        pickle.dump(mean_zozi_ate_error, f)
    with open(output_path + str(samples_list) + '_mean_bzi.bin', 'wb') as f:
        pickle.dump(mean_bzi_ate_error, f)
    with open(output_path + str(samples_list) + '_std_base.bin', 'wb') as f:
        pickle.dump(std_base_ate_error, f)
    with open(output_path + str(samples_list) + '_std_zozi.bin', 'wb') as f:
        pickle.dump(std_zozi_ate_error, f)
    with open(output_path + str(samples_list) + '_std_bzi.bin', 'wb') as f:
        pickle.dump(std_bzi_ate_error, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--niter', help='number_iterations', default=100, type=int)
    parser.add_argument('--nobs', action='store', nargs='*', dest='nobs', help='number_observed_list', type=int)
    parser.add_argument('--d', help='degree', default=2, type=int)
    parser.add_argument('--qq', help='qq', default=1.0, type=float)
    parser.add_argument('--nr', help='number_repetitions', default=10, type=int)
    parser.add_argument('--all_samples', help='all_samples', default=100000, type=int)
    parser.add_argument('--samples_list', action='store', nargs='*', dest='samples_list', help='samples_list', type=int)
    parser.add_argument('--table', help='table number (1 or 2)', default=1, type=int)
    args = parser.parse_args()
    main(args)