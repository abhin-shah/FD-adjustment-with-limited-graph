"""
Reference: A Shah, K Shanmugam, M Kocaoglu
"Front-door Adjustment Beyond Markov Equivalence with Limited Graph Knowledge,"
In 37th Conference on Neural Information Processing Systems (NeurIPS), 2023

Last updated: December 15, 2023
Code author: Abhin Shah

File name: synthetic.py

Description: Generate synthetic data for specific graph structures (Figure 8) and 
test ATE estimation performance for Figure 9. Creates data for modified versions
of G_toy graph and evaluates conditional independence criteria.
"""

import time
import pickle
import argparse
import multiprocessing
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib as mpl
from functools import partial
from sklearn.preprocessing import StandardScaler
from scipy.special import softmax, expit
from contextlib import contextmanager
from multiprocessing import get_context
from multiprocessing import set_start_method
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV

import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
from rpy2.robjects.packages import importr
importr('RCIT')
RCoT = ro.r('RCoT')

def plot_synthetic_results(number_repetitions, graph, number_dimensions_list):
    """Generate bar plots for Figure 9 comparing ATE estimation errors across different p-value thresholds."""
    # Configure matplotlib for publication quality
    matplotlib.rcParams['ps.useafm'] = True
    matplotlib.rcParams['pdf.use14corefonts'] = True
    matplotlib.rcParams['text.usetex'] = True
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.spines.top'] = False
    
    # Load the error data
    with open('synthetic_' + str(graph) + '/2_z0zi_ate_error.pkl', 'rb') as pkl_file:
        z0zi_ate_error = pickle.load(pkl_file)
    with open('synthetic_' + str(graph) + '/2_bzi_ate_error.pkl', 'rb') as pkl_file:
        bzi_ate_error = pickle.load(pkl_file)
    with open('synthetic_' + str(graph) + '/2_baseline_ate_error.pkl', 'rb') as pkl_file:
        baseline_ate_error = pickle.load(pkl_file)
    
    # Prepare data for plotting
    p_thresholds = ['-0.05','-0.1','-0.15','-0.2','-0.25']
    avg_ate_error = pd.DataFrame()
    avg_ate_error['method'] = pd.DataFrame(['b'] + len(p_thresholds)*['z'] + len(p_thresholds)*['f'])
    avg_ate_error['p-value'] = pd.DataFrame(['-'] + p_thresholds + p_thresholds)
    avg_ate_error['error'] = pd.DataFrame(np.concatenate(((np.mean(baseline_ate_error, axis=0)).reshape(1,-1),(np.nanmean(z0zi_ate_error, axis=1)),(np.nanmean(bzi_ate_error, axis=1))), axis = 0))
    avg_ate_error['std'] = pd.DataFrame(np.concatenate(((np.std(baseline_ate_error, axis=0)).reshape(1,-1)/np.sqrt(number_repetitions),(np.nanstd(z0zi_ate_error, axis=1))/np.sqrt(np.count_nonzero(~np.isnan(z0zi_ate_error), axis=1)),(np.nanstd(bzi_ate_error, axis=1))/np.sqrt(np.count_nonzero(~np.isnan(bzi_ate_error), axis=1)))))

    # Create the plot
    params = {'mathtext.default': 'regular'}          
    plt.rcParams.update(params)
    plt.figure(figsize=(9, 6))
    plt.bar(data=avg_ate_error.loc[avg_ate_error['method']=='b'], x=avg_ate_error.loc[avg_ate_error['method']=='b'].index, height='error', label='b', yerr='std')
    plt.bar(data=avg_ate_error.loc[avg_ate_error['method']=='z'], x=avg_ate_error.loc[avg_ate_error['method']=='z'].index+0.5, height='error', label='z', width=0.9, yerr='std')
    plt.bar(data=avg_ate_error.loc[avg_ate_error['method']=='f'], x=avg_ate_error.loc[avg_ate_error['method']=='f'].index+1, height='error', label='f', width=0.9, yerr='std')
    plt.tick_params(axis='both', which='major', labelsize = 27)
    plt.ylabel("average ATE error",fontsize=27)
    plt.xticks(avg_ate_error.loc[avg_ate_error['method']=='b'].index.tolist() + list(avg_ate_error.loc[avg_ate_error['method']=='z'].index+0.5) + list(avg_ate_error.loc[avg_ate_error['method']=='f'].index+1), 
               ['baseline','0.05','0.1','0.15\nExhaustive$_z$','0.2','0.25','0.05','0.1','0.15\nExhaustive$_s$','0.2','0.25'],fontsize=20)
    plt.grid(True, alpha=0.4)
    plt.savefig('synthetic_' + str(graph) + '/d' + str(number_dimensions_list[-1]) + 'g' + str(graph) + '.pdf')
    plt.close()
    print("Figure 9 saved as: " + 'synthetic_' + str(graph) + '/d' + str(number_dimensions_list[-1]) + 'g' + str(graph) + '.pdf')

@contextmanager
def poolcontext(*args, **kwargs):
    """Context manager for multiprocessing pool to ensure proper cleanup."""
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()
    
def get_u(number_observations, number_dimensions):
    """Generate unobserved confounders u1, u2, u3, u4 for synthetic data."""
    u1 = np.random.uniform(1.0,2.0,(number_observations,number_dimensions))
    u2 = np.random.uniform(1.0,2.0,(number_observations,number_dimensions))
    u3 = np.random.uniform(1.0,2.0,(number_observations,number_dimensions))
    u4 = np.random.uniform(1.0,2.0,(number_observations,number_dimensions))
    return u1, u2, u3, u4

def get_data(u1,u2,u3,u4,number_observations,number_dimensions,graph):
    """Generate synthetic data for specified graph structure (1-4 corresponding to Figure 8)."""
    theta1 = np.random.uniform(1.0,2.0,(number_dimensions,number_dimensions)) # for x1
    x1 = np.dot(u1, theta1) + 0.1* np.random.normal(0,1,(number_observations,number_dimensions))
    x1 = x1/theta1.shape[0]
    
    theta2 = np.random.uniform(1.0,2.0,(2*number_dimensions,1)) # for t
    t = np.random.binomial(1, expit(np.dot(np.concatenate((u2, x1), axis = 1), theta2)-np.mean(np.dot(np.concatenate((u2, x1), axis = 1), theta2))))
    zeros =  np.zeros((number_observations,1))
    ones =  np.ones((number_observations,1))
    
    theta3 = np.random.uniform(1.0,2.0,(number_dimensions,number_dimensions)) # for zi
    zi = np.dot(u3, theta3) + 0.1* np.random.normal(0,1,(number_observations,number_dimensions))
    zi = zi/theta3.shape[0] 
    
    
    if graph == 1 or graph == 4:
        theta4 = np.random.uniform(1.0,2.0,(3*number_dimensions,number_dimensions)) # for x2
        x2 = np.dot(np.concatenate((u1,u2,u4),axis=1), theta4) + 0.1* np.random.normal(0,1,(number_observations,number_dimensions))
    elif graph == 2:
        theta4 = np.random.uniform(1.0,2.0,(4*number_dimensions,number_dimensions)) # for x2
        x2 = np.dot(np.concatenate((u1,u2,u4,x1),axis=1), theta4) + 0.1* np.random.normal(0,1,(number_observations,number_dimensions))
    elif graph == 3:
        theta4 = np.random.uniform(1.0,2.0,(4*number_dimensions,number_dimensions)) # for x2
        x2 = np.dot(np.concatenate((u1,u2,u4,zi),axis=1), theta4) + 0.1* np.random.normal(0,1,(number_observations,number_dimensions))
    x2 = x2/theta4.shape[0]
        
    
    noise_b = 0.1* np.random.normal(0,1,(number_observations,1))
    if graph == 1 or graph == 2 or graph == 4:
        theta5 = np.random.uniform(1.0,2.0,(2*number_dimensions+1,1)) # for b
        b = np.dot(np.concatenate((u4,t,zi),axis=1), theta5) + noise_b
        b_0 = np.dot(np.concatenate((u4,zeros,zi),axis=1), theta5) + noise_b
        b_1 = np.dot(np.concatenate((u4,ones,zi),axis=1), theta5) + noise_b
    elif graph == 3:
        theta5 = np.random.uniform(1.0,2.0,(3*number_dimensions+1,1)) # for b
        b = np.dot(np.concatenate((u4,t,zi,u3),axis=1), theta5) + noise_b
        b_0 = np.dot(np.concatenate((u4,zeros,zi,u3),axis=1), theta5) + noise_b
        b_1 = np.dot(np.concatenate((u4,ones,zi,u3),axis=1), theta5) + noise_b
    b = b/theta5.shape[0]
    b_0 = b_0/theta5.shape[0]
    b_1 = b_1/theta5.shape[0]
        
    
    noise_zo = 0.1* np.random.normal(0,1,(number_observations,number_dimensions))
    if graph == 1:
        theta6 = np.random.uniform(1.0,2.0,(1,number_dimensions)) # for zo
        zo = np.dot(b, theta6) + noise_zo
        zo_0 = np.dot(b_0, theta6) + noise_zo
        zo_1 = np.dot(b_1, theta6) + noise_zo
    elif graph == 2 or graph == 3:
        theta6 = np.random.uniform(1.0,2.0,(1+number_dimensions,number_dimensions)) # for zo
        zo = np.dot(np.concatenate((b,zi),axis=1), theta6) + noise_zo
        zo_0 = np.dot(np.concatenate((b_0,zi),axis=1), theta6) + noise_zo
        zo_1 = np.dot(np.concatenate((b_1,zi),axis=1), theta6) + noise_zo
    elif graph == 4:
        theta6 = np.random.uniform(1.0,2.0,(1+2*number_dimensions,number_dimensions)) # for zo
        zo = np.dot(np.concatenate((b,zi,u3),axis=1), theta6) + noise_zo
        zo_0 = np.dot(np.concatenate((b_0,zi,u3),axis=1), theta6) + noise_zo
        zo_1 = np.dot(np.concatenate((b_1,zi,u3),axis=1), theta6) + noise_zo
    zo = zo/theta6.shape[0]
    zo_0 = zo_0/theta6.shape[0]
    zo_1 = zo_1/theta6.shape[0]
    
    noise_y = 0.1* np.random.normal(0,1,(number_observations,1))
    if graph == 1 or graph == 3:
        theta7 = np.random.uniform(1.0,2.0,(3*number_dimensions,1)) # for y
        y = np.dot(np.concatenate((u2,zo,zi),axis=1),theta7) + noise_y 
        y_0 = np.dot(np.concatenate((u2,zo_0,zi),axis=1),theta7) + noise_y
        y_1 = np.dot(np.concatenate((u2,zo_1,zi),axis=1),theta7) + noise_y
    elif graph == 2:
        theta7 = np.random.uniform(1.0,2.0,(4*number_dimensions,1)) # for y
        y = np.dot(np.concatenate((u2,zo,zi,u3),axis=1),theta7) + noise_y 
        y_0 = np.dot(np.concatenate((u2,zo_0,zi,u3),axis=1),theta7) + noise_y
        y_1 = np.dot(np.concatenate((u2,zo_1,zi,u3),axis=1),theta7) + noise_y
    elif graph == 4:
        theta7 = np.random.uniform(1.0,2.0,(4*number_dimensions,1)) # for y
        y = np.dot(np.concatenate((u2,zo,zi,x1),axis=1),theta7) + noise_y 
        y_0 = np.dot(np.concatenate((u2,zo_0,zi,x1),axis=1),theta7) + noise_y
        y_1 = np.dot(np.concatenate((u2,zo_1,zi,x1),axis=1),theta7) + noise_y
    y = y/theta7.shape[0]
    y_0 = y_0/theta7.shape[0]
    y_1 = y_1/theta7.shape[0]
    
    return x1,x2,t,b,zo,zi,y,y_0,y_1


def get_train_test_data(number_observations,number_dimensions,graph):
    u1,u2,u3,u4 = get_u(number_observations,number_dimensions)
    x1,x2,t,b,zo,zi,y,y_0,y_1 = get_data(u1,u2,u3,u4,number_observations,number_dimensions,graph)
    true_ate = np.mean(y_1 - y_0)
    x1_train, x1_test, x2_train, x2_test, t_train, t_test, b_train, b_test, zo_train, zo_test, zi_train, zi_test, y_train, y_test = train_test_split(x1, x2, t, b, zo, zi, y, test_size=0.2)
    
    covariates_train = np.concatenate([x1_train, x2_train, zo_train, zi_train], axis = 1)
    covariates_test = np.concatenate([x1_test, x2_test, zo_test, zi_test], axis = 1)
    return covariates_train, t_train, b_train, y_train, covariates_test, t_test, b_test, y_test, true_ate

def get_joint_t(t_train):
    t_train = pd.Series(t_train[:,0])
    joint_t = t_train.value_counts().to_dict()
    joint_t = np.array(list(joint_t.values()))
    joint_t = joint_t/np.sum(joint_t)
    return joint_t

def get_effect(features_train, t_train, y_train, features_test, t_test):
    joint_t = get_joint_t(t_train)
    t0_index = (t_train == 0).reshape(-1)
    t1_index = (t_train == 1).reshape(-1)
    t0_index_ = (t_test == 0).reshape(-1)
    t1_index_ = (t_test == 1).reshape(-1)
    
    regressor0 = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1])
    regressor1 = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1])
    regressor0.fit(features_train[t0_index,:], y_train[t0_index])
    regressor1.fit(features_train[t1_index,:], y_train[t1_index])
    ydo0 = joint_t[0]*regressor0.predict(features_test[t0_index_,:]) + joint_t[1]*regressor1.predict(features_test[t0_index_,:])
    ydo1 = joint_t[0]*regressor0.predict(features_test[t1_index_,:]) + joint_t[1]*regressor1.predict(features_test[t1_index_,:])
    return np.mean(ydo1) - np.mean(ydo0)

def get_all_subsets(number_cov):
    col = []
    for j in range(0,number_cov):
        col.append(str(j))
    return list(powerset(col, number_cov))

def powerset(s,number_cov):
    masks = [1 << i for i in range(number_cov)]
    for i in range(1,1 << number_cov):
        yield [ss for mask, ss in zip(masks, s) if i & mask]
        
def worker(subset,b_n,y_n,relevant_features,t_n):
    return RCoT(b_n,y_n,np.concatenate((relevant_features[subset].to_numpy(),t_n),axis=1))[0]

def worker_zizo(candidate_zo,b_n,relevant_features_n,t_n,candidate_z):
    relevant_zo = relevant_features_n[candidate_zo].to_numpy()
    if candidate_zo != candidate_z:
        candidate_zi = list(set(candidate_z) - set(candidate_zo))
        relevant_zi = relevant_features_n[candidate_zi].to_numpy()
        return [RCoT(relevant_zi,t_n)[0][0], RCoT(relevant_zo,t_n,np.concatenate((relevant_zi,b_n),axis=1))[0][0]]
    else:
        return [1.0, RCoT(relevant_zo,t_n,b_n)[0][0]]

def get_correct_subset(i_subset):
    return [int(subset) for subset in i_subset]


def main(args):
    starting_time = time.time()
    number_repetitions = args.nr
    number_observations = args.no
    graph = args.graph
    number_dimensions_list = [int(float(item)) for item in args.dim_list[0].split(',')]
    p_thresholds = [0.05,0.1,0.15,0.2,0.25]
    
    z0zi_ate_error = np.zeros((len(p_thresholds), number_repetitions,len(number_dimensions_list)))
    bzi_ate_error = np.zeros((len(p_thresholds), number_repetitions,len(number_dimensions_list)))
    baseline_ate_error = np.zeros((number_repetitions, len(number_dimensions_list)))
    
    set_start_method("spawn")
    for r_iter in range(number_repetitions):
        print(r_iter+1)
        for d_iter, number_dimensions in enumerate(number_dimensions_list):
            number_cov = 4*number_dimensions
            print(number_cov)
            
            covariates_train, t_train, b_train, y_train, covariates_test, t_test, b_test, y_test, true_ate = get_train_test_data(50000,number_dimensions,graph)
            
            baseline_ate_error[r_iter,d_iter] = np.abs(get_effect(b_train.reshape(-1,1), t_train, y_train, b_test.reshape(-1,1), t_test)-true_ate)
            covariates_train = covariates_train[:number_observations,:]
            t_train = t_train[:number_observations,:]
            b_train = b_train[:number_observations,:]
            y_train = y_train[:number_observations,:]
            covariates_test = covariates_test[:number_observations,:]
            t_test = t_test[:number_observations,:]
            b_test = b_test[:number_observations,:]
            y_test = y_test[:number_observations,:]
            
            covariates_n = StandardScaler().fit_transform(covariates_train)
            t_n = np.array(t_train).reshape(-1, 1)
            b_n = StandardScaler().fit_transform(b_train)
            y_n = StandardScaler().fit_transform(y_train)
            
            relevant_features_n = pd.DataFrame(covariates_n)
            relevant_features_n.columns = list(map(str, range(number_cov)))
            all_subsets = get_all_subsets(number_cov)
            all_p_values = []
            # with get_context("spawn").Pool(processes = 40) as pool:
            #     all_p_values = pool.map(partial(worker, b_n=b_n, y_n=y_n, relevant_features=relevant_features_n, t=t_n), all_subsets)
            for subset in all_subsets:
                all_p_values.append(worker(subset,b_n,y_n,relevant_features_n,t_n))
                
            for p_iter, p_threshold in enumerate(p_thresholds):
                indices_above_threshold = [i for i in range(len(all_p_values)) if all_p_values[i] >= p_threshold]
                all_z_above_threshold = list(map(all_subsets.__getitem__, indices_above_threshold))
                count_1 = 0
                count_2 = 0
                if len(all_z_above_threshold):
                    for candidate_z in all_z_above_threshold:
                        subsets_of_z = list(powerset(candidate_z,len(candidate_z)))
                        p_values_zizo = []
                        # try:
                        #     with get_context("spawn").Pool(processes = 40) as pool:
                        #         p_values_zizo = pool.map(partial(worker_zizo, candidate_zo=candidate_zo, b_n=b_n, relevant_features_n=relevant_features_n, t_n=t_n, candidate_z=candidate_z), subsets_of_z)
                        # except:
                        #     print("Failed at some z")
                        for candidate_zo in subsets_of_z:
                            try: 
                                p_values_zizo.append(worker_zizo(candidate_zo,b_n,relevant_features_n,t_n,candidate_z))
                            except:
                                print("Failed at zo " + str(candidate_zo) + "with z " + str(candidate_z))
                        thresholded_p_values_zizo = np.array(p_values_zizo) > p_threshold
                        zizo_indices = np.where(np.logical_and(thresholded_p_values_zizo[:,0], thresholded_p_values_zizo[:,1])==True)[0]
                        if len(zizo_indices):
                            correct_subset = get_correct_subset(candidate_z)
                            z0zi_ate_error[p_iter,r_iter,d_iter] += np.abs(get_effect(covariates_train[:,correct_subset], t_train, y_train, covariates_test[:,correct_subset], t_test)-true_ate)   
                            count_1 += 1
                            for zizo_index in zizo_indices:
                                zo = subsets_of_z[zizo_index]
                                if zo == candidate_z:
                                    bzi_ate_error[p_iter,r_iter,d_iter] += np.abs(get_effect(b_train.reshape(-1,1), t_train, y_train, b_test.reshape(-1,1), t_test)-true_ate)
                                    count_2 += 1
                                else:
                                    zi = list(set(candidate_z) - set(zo))
                                    correct_zi = get_correct_subset(zi)
                                    bzi_ate_error[p_iter,r_iter,d_iter] += np.abs(get_effect(np.concatenate((covariates_train[:,correct_zi], b_train.reshape(-1,1)), axis = 1), t_train, y_train, np.concatenate((covariates_test[:,correct_zi], b_test.reshape(-1,1)), axis = 1), t_test)-true_ate)
                                    count_2 += 1
                if count_1 > 0:
                    z0zi_ate_error[p_iter,r_iter,d_iter] = z0zi_ate_error[p_iter,r_iter,d_iter]/count_1
                else:
                    z0zi_ate_error[p_iter,r_iter,d_iter] = 'nan'
                if count_2 > 0:
                    bzi_ate_error[p_iter,r_iter,d_iter] = bzi_ate_error[p_iter,r_iter,d_iter]/count_2
                else:
                    bzi_ate_error[p_iter,r_iter,d_iter] = 'nan'
                    
        print(time.time() - starting_time)
        
    dim = number_dimensions_list[-1]   
    
    output = open('synthetic_' + str(graph) + '/' + str(dim) + '_' + str(number_observations) + '_z0zi_ate_error.pkl', 'wb')
    pickle.dump(z0zi_ate_error, output)
    output.close()
    output = open('synthetic_' + str(graph) + '/' + str(dim) + '_' + str(number_observations) + '_bzi_ate_error.pkl', 'wb')
    pickle.dump(bzi_ate_error, output)
    output.close()
    output = open('synthetic_' + str(graph) + '/' + str(dim) + '_' + str(number_observations) + '_baseline_ate_error.pkl', 'wb')
    pickle.dump(baseline_ate_error, output)
    output.close()
        
    avg_ate_error = pd.DataFrame(np.concatenate(((np.mean(baseline_ate_error, axis=0)).reshape(1,-1),(np.nanmean(z0zi_ate_error, axis=1)),(np.nanmean(bzi_ate_error, axis=1))), axis = 0),columns=number_dimensions_list)
    avg_ate_error.index = ['$b$','$z$ and 0.1','$z$ and 0.2','$z$ and 0.3','$z$ and 0.4','$z$ and 0.5','$f$ and 0.1','$f$ and 0.2','$f$ and 0.3','$f$ and 0.4','$f$ and 0.5']
    avg_ate_error.columns = [4*dim for dim in number_dimensions_list]
    avg_ate_error = avg_ate_error.replace(np.nan, 0)

    std_ate_error = pd.DataFrame(np.concatenate(((np.std(baseline_ate_error, axis=0)).reshape(1,-1)/np.sqrt(number_repetitions),(np.nanstd(z0zi_ate_error, axis=1))/np.sqrt(np.count_nonzero(~np.isnan(z0zi_ate_error), axis=1)),(np.nanstd(bzi_ate_error, axis=1))/np.sqrt(np.count_nonzero(~np.isnan(bzi_ate_error), axis=1)))),columns=number_dimensions_list)
    std_ate_error.index = ['$b$','$z$ and 0.1','$z$ and 0.2','$z$ and 0.3','$z$ and 0.4','$z$ and 0.5','$f$ and 0.1','$f$ and 0.2','$f$ and 0.3','$f$ and 0.4','$f$ and 0.5']
    std_ate_error.columns = [4*dim for dim in number_dimensions_list]
    std_ate_error = std_ate_error.replace(np.nan, 0)

    print("avg error")
    print(avg_ate_error)
    print(np.count_nonzero(~np.isnan(z0zi_ate_error), axis=1))
    print(np.count_nonzero(~np.isnan(bzi_ate_error), axis=1))
    print("std error")
    print(std_ate_error)
    
    # Generate Figure 9 bar plot
    plot_synthetic_results(number_repetitions, graph, number_dimensions_list)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--nr', help='number of repetitions', default=50, type=int)
    parser.add_argument('--no', help='number of observations', default=5000, type=int)
    parser.add_argument('--graph', help='which graph to use', default=1, type=int)
    parser.add_argument('-d', '--item', action='store', nargs='*', dest='dim_list', help='list of dimensions', type=str)
    args = parser.parse_args()
    main(args)