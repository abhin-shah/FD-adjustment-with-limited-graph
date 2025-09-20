"""
Reference: A Shah, K Shanmugam, M Kocaoglu
"Front-door Adjustment Beyond Markov Equivalence with Limited Graph Knowledge,"
In 37th Conference on Neural Information Processing Systems (NeurIPS), 2023

Last updated: December 15, 2023
Code author: Abhin Shah

File name: supporting_func.py

Description: Supporting functions for front-door adjustment experiments including
graph generation, conditional independence testing, and causal effect estimation.
Contains utility functions for Tables 1-2 and Figures 5, 8-10 from the paper.
"""

import numpy as np
import pandas as pd
import networkx as nx

from itertools import chain, combinations
from sklearn.linear_model import RidgeCV
from scipy.special import softmax, expit
from sklearn.preprocessing import StandardScaler

from causallearn.graph.Dag import Dag
from causallearn.utils.DAG2PAG import dag2pag
from causallearn.graph.GraphNode import GraphNode
from causallearn.utils.GraphUtils import GraphUtils

import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
from rpy2.robjects.packages import importr
importr('RCIT')
RCoT = ro.r('RCoT')


def get_pag_R(A):
    """Convert PAG adjacency matrix to R-compatible format by remapping edge types."""
    ones = A == 1
    twos = A == 2
    negones = A == -1
    A[ones] = 2
    A[twos] = 1
    A[negones] = 3
    return A.T


def get_pag_adj(number_observed, number_confounders, A_total):
    """Convert DAG to PAG by marginalizing out unobserved confounders."""
    dag_nodes = []
    for node_name in range(number_observed + number_confounders):
        dag_nodes.append(GraphNode(node_name))
    dag = Dag(dag_nodes)
    for i in range(number_observed + number_confounders):
        for j in range(number_observed + number_confounders):
            if A_total[i,j] == 1:
                dag.add_directed_edge(dag_nodes[i],dag_nodes[j])
    cl_confounders = []
    for node_name in range(number_observed,number_observed+number_confounders):
        cl_confounders.append(dag_nodes[node_name])
    pag = dag2pag(dag, cl_confounders)
    order = []
    for node_name in dag_nodes[:number_observed]:
        order.append(pag.nodes.index(node_name))
    return pag.graph[order, :][:, order]


def get_tby(A_observed, possible_t):
    """Select treatment t, children b, and outcome y from graph structure."""
    number_observed = np.shape(A_observed)[0]
    t = np.random.choice(possible_t)
    b = set(np.where(A_observed[t,:] == 1)[0])
    y = number_observed-1
    return t,b,y


def get_possible_t(G_directed):
    """Find valid treatment variables that are ancestors but not parents/grandparents of y."""
    A_observed = nx.adjacency_matrix(G_directed).todense()
    number_observed = np.shape(A_observed)[0]
    ancestors_y = sorted(nx.ancestors(G_directed, number_observed-1))
    first_gen_y = list(np.where(A_observed[:,number_observed-1] == 1)[0])
    second_gen_y = []
    for node in first_gen_y:
        second_gen_y += list(np.where(A_observed[:,node] == 1)[0])
    return list(set(ancestors_y)-set(first_gen_y + second_gen_y))


def get_easy_possible_t(G_directed):
    """Find valid treatment variables that are ancestors but not parents of y (relaxed version)."""
    A_observed = nx.adjacency_matrix(G_directed).todense()
    number_observed = np.shape(A_observed)[0]
    ancestors_y = sorted(nx.ancestors(G_directed, number_observed-1))
    first_gen_y = list(np.where(A_observed[:,number_observed-1] == 1)[0])
    return list(set(ancestors_y)-set(first_gen_y))


def get_ER_adjacency_matrix(number_observed, degree, pp = 0.5):
    """Generate random DAG adjacency matrix with expected in-degree 'degree'."""
    A_observed = np.zeros((number_observed,number_observed))
    for j in range(number_observed):
        for i in range(j):
            if j <= 2*degree - 1:
                if np.random.uniform(0,1,1) < pp:
                    A_observed[i,j] = 1
            else:
                if np.random.uniform(0,1,1) < 2*pp*degree/j:
                    A_observed[i,j] = 1
    return A_observed


def check_additional_CI(G_total, t, b, candidate_z):
    """Check if candidate z satisfies additional conditional independence conditions (8) from Theorem 3.2."""
    condition = False
    for candidate_zo in powerset(candidate_z):
        candidate_zo = set(candidate_zo)
        candidate_zi = set(candidate_z) - candidate_zo
        if nx.d_separated(G_total, {t}, candidate_zi, {}) and nx.d_separated(G_total, {t}, candidate_zo, set(list(candidate_zi) + list(b))):
            return True, candidate_zo, candidate_zi
    return condition, None, None


def powerset(iterable):
    """Generate all subsets of an iterable (power set)."""
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
    

def degree_bounded_set(iterable, degree):
    """Generate all subsets of an iterable with cardinality at most 'degree'."""
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(degree+1))
  

def get_data(number_observed, number_confounders, samples, A_total, t, y, b):
    """Generate synthetic data from structural equation model with given graph structure."""

    col_names = []
    for i in range(number_observed+number_confounders):
        col_names.append(str(i))

    data = pd.DataFrame(columns=col_names)
    data_0 = pd.DataFrame(columns=col_names)
    data_1 = pd.DataFrame(columns=col_names)

    for con in range(number_observed, number_observed+number_confounders):
        data[str(con)] = np.random.uniform(1.0,2.0,(samples))
        data_0[str(con)] = data[str(con)]
        data_1[str(con)] = data[str(con)]

    all_theta = []
    for obs in range(number_observed):
        obs_parents = list(np.where(A_total[:,obs] == 1)[0])
        num_parents = len(obs_parents)

        if num_parents > 0:
            theta = np.random.uniform(1.0,2.0,num_parents)
            all_theta.append(theta)
            parent_data = (data[list(map(str, obs_parents))]).to_numpy()
            parent_data_0 = (data_0[list(map(str, obs_parents))]).to_numpy()
            parent_data_1 = (data_1[list(map(str, obs_parents))]).to_numpy()
            if obs == t:
                data[str(obs)] = np.random.binomial(1, expit(np.dot(parent_data, theta)-np.mean(np.dot(parent_data, theta))))
                data_0[str(obs)] = np.zeros(samples)
                data_1[str(obs)] = np.ones(samples)
            else:
                noise = 0.1*np.random.normal(0,1,samples)
                data[str(obs)] = (np.dot(parent_data, theta))/num_parents + noise
                data_0[str(obs)] = (np.dot(parent_data_0, theta))/num_parents + noise
                data_1[str(obs)] = (np.dot(parent_data_1, theta))/num_parents + noise
        else:
            if obs == t:
                data[str(obs)] = np.random.binomial(1,0.5,samples)
                data_0[str(obs)] = np.zeros(samples)
                data_1[str(obs)] = np.ones(samples)
            else:
                data[str(obs)] = np.random.uniform(1.0,2.0,samples)
                data_0[str(obs)] = data[str(obs)]
                data_1[str(obs)] = data[str(obs)]

    true_ate = np.mean(data_1[str(y)] - data_0[str(y)])

    data_t = data[str(t)]
    data_y = data[str(y)]
    data_b = data[list(map(str, list(b)))]
    data_x = data[list(map(str, list(set(range(number_observed)) - {t} - {y} - b)))]

    return data[list(map(str, range(number_observed)))], data_t, data_y, data_b, data_x, true_ate


def get_joint_t(t_train):
    """Compute marginal distribution of treatment variable."""
    t_train = pd.Series(t_train[:])
    joint_t = t_train.value_counts().to_dict()
    joint_t = np.array(list(joint_t.values()))
    joint_t = joint_t/np.sum(joint_t)
    return joint_t


def get_effect(features_train, t_train, y_train, features_test, t_test):
    """Estimate average treatment effect using outcome regression and marginalization."""
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
    
    
def worker(subset,b_n,y_n,relevant_features,t_n):
    """Test conditional independence b ⊥ y | t,z using RCoT for given subset z."""
    return RCoT(b_n,y_n,np.concatenate((relevant_features[subset].to_numpy(),t_n),axis=1))[0]


def worker_zizo(candidate_zo,b_n,relevant_features_n,t_n,candidate_z):
    """Test conditional independence conditions (8) for z^(i) and z^(o) decomposition."""
    if len(candidate_zo) == 0:
        candidate_zi = list(set(candidate_z) - set(candidate_zo))
        relevant_zi = relevant_features_n[candidate_zi].to_numpy()
        return [RCoT(relevant_zi,t_n)[0][0], 1.0]
    elif len(candidate_zo) < len(candidate_z):
        relevant_zo = relevant_features_n[candidate_zo].to_numpy()
        candidate_zi = list(set(candidate_z) - set(candidate_zo))
        relevant_zi = relevant_features_n[candidate_zi].to_numpy()
        return [RCoT(relevant_zi,t_n)[0][0], RCoT(relevant_zo,t_n,np.concatenate((relevant_zi,b_n),axis=1))[0][0]]
        
        
def get_errors(p_thresholds, c_train_all, c_test_all, t_train_all, t_test_all, b_train_all, b_test_all, y_train_all, y_test_all, samples, true_ate):
    """Compute ATE estimation errors for baseline and proposed methods across p-value thresholds."""
    z0zi_error = np.zeros((len(p_thresholds)))
    bzi_error = np.zeros((len(p_thresholds)))

    number_cov = c_train_all.shape[1]

    train_samples = int(0.8*samples)
    test_samples = int(0.2*samples)
        
    c_train = c_train_all.to_numpy()[:train_samples,:]
    t_train = t_train_all.to_numpy()[:train_samples]
    b_train = b_train_all.to_numpy().reshape(-1,b_train_all.shape[1])[:train_samples,:]
    y_train = y_train_all.to_numpy().reshape(-1,1)[:train_samples]
    c_test = c_test_all.to_numpy()[:test_samples,:]
    t_test = t_test_all.to_numpy()[:test_samples]
    b_test = b_test_all.to_numpy().reshape(-1,b_test_all.shape[1])[:test_samples,:]
    y_test = y_test_all.to_numpy().reshape(-1,1)[:test_samples]

    base_error = np.abs(get_effect(b_train, t_train, y_train, b_test, t_test)-true_ate)

    c_n = StandardScaler().fit_transform(c_train)
    t_n = np.array(t_train).reshape(-1, 1)
    b_n = StandardScaler().fit_transform(b_train)
    y_n = StandardScaler().fit_transform(y_train)

    relevant_features_n = pd.DataFrame(c_n)
    relevant_features_n.columns = list(map(str, range(number_cov)))
    all_subsets = list(powerset(list(range(number_cov))))
    all_subsets.pop(0) # the set z cannot be empty
    all_p_values = []
    for subset in all_subsets:
        all_p_values.append(worker(list(map(str, list(subset))),b_n,y_n,relevant_features_n,t_n))

    for p_iter, p_threshold in enumerate(p_thresholds):
        print("Percentage = " + str((p_iter+1)/len(p_thresholds)))
        indices_above_threshold = [i for i in range(len(all_p_values)) if all_p_values[i] >= p_threshold]
        all_z_above_threshold = list(map(all_subsets.__getitem__, indices_above_threshold))
        count_1 = 0
        count_2 = 0
        if len(all_z_above_threshold):
            for abhin, candidate_z in enumerate(all_z_above_threshold):
                    subsets_of_z = list(powerset(candidate_z))
                    subsets_of_z.pop(-1)
                    p_values_zizo = []
                    for candidate_zo in subsets_of_z: # candidate zo can be empty
                        try: 
                            p_values_zizo.append(worker_zizo(list(map(str, list(candidate_zo))),b_n,relevant_features_n,t_n,list(map(str, list(candidate_z)))))
                        except:
                            p_values_zizo.append([0.0,0.0])
                            print("Failed at zo " + str(candidate_zo) + "with z " + str(candidate_z))
                    thresholded_p_values_zizo = np.array(p_values_zizo) > p_threshold
                    zizo_indices = np.where(np.logical_and(thresholded_p_values_zizo[:,0], thresholded_p_values_zizo[:,1])==True)[0]
                    if len(zizo_indices):
                        correct_subset = list(candidate_z)
                        z0zi_error[p_iter] += np.abs(get_effect(c_train[:,correct_subset], t_train, y_train, c_test[:,correct_subset], t_test)-true_ate) 
                        count_1 += 1
                        for zizo_index in zizo_indices:
                            zo = subsets_of_z[zizo_index]
                            if len(zo) < len(candidate_z):
                                zi = list(set(candidate_z) - set(zo))
                                bzi_error[p_iter] += np.abs(get_effect(np.concatenate((c_train[:,zi], b_train), axis = 1), t_train, y_train, np.concatenate((c_test[:,zi], b_test), axis = 1), t_test)-true_ate)
                                count_2 += 1
        if count_1 > 0:
            z0zi_error[p_iter] = z0zi_error[p_iter]/count_1
        else:
            z0zi_error[p_iter] = 'nan'
        if count_2 > 0:
            bzi_error[p_iter] = bzi_error[p_iter]/count_2
        else:
            bzi_error[p_iter] = 'nan'
            
    return base_error, z0zi_error, bzi_error