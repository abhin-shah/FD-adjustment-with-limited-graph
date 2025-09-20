"""
Reference: A Shah, K Shanmugam, M Kocaoglu
"Front-door Adjustment Beyond Markov Equivalence with Limited Graph Knowledge,"
In 37th Conference on Neural Information Processing Systems (NeurIPS), 2023

Last updated: December 15, 2023
Code author: Abhin Shah

File name: german.py

Description: Analysis of the German Credit dataset to demonstrate front-door adjustment method.
Generates Figure 6 histograms showing p-value distributions for conditional independence tests
and computes bootstrap ATE estimates.
"""

import time
import pickle
import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from aif360.sklearn.datasets import fetch_german
from sklearn.linear_model import LogisticRegression
import argparse

import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
from rpy2.robjects.packages import importr
importr('RCIT')
RCoT = ro.r('RCoT')

def get_correct_subset(i_subset):
    """Convert string subset to integer indices."""
    return [int(subset) for subset in i_subset]

def get_joint_t(t_train):
    """Compute marginal distribution of treatment variable."""
    t_train = pd.Series(t_train[:, 0])
    joint_t = t_train.value_counts().to_dict()
    joint_t = np.array(list(joint_t.values()))
    joint_t = joint_t / np.sum(joint_t)
    return joint_t

def get_effect(features_train, t_train, y_train, features_test, t_test):
    """Estimate average treatment effect using outcome regression."""
    joint_t = get_joint_t(t_train)
    t0_index = (t_train == 0).reshape(-1)
    t1_index = (t_train == 1).reshape(-1)
    t0_index_ = (t_test == 0).reshape(-1)
    t1_index_ = (t_test == 1).reshape(-1)

    regressor0 = LogisticRegression()
    regressor1 = LogisticRegression()
    regressor0.fit(features_train[t0_index, :], y_train[t0_index].reshape(-1))
    regressor1.fit(features_train[t1_index, :], y_train[t1_index].reshape(-1))
    ydo0 = joint_t[0] * regressor0.predict_proba(features_test[t0_index_, :])[:, 1] + \
           joint_t[1] * regressor1.predict_proba(features_test[t0_index_, :])[:, 1]
    ydo1 = joint_t[0] * regressor0.predict_proba(features_test[t1_index_, :])[:, 1] + \
           joint_t[1] * regressor1.predict_proba(features_test[t1_index_, :])[:, 1]
    return np.mean(ydo1) - np.mean(ydo0)

def get_pval(*args):
    """Call RCoT and extract p-value."""
    res = RCoT(*args)
    return res.rx2('p.value')[0]

def main(args):
    """Main workflow for German Credit dataset Figure 6 analysis."""
    start_time = time.perf_counter()
    
    # Load dataset
    X, y = fetch_german()
    X = X.reset_index(drop=True)
    
    # Convert categorical variables to numeric
    for col in ['sex', 'own_telephone', 'foreign_worker']:
        X[col] = pd.factorize(X[col])[0]

    X.columns = pd.MultiIndex.from_arrays([X.columns.tolist(), X.columns.tolist()])

    # Create dummy variables for categorical features
    dummy_cols = ['checking_status', 'credit_history', 'purpose', 'savings_status', 'employment', 'other_parties',
                  'property_magnitude', 'other_payment_plans', 'housing', 'job', 'marital_status']
    for col in dummy_cols:
        dummies = pd.get_dummies(X[col])
        dummies.columns = pd.MultiIndex.from_arrays([[col] * len(dummies.columns), dummies.columns])
        X = X.drop(col, axis=1, level=0)
        X = pd.concat([X, dummies], axis=1)

    # Define treatment, children, and outcome variables
    treatment = ['age']
    t = X[treatment]
    t_binary = pd.DataFrame(y.reset_index()['age'])
    t_binary['age'] = t_binary['age'].replace('aged', 1).replace('young', 0)

    child = ['num_dependents', 'savings_status']
    b = X[child]
    X = X.drop(treatment + child, axis=1, level=0)

    y.index = pd.MultiIndex.from_arrays(y.index.codes, names=y.index.names)
    y = pd.Series(y.factorize(sort=True)[0], index=y.index).to_frame()

    # Convert to numpy arrays
    X_np = np.array(X)
    t_np = np.array(t)
    b_np = np.array(b)
    y_np = np.array(y)
    t_bin_np = np.array(t_binary)

    # Train-test split
    train_size_ = args.train_size
    covariates_train, covariates_test, t_train, t_test, t_bin_train, t_bin_test, \
    b_train, b_test, y_train, y_test = train_test_split(
        X_np, t_np, t_bin_np, b_np, y_np, train_size=train_size_, test_size=train_size_/4, random_state=1234567
    )

    # Prepare data for conditional independence testing
    covariates_n = StandardScaler().fit_transform(covariates_train)
    t_n = np.array(t_train).reshape(-1, 1)
    b_n = StandardScaler().fit_transform(b_train)
    y_n = StandardScaler().fit_transform(y_train)

    relevant_features_n = pd.DataFrame(covariates_n)
    relevant_features_n.columns = X.columns
    mapping = {old: str(i) for i, old in enumerate(X.columns.get_level_values(0).unique())}
    relevant_features_n = relevant_features_n.rename(columns=mapping)

    # Define adjustment sets
    final_zi = ['8', '10']
    final_zo = ['14']
    final_z = final_zi + final_zo

    # Generate p-value distributions
    number_training_data = b_train.shape[0]
    reps = args.reps

    # b ⊥ y | t,z
    pz_ = [get_pval(b_n[np.random.permutation(number_training_data//2), :],
                    y_n[np.random.permutation(number_training_data//2), :],
                    np.concatenate((relevant_features_n[final_z].to_numpy(), t_n), axis=1)[np.random.permutation(number_training_data//2), :])
           for _ in range(reps)]

    # z^(i) ⊥ t
    pzi_ = [get_pval(relevant_features_n[final_zi].to_numpy()[np.random.permutation(number_training_data//2), :],
                     t_n[np.random.permutation(number_training_data//2), :])
            for _ in range(reps)]

    # z^(o) ⊥ t | z^(i), b
    if final_zo != final_z:
        pzo_ = [get_pval(relevant_features_n[final_zo].to_numpy()[np.random.permutation(number_training_data//2), :],
                         t_n[np.random.permutation(number_training_data//2), :],
                         np.concatenate((relevant_features_n[final_zi].to_numpy(), b_n), axis=1)[np.random.permutation(number_training_data//2), :])
                for _ in range(reps)]
    else:
        pzo_ = [get_pval(relevant_features_n[final_zo].to_numpy()[np.random.permutation(number_training_data//2), :],
                         t_n[np.random.permutation(number_training_data//2), :],
                         b_n[np.random.permutation(number_training_data//2), :])
                for _ in range(reps)]

    # Save p-values
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, '1pz.pkl'), 'wb') as f:
        pickle.dump(pz_, f)
    with open(os.path.join(args.output_dir, '1pzi.pkl'), 'wb') as f:
        pickle.dump(pzi_, f)
    with open(os.path.join(args.output_dir, '1pzo.pkl'), 'wb') as f:
        pickle.dump(pzo_, f)

    # Plot histograms
    plt.rcParams.update({'mathtext.default': 'regular'})
    data = [pz_, pzi_, pzo_]
    colors = ['red', 'green', 'blue']
    axes = plt.subplots(1, 3, figsize=[12, 4], sharey=True)[1]
    titles = [r'$ b\perp\!_p ~ y | t,z$', r'$ z^{(i)} \perp\!_p t $', r'$z^{(o)} \perp\!_p t | z^{(i)}$']
    for i, ax in enumerate(axes):
        ax.hist(data[i], bins=np.linspace(0, 1, 11), alpha=0.2, color=colors[i])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xticks([0.0, 0.25, 0.50, 0.75, 1])
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.set_xlabel('p-value', fontsize=16)
        if i == 0:
            ax.set_ylabel('count', fontsize=16)
        ax.set_title(titles[i], fontsize=16)
        ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'Histogram.pdf'))
    
    print(f"\nCompleted in {time.perf_counter() - start_time:.2f} seconds.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_size', type=float, default=0.8, help='Training set fraction')
    parser.add_argument('--reps', type=int, default=100, help='Number of repetitions for p-value distributions')
    parser.add_argument('--output_dir', type=str, default='German', help='Directory to save results')
    args = parser.parse_args()
    main(args)
