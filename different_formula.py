"""
Reference: A Shah, K Shanmugam, M Kocaoglu
"Front-door Adjustment Beyond Markov Equivalence with Limited Graph Knowledge,"
In 37th Conference on Neural Information Processing Systems (NeurIPS), 2023

Last updated: December 15, 2023
Code author: Abhin Shah

File name: different_formula.py

Description: Demonstrate different causal effect formulas for Section 3.1.
Shows that different graphs satisfying same CI conditions can have different
formulas, illustrating the need for additional conditions in Theorem 3.2.
"""

import time
import pickle
import argparse
import numpy as np
import pandas as pd
from functools import partial
from sklearn.preprocessing import StandardScaler
from scipy.special import softmax, expit
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV
    
def get_u(number_observations):
    """Generate unobserved confounder u1 for Figure 3 example."""
    u1 = np.random.uniform(1.0,2.0,(number_observations,1))
    return u1

def get_theta():
    """Generate structural equation coefficients for Figure 3 graph."""
    theta1 = np.random.uniform(1.0,2.0,(2,1))
    theta2 = np.random.uniform(1.0,2.0,(2,1))
    theta3 = np.random.uniform(1.0,2.0,(1,1))
    theta4 = np.random.uniform(1.0,2.0,(2,1))
    return theta1, theta2, theta3, theta4

def get_data(u1,theta1,theta2,theta3,theta4,number_observations):
    """Generate synthetic data from Figure 3(top) structural equations."""
    zi = np.random.uniform(1.0,2.0,(number_observations,1))
    t = np.random.binomial(1, expit(np.dot(np.concatenate((u1,zi), axis = 1), theta1)-np.mean(np.dot(np.concatenate((u1,zi), axis = 1), theta1)))) 
    
    noise_b = 0.1* np.random.normal(0,1,(number_observations,1))
    b = np.dot(np.concatenate((t,zi),axis=1), theta2) + noise_b
    
    noise_zo = 0.1* np.random.normal(0,1,(number_observations,1))
    zo = np.dot(b, theta3) + noise_zo
    
    noise_y = 0.1* np.random.normal(0,1,(number_observations,1))
    y = np.dot(np.concatenate((u1,zo),axis=1),theta4) + noise_y   
    
    zeros =  np.zeros((number_observations,1))
    ones =  np.ones((number_observations,1))
    b_0 = np.dot(np.concatenate((zeros,zi),axis=1), theta2) + noise_b
    b_1 = np.dot(np.concatenate((ones,zi),axis=1), theta2) + noise_b
    zo_0 = np.dot(b_0, theta3) + noise_zo
    zo_1 = np.dot(b_1, theta3) + noise_zo
    y_0 = np.dot(np.concatenate((u1,zo_0),axis=1),theta4) + noise_y
    y_1 = np.dot(np.concatenate((u1,zo_1),axis=1),theta4) + noise_y
    return t,b,zo,zi,y,y_0,y_1

def get_train_test_data(number_observations):
    """Generate training and testing data for ATE estimation comparison."""
    u1 = get_u(number_observations)
    theta1, theta2, theta3, theta4 = get_theta()
    t,b,zo,zi,y,y_0,y_1 = get_data(u1,theta1,theta2,theta3,theta4,number_observations)
    true_ate = np.mean(y_1 - y_0)
    t_train, t_test, b_train, b_test, zo_train, zo_test, zi_train, zi_test, y_train, y_test = train_test_split(t, b, zo, zi, y, test_size=0.2)
    return zo_train, zi_train, t_train, b_train, y_train, zo_test, zi_test, t_test, b_test, y_test, true_ate

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


def main(args):
    starting_time = time.time()
    number_repetitions = args.nr
    number_observations = args.no
    
    zozi_ate_error = np.zeros((number_repetitions,1))
    bzi_ate_error = np.zeros((number_repetitions,1))
    all_ate = np.zeros((number_repetitions,1))
    
    for r_iter in range(number_repetitions):            
        zo_train, zi_train, t_train, b_train, y_train, zo_test, zi_test, t_test, b_test, y_test, true_ate = get_train_test_data(number_observations)
        
        
        all_ate[r_iter, 0] =    true_ate  
        zozi_ate_error[r_iter, 0] = np.abs(get_effect(np.concatenate((zo_train, zi_train), axis=1), t_train, y_train, np.concatenate((zo_test, zi_test), axis=1), t_test)-true_ate)
        bzi_ate_error[r_iter, 0] = np.abs(get_effect(np.concatenate((b_train, zi_train), axis=1), t_train, y_train, np.concatenate((b_test, zi_test), axis=1), t_test)-true_ate)
    
    output = open('diff_formula/zozi_ate_error.pkl', 'wb')
    pickle.dump(zozi_ate_error, output)
    output.close()
    output = open('diff_formula/bzi_ate_error.pkl', 'wb')
    pickle.dump(bzi_ate_error, output)
    output.close()

    print("avg ate")
    print(np.mean(all_ate))
    print(np.std(all_ate)/np.sqrt(number_repetitions))
    print("zozi error")
    print(np.mean(zozi_ate_error))
    print(np.std(zozi_ate_error)/np.sqrt(number_repetitions))
    print("bzi error")
    print(np.mean(bzi_ate_error))
    print(np.std(bzi_ate_error)/np.sqrt(number_repetitions))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--nr', help='number of repetitions', default=50, type=int)
    parser.add_argument('--no', help='number of observations', default=50000, type=int)
    args = parser.parse_args()
    main(args)