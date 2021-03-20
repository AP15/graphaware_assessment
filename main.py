# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 10:49:11 2021

@author: apaudice
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

import functions as fns

# Import the data
df = pd.read_csv('cal-housing.csv')

# Codifica le feature categoriche
df =  pd.get_dummies(df[['longitude','latitude','housing_median_age',
'total_rooms','total_bedrooms','population','households','median_income',
  'median_house_value','ocean_proximity']])

# Check and count for missing values
print('# missing values for column:')
print(df.isnull().sum())
print('Fraction of missing values: {:1.5f}'.format(
    max(df.isnull().sum())/len(df.index) * 100))

# Delete rows with missing values as they represent a neglible fraction of the
# dataset
df = df.dropna(axis=0) 

# Split data in training and test datasets
df_train, df_test = train_test_split(df,test_size=20383, random_state=0)

X_train = df_train.drop(['median_house_value'], axis=1).values
y_train = df_train['median_house_value'].values/1000 # Rescale so that 1 unit corresponds to 1000 dollars (for readability)

X_test = df_test.drop(['median_house_value'], axis=1).values
y_test = df_test['median_house_value'].values/1000 # Rescale so that 1 unit corresponds to 1000 dollars (for readability)

# Training
std_scaler = MinMaxScaler()

# Hyper parameter tuning
n_fold = 5
n_params = 20
alpha_range = np.linspace(0.0005, 0.001, n_params)

kf = KFold(n_splits=n_fold)
err_estimate = np.zeros((n_fold, n_params))

for i in range(len(alpha_range)):
    alpha = alpha_range[i]
    for k, (train_idx, test_idx) in enumerate(kf.split(X_train, y_train)):
        X_val_train, y_val_train = X_train[train_idx], y_train[train_idx]
        X_val_test, y_val_test = X_train[test_idx], y_train[test_idx]

        # Standardize the data
        std_scaler.fit(X_val_train)
        X_train_std = std_scaler.transform(X_val_train)
        X_test_std = std_scaler.transform(X_val_test)
        
        # Learn the model
        w_hat = fns.optimalSolution(X_train_std, y_val_train, alpha)
        
        # Predict the labels
        err_estimate[k, i] = fns.computeER(X_test_std, y_val_test, w_hat)

scores = np.mean(err_estimate, axis=0)
plt.figure()
plt.plot(alpha_range, scores)
plt.xlabel('lambda')
plt.axhline(np.min(scores), linestyle='--', color='.5')
plt.xlim([alpha_range[0], alpha_range[-1]])
plt.title('5-fold-CV for tuning alpha')
plt.show()

# Train the model with the best alpha
alpha_opt = alpha_range[np.argmin(scores)]

# Standardize the data
std_scaler.fit(X_train)
X_train_std = std_scaler.transform(X_train)
X_test_std = std_scaler.transform(X_test)

# Train the model
w_hat = fns.optimalSolution(X_train_std, y_train, alpha_opt)

# Test the model
test_err = fns.computeER(X_test_std, y_test, w_hat)
print('Test RMSE: %1.5f'%(test_err))

# Compare against means, median and least square regression
mean = np.mean(y_train)
test_err_mean = fns.computeErrorPointEstimate(X_test_std, y_test, mean)
print('Test RMSE mean: %1.5f'%(test_err_mean))

median = np.median(y_train)
test_err_median = fns.computeErrorPointEstimate(X_test_std, y_test, median)
print('Test RMSE median: %1.5f'%(test_err_median))

w_lsr = fns.optimalLSR(X_train_std, y_train)
test_err_lsr = fns.computeER(X_test_std, y_test, w_lsr)
print('Test RMSE LSR: %1.5f'%(test_err_lsr))

# Will look at learning curve to see if we can improve the model
n_fold = 10
kf = KFold(n_splits=n_fold)

training_sizes = [10, 20, 30, 40, 50]
n_sizes = len(training_sizes)

err_estimate_train = np.zeros((n_fold, n_sizes))
err_estimate_cv = np.zeros((n_fold, n_sizes))
for i in range(len(training_sizes)):
    s = training_sizes[i]
    X_train_lc, y_train_lc = X_train[0:s], y_train[0:s]
    for k, (train_idx, test_idx) in enumerate(kf.split(X_train_lc, y_train_lc)):
        X_val_train, y_val_train = X_train_lc[train_idx], y_train_lc[train_idx]
        X_val_test, y_val_test = X_train_lc[test_idx], y_train_lc[test_idx]

        # Standardize the data
        std_scaler.fit(X_val_train)
        X_train_std = std_scaler.transform(X_val_train)
        X_test_std = std_scaler.transform(X_val_test)
        
        # Learn the model
        w_hat = fns.optimalSolution(X_train_std, y_val_train, alpha_opt)
        
        # Predict the labels
        err_estimate_train[k, i] = fns.computeER(X_train_std, y_val_train, w_hat)
        err_estimate_cv[k, i] = fns.computeER(X_test_std, y_val_test, w_hat)
        
scores_train = np.mean(err_estimate_train, axis=0)
scores_cv = np.mean(err_estimate_cv, axis=0)
bias = (scores_cv[-1] - scores_train[-1])/2
plt.figure()
plt.plot(training_sizes, scores_train, label='train_err')
plt.plot(training_sizes, scores_cv, label='cv_err')
plt.axhline(scores_train[-1] + bias, linestyle='--', color='.5', label='bias')
plt.ylabel('RMSE')
plt.xlabel('size')
plt.xlim([training_sizes[0], training_sizes[-1]])
plt.title('Learning curves')
plt.legend()
plt.show()
