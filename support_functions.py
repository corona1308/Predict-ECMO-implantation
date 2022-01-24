#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 31 14:51:34 2021

@author: loulou

Support functions
"""
from sklearn.metrics import  precision_recall_curve, auc, make_scorer
from sklearn import preprocessing
from sklearn.base import TransformerMixin
import numpy as np



class CustomScaler(TransformerMixin): 
    def __init__(self, continuous_variables):
        self.scaler = preprocessing.StandardScaler()
        self.continuous_variables = continuous_variables

    def fit(self, X, y):
        self.scaler.fit(X[X.columns[X.columns.isin(self.continuous_variables)]], y)
        return self

    def transform(self, X):
        X_head = self.scaler.transform(X[X.columns[X.columns.isin(self.continuous_variables)]])
        return np.concatenate((X_head, X[X.columns[~X.columns.isin(self.continuous_variables)]].to_numpy()), axis=1)


class CustomMinMaxScaler(TransformerMixin): 
    def __init__(self, continuous_variables):
        self.scaler = preprocessing.MinMaxScaler()
        self.continuous_variables = continuous_variables

    def fit_transform(self, X):
        self.scaler.fit_transform(X[X.columns[X.columns.isin(self.continuous_variables)]])
        return self

    def transform(self, X):
        X_head = self.scaler.transform(X[X.columns[X.columns.isin(self.continuous_variables)]])
        return np.concatenate((X_head, X[X.columns[~X.columns.isin(self.continuous_variables)]].to_numpy()), axis=1)
    
    
def precision_recall_auc(y_true, y_pred):
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    auc_score = auc(recall, precision)
    return auc_score



def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})"
                  .format(results['mean_test_score'][candidate],
                          results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")