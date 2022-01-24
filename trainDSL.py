#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 10:39:25 2022

@author: loulou

Training base learners

"""
# Packages
import pandas as pd
import os
import sklearn
import scipy
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, BaggingClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, make_scorer, accuracy_score, f1_score, roc_curve
import numpy as np
from deepSuperLearner import *
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from support_functions import *
from joblib import dump, load
from sklearn.calibration import calibration_curve, CalibratedClassifierCV, CalibrationDisplay
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms
from matplotlib.gridspec import GridSpec
from imblearn.over_sampling import SMOTENC
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter


#%% Import Data
saved_classifiers_path = '/DeepSuperLearner/saved_classifiers/'
data_psl = pd.read_csv("data_psl_python.csv")
data_bichat = pd.read_csv("data_bichat_python.csv")

#%%
X_psl = data_psl.drop(["ecmo_in_ICU","ID"], axis = 1)
Y_psl = data_psl["ecmo_in_ICU"]
ID_psl = data_psl["ID"]

X_test = data_bichat.drop(["ecmo_in_ICU"],axis = 1)
y_test = data_bichat["ecmo_in_ICU"]

# Put continuous variables first to apply selective standardization over continuous variables
continuous_variables = ["age","BMI","euroscore","CPB_time","ACC_time","CPB_assist_time","RBC_units","FFP_units","platelets_units","max_dobu","intraoperative_VIS","max_norepinephrine","max_lac"]
categorical_variables = X_psl[X_psl.columns.difference(continuous_variables)].columns.tolist()
new_columns_order = continuous_variables + categorical_variables
X_psl = X_psl[new_columns_order]
X_test = X_test[new_columns_order]


## Normalize continuous variables
scaler = CustomMinMaxScaler(continuous_variables).fit_transform(X_psl)

X_psl_scaled = scaler.transform(X_psl)
X_test_scaled  = scaler.transform(X_test)

Y_psl = Y_psl.to_numpy()
y_test = y_test.to_numpy()


#%%
skf = StratifiedKFold(n_splits = 5,
                      shuffle = True,
                      random_state = 1990)

n_iter = 100


#%% Base Learners and Parameters spaces for hyperparameters tuning

### Random forest Classifier

RandomForest = RandomForestClassifier()

# parameter_space 
param_space_rf = {"n_estimators":scipy.stats.randint(100,400),
                  "max_features": ["auto","sqrt"],
                  "min_samples_leaf": [1,2,4],
                  "min_samples_split": [2,5,10],
                  "max_depth": [5,10,15,20,25,30,35,40,45,50],
                  "bootstrap": [True, False]
                  }



#XGBoost Classifier
XGB = XGBClassifier(n_jobs = -1,
                    booster = "gbtree",
                    objective="binary:logistic",
                    use_label_encoder = False,
                    eval_metric = "auc")

 
param_space_xgb = {"n_estimators": scipy.stats.randint(1,100),
                   "max_depth": [5,10,15,20,25,30,35,40,45,50],
                   "min_child_weight": [1,2,3,4,5,6,7,8,9,10],
                   "subsample": scipy.stats.uniform(),
                   "colsample_bytree":scipy.stats.uniform(),
                   "learning_rate":scipy.stats.loguniform(0.01,1)}





# Logistic Regression with Elasticnet penalty
LogReg = LogisticRegression(n_jobs = -1,
                            penalty = "elasticnet",
                            solver = "saga",
                            max_iter = 200)
param_space_logreg = {"C": scipy.stats.loguniform(0.001,1000),
                      "l1_ratio": [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7, 0.8,0.9,1]}

# Adaboost

AdaBoost = AdaBoostClassifier()

param_space_adaboost = {"n_estimators": scipy.stats.randint(1,100)}

# Bagging

Bagging_classif = BaggingClassifier()

param_space_bagging = {"n_estimators": [5,10,20,30,40,50],
                       "max_samples": [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
                       "max_features": [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]}


pr_auc = make_scorer(precision_recall_auc, greater_is_better=True)
#%% Train Test Split
skf_data = StratifiedKFold(n_splits = 4,
                      shuffle = True,
                      random_state = 1990)
best_auc = 0.0
best_pr = 0.0
fold = 0
for train_idx, valid_idx in skf_data.split(X_psl_scaled, Y_psl):
    print("Starting training fold: {}".format(fold))
    # Separate internal valid dataset
    X_train, X_valid = X_psl_scaled[train_idx], X_psl_scaled[valid_idx]
    y_train, y_valid = Y_psl[train_idx], Y_psl[valid_idx]
    
    over = SMOTENC(categorical_features= np.arange(7,20), sampling_strategy = 0.4)
    under = RandomUnderSampler(sampling_strategy=0.75)
    X_train_over, y_train_over = over.fit_resample(X_train, y_train)
    X_train_under, y_train_under = under.fit_resample(X_train_over, y_train_over)
    
    # Find best parameters for base learners using cross validation
    
    # RF
    random_search_rf = RandomizedSearchCV(RandomForest,
                                       param_distributions = param_space_rf,
                                       n_iter = n_iter,
                                       cv = skf,
                                       scoring = "roc_auc",
                                       n_jobs = -1)

    results_search_rf = random_search_rf.fit(X_train_under, y_train_under)

    report(results_search_rf.cv_results_)
    optimised_RF = results_search_rf.best_estimator_
    
    # XGBoost
    random_search_xgb = RandomizedSearchCV(XGB,
                                       param_distributions = param_space_xgb,
                                       n_iter = n_iter,
                                       cv = skf,
                                       scoring = "roc_auc",
                                       n_jobs = -1,
                                       random_state= 199)

    results_search_xgb = random_search_xgb.fit(X_train_under, y_train_under)
    report(results_search_xgb.cv_results_)
    optimised_XGB = results_search_xgb.best_estimator_
    
    
    #Elasticnet
    random_search_logreg = RandomizedSearchCV(LogReg,
                                              param_distributions= param_space_logreg,
                                              n_iter = n_iter,
                                              cv = skf, 
                                              scoring = "roc_auc",
                                              n_jobs = -1,
                                              random_state = 1990)


    results_search_logreg = random_search_logreg.fit(X_train_under, y_train_under)
    report(results_search_logreg.cv_results_)
    optimised_logreg = results_search_logreg.best_estimator_
    
    #Adaboost
    random_search_adaboost = RandomizedSearchCV(AdaBoost, 
                                                param_distributions = param_space_adaboost,
                                                cv = skf,
                                                scoring = "roc_auc",
                                                n_jobs = -1,
                                                random_state = 1990)

    results_search_adaboost = random_search_adaboost.fit(X_train_under, y_train_under)
    report(results_search_adaboost.cv_results_)
    optimised_adaboost = results_search_adaboost.best_estimator_
    
    #Bagging
    
    random_search_bagging = RandomizedSearchCV(Bagging_classif,
                                               param_distributions=param_space_bagging,
                                               n_iter = n_iter,
                                               cv = skf,
                                               scoring = "roc_auc",
                                               n_jobs = -1,
                                               random_state=1990)

    results_search_bagging = random_search_bagging.fit(X_train_under, y_train_under)
    report(results_search_bagging.cv_results_)
    optimised_bagging = results_search_bagging.best_estimator_
    
    
    #Deep Super Learner
    Base_learners = {'RandomForest': optimised_RF,  "XGBoost": optimised_XGB,"Elasticnet LogReg": optimised_logreg,
                     "AdaBoost": optimised_adaboost,  'Bagging': optimised_bagging}
    
    DSL_learner = DeepSuperLearner(Base_learners, K = 5)
    DSL_learner.fit(X_train_under,y_train_under)
    
    #Internal Validation
    int_valid_preds = DSL_learner.predict(X_valid)
    fold_auc = roc_auc_score(y_valid, int_valid_preds[:,1])
    fold_pr = precision_recall_auc(y_valid, int_valid_preds[:,1])
    print('Internal Valid ROC AUC: {:.4f}'.format(fold_auc))
    print('Internal Valid PR AUC: {:.4f}'.format(fold_pr))
    if fold_pr > best_pr:
        best_RF = optimised_RF
        best_XGB = optimised_XGB
        best_logreg = optimised_logreg
        best_adaboost = optimised_adaboost
        best_bagging = optimised_bagging
        best_DSL = DSL_learner
        best_fold = fold
        best_auc = fold_auc
        best_pr = fold_pr
        best_valid_idx = valid_idx
        best_train_idx = train_idx
    
    fold += 1

print("Best Fold: ",best_fold)
#%%% Save learners
dump(best_RF,os.path.join(saved_classifiers_path,"best_RF.joblib"))
dump(best_XGB,os.path.join(saved_classifiers_path,"best_XGB.joblib"))
dump(best_logreg,os.path.join(saved_classifiers_path,"best_logreg.joblib"))
dump(best_adaboost,os.path.join(saved_classifiers_path,"best_adaboost.joblib"))
dump(best_bagging,os.path.join(saved_classifiers_path,"best_bagging.joblib"))
dump(best_DSL,os.path.join(saved_classifiers_path,"best_DSL.joblib"))

#### FIN #####
