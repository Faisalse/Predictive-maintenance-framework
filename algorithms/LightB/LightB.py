# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 16:21:37 2023

@author: shefai
"""

import lightgbm as lgb
import numpy as np

import warnings
warnings.filterwarnings('ignore')

class LightB:
    def __init__(self, n_estimators = 30, learning_rate = 0.1, max_depth = None, 
                 num_leaves = 10, min_child_samples = 31):
        
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        
    def fit(self, train, test):
        clf = lgb.LGBMClassifier(n_estimators = self.n_estimators, learning_rate = self.learning_rate,  
                                 max_depth = self.max_depth,
                                     num_leaves = self.num_leaves,
                                     min_child_samples = self.min_child_samples)
        
        clf.fit(train.iloc[:, :-1], train.iloc[:, -1])
        self.clf = clf
        
    def predict(self, test):
        
        y_predict = self.clf.predict_proba(test.iloc[:,:-1])
        return np.round(y_predict[:, 1])
    
    def clear(self):
        self.n_estimators = 0
        self.learning_rate = 0
        self.max_depth = 0
        self.min_samples_split = 0
        self.min_samples_leaf = 0
    
    
    
    