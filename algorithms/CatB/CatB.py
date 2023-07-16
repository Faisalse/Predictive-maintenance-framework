# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 16:21:37 2023

@author: shefai
"""

from catboost import CatBoostClassifier
import numpy as np

class CatB:
    def __init__(self, iterations = 30, learning_rate = 0.1, depth = 4, l2_leaf_reg = 1):
        
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
       
        
    def fit(self, train, test):
        clf = CatBoostClassifier(iterations = self.iterations, learning_rate = self.learning_rate, 
                                 depth = self.depth, l2_leaf_reg = self.l2_leaf_reg)
        
        clf.fit(train.iloc[:, :-1], train.iloc[:, -1])
        self.clf = clf
        
    def predict(self, test):
        
        y_predict = self.clf.predict_proba(test.iloc[:,:-1])
        return np.round(y_predict[:, 1])
    
    def clear(self):
        self.iterations = 0
        self.learning_rate = 0
        self.depth = 0
        self.l2_leaf_reg = 0
    
    
    
    