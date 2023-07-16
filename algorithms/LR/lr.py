# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 15:34:57 2023

@author: shefai
"""
from sklearn.linear_model import LogisticRegression
import numpy as np

class LR:
    def __init__(self, solver = "newton-cg", penalty = "l1", C = 100, var_smoothing = 0.000001):
        self.solver = solver
        self.penalty = penalty
        self.C = C
             
        
    def fit(self, train, test):
        clf = LogisticRegression(solver = self.solver, penalty= self.penalty, C = self.C, random_state=0)
        clf.fit(train.iloc[:, :-1], train.iloc[:, -1])
        self.clf = clf
        
    def predict(self, test):
        
        y_predict = self.clf.predict_proba(test.iloc[:,:-1])
        return np.round(y_predict[:, 1])
    def clear(self):
        self.solver = 0
        self.penalty = 0
        self.C = 0
        