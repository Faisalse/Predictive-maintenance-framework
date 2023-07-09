# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 15:34:57 2023

@author: shefai
"""
from sklearn.naive_bayes import GaussianNB
import numpy as np

class NB:
    def __init__(self, var_smoothing = 0.000001):
        self.var_smoothing = var_smoothing
        
    def fit(self, train, test):
        clf = GaussianNB()
        clf.fit(train.iloc[:, :-1], train.iloc[:, -1])
        self.clf = clf
        
    def predict(self, test):
        
        y_predict = self.clf.predict_proba(test)
        return np.round(y_predict[:, 1])
    def clear(self):
        self.var_smoothing = 0