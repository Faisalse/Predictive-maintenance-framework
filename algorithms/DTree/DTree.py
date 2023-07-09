# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 16:21:37 2023

@author: shefai
"""

from sklearn.tree import DecisionTreeClassifier
import numpy as np

class DTree:
    def __init__(self, criterion = "gini", max_depth = 10, splitter = "random"):
        self.max_depth = max_depth
        self.criterion = criterion
        self.splitter = splitter
        
    def fit(self, train, test):
        clf = DecisionTreeClassifier(criterion = self.criterion, max_depth = self.max_depth)
        clf.fit(train.iloc[:, :-1], train.iloc[:, -1])
        self.clf = clf
        
    def predict(self, test):
        
        y_predict = self.clf.predict_proba(test)
        return np.round(y_predict[:, 1])
    def clear(self):
        self.max_depth = 0
        self.criterion = ""
        self.splitter = 0
    
    
    
    