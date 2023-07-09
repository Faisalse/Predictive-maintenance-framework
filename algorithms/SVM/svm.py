# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 09:19:53 2023

@author: shefai
"""

from sklearn import svm
import numpy as np

class SVM:
    def __init__(self, kernel= 'linear'):
        self.kernel = kernel
        
        
    def fit(self, train, test):
        clf = clf = svm.SVC(kernel = self.kernel, probability= True)
        clf.fit(train.iloc[:, :-1], train.iloc[:, -1])
        self.clf = clf
        
    def predict(self, test):
        y_predict = self.clf.predict_proba(test)
        return np.round(y_predict[:, 1])
    def clear(self):
        self.kernel = 0
        