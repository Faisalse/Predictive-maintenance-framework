# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 21:25:31 2023

@author: shefai
"""

from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import BCELoss
from algorithms.MLP.data import *
from algorithms.MLP.model import *

class MLP:
    def __init__(self, epoch = 5, lr = 0.001, batch_size = 40):
        self.lr = lr
        self.batch_size = batch_size
        self.epoch = epoch
    def fit(self, train, test):
        
        numberOfFeatures = train.shape[1] - 1
        data = DataClass(train)
        
        
        train_dl = DataLoader(data, batch_size= self.batch_size, shuffle= False)
        
        model = MlpModel(numberOfFeatures)
        
        criterion = BCELoss()
        optimizer = Adam(model.parameters(), lr=self.lr )
        
        for i in range(self.epoch):
            for k, (features, targets) in enumerate(train_dl):
                optimizer.zero_grad()
                y_pred = model(features)
                loss = criterion(y_pred, targets)
                loss.backward()
                optimizer.step()
                
        self.model = model
        
        
    def predict(self, test):
        
        data = DataClass(test)
        data_dl = DataLoader(data, batch_size = len(test) , shuffle= True)
        for i, (inputs, targets) in enumerate(data_dl):
            y_pre = self.model(inputs)
            y_pre = y_pre.detach().numpy()
            y_pre = np.round(y_pre)
            y_pre = y_pre.flatten()
        return y_pre
    
    def clear(self):
        pass
    
    
    
    
    
    
    
    
    # for i, (inputs, targets) in enumerate(test_dl):
    #     # evaluate the model on the test set
    #     yhat = model(inputs)
    #     # retrieve numpy array
    #     yhat = yhat.detach().numpy()
    #     actual = targets.numpy()
    #     actual = actual.reshape((len(actual), 1))
    #     # round to class values
    #     yhat = yhat.round()
    #     # store
    #     predictions.append(yhat)
    #     actuals.append(actual)
    # predictions, actuals = vstack(predictions), vstack(actuals)
    # # calculate accuracy
    # acc = accuracy_score(actuals, predictions)
    # return acc
    
    
    
    
    
    
    
    