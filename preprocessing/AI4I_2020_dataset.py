# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 21:29:36 2023

@author: shefai
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
DATA_PATH = r'./data/AI4I/raw/'
DATA_PATH_PROCESSED = r'./data/AI4I/fulltrain/'


DATA_FILE = "AI4I_2020_dataset"


def data_precessing(data_path = DATA_PATH, path_processed = DATA_PATH_PROCESSED, data_name = DATA_FILE):
    x, y = data_load(data_path, data_name)
    data_spilit(x, y, 0.20, 0.10, path_processed)
    

def data_load(path, name):
    path_name = path+name+".csv"
    data = pd.read_csv(path_name)
    
    print("Shape of data  ", data.shape)
    
    print("Data columns   ", data.columns)
    
    # delete unnecessasy_columns
    data = data[["Type", "Air temperature [K]", "Process temperature [K]",
                 "Rotational speed [rpm]","Torque [Nm]","Tool wear [min]", "Machine failure"]]
    
    print("After removing unnessary columns Data shape   ", data.shape)
    
    

    print("Numver of missing value   ", data.isnull().sum().sum())
    
    # SPILTS INTO FEATURES AND COLUMNS
    
    X = data.iloc[:, :-1]
    y =  data["Machine failure"]
    
    onehot_encoder = OneHotEncoder(handle_unknown='ignore')
    
    encoder_df = pd.DataFrame(onehot_encoder.fit_transform(X[['Type']]).toarray())
    
    
    del X["Type"]
    scaler = MinMaxScaler()
    # transform data
    X = scaler.fit_transform(X)
    
    X = pd.DataFrame(X)
    X.columns = ["Air temperature", "Process temperature", "Rotational speed","Torque","Tool wear"]
    # ordinal encode target variable
    
    X = X.join(encoder_df)
    
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    return X, y


def data_spilit(X, y, ratio, validation_ratio, path_processed):
    X_train, X_test, y_train, y_test = train_test_split( X, y, stratify = y, test_size=ratio, random_state=42, shuffle = True)
    
    train_full = X_train.copy()
    train_full["label"] = y_train
    
    test = X_test.copy()
    test["label"] = y_test
    
    dataName = "AI4I"
    
    train_full.to_csv(path_processed+dataName+"_train_full.txt", index = False)
    test.to_csv(path_processed+dataName+"_test.txt", index = False)
    
    X_train_tr, X_test_tr, y_train_tr, y_test_tr = train_test_split( X_train, y_train, stratify = y_train, 
                                                                    test_size=validation_ratio, random_state=42, shuffle = True)
    train_tr = X_train_tr.copy()
    train_tr["label"] = y_train_tr
    
    test_tr = X_test_tr.copy()
    test_tr["label"] = y_test_tr 
    
    train_tr.to_csv(path_processed+dataName+"_train_tr.txt", index = False)
    test_tr.to_csv(path_processed+dataName+"_train_valid.txt", index = False)
    
    
if __name__ =="__main__":
    data_precessing()

    
    
    


