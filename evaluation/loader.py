import time
import os.path
import numpy as np
import pandas as pd
from _datetime import timezone, datetime


def load_data(path, file, train_eval=False):
    '''
    

    '''

    print('START load data')
    st = time.time()
    sc = time.time()


    train_appendix = '_train_full'
    test_appendix = '_test'
    if train_eval:
        train_appendix = '_train_tr'
        test_appendix = '_train_valid'

    train = pd.read_csv(path+file+train_appendix+".txt")
    test = pd.read_csv(path+file+test_appendix+".txt")
    
    print("Number of samples in train data  ", train.shape)
    print("Number of samples in test data  ", test.shape)


    return train, test


