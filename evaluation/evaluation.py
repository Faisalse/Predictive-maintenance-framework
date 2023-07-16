import time
import numpy as np
import pandas as pd

def evaluate_sessions(pr, key, metrics, test_data, training_time): 
    st = time.time();
    y_predict = pr.predict(test_data)
    y_actual = test_data.iloc[:,-1]
    
    
    for m in metrics:
        if hasattr(m, 'measure'):
            m.measure(y_actual, y_predict)
    end = time.time()    
    res = []
    for m in metrics:
        if hasattr(m, 'result'):
            score = m.result()
            res.append(score)
            
    res.append(("Training time ", np.round(training_time, decimals = 4)))
    return res


