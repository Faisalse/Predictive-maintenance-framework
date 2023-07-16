# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 12:24:37 2023

@author: shefai
"""

import pandas as pd
s = 0
data = pd.read_csv("diginetica_train_full.txt", sep = "\t")

data = data.sort_values(by = ["SessionId"])


ab = data[data["CatId"]  == 1010]
cb = data[data["CatId"]  == 699]

d = pd.concat([ab, cb])





index_item = ab.columns.get_loc("ItemId")
index_Name = ab.columns.get_loc("Name" )
item_dict = dict()


for row in ab.itertuples(index=False):
    if row[index_item] not in item_dict:
        item_dict[row[index_item]] = set(row[index_Name].split(","))
        

simi_dict = dict()
for key, values in item_dict.items():
    
    temp = list()
    for key1, values1 in item_dict.items():
        sim = len(values & values1) / (len(values) * len(values1))
        temp.append((key1, sim))
    
    temp = [v for v in sorted(temp, reverse= True, key = lambda x: x[1] )]
        
    simi_dict.update({key: temp})
    