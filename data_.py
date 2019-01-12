#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 19:58:25 2019

    -default 'path' of the dataset is 'demo'
    -for whole dataset: please set the 'path' of your dataset's location
    -loaded data automaticaly and assigned into 'dataset' attribute
    -For sampling, call sampling method

@author: Shariful
"""

#import necessary modules
import os, sys
import pandas as pd
import random

sys.path.append('../deeplearning') 

class Data:
    def __init__(self, path_of_data=''):
        
        if os.path.isfile(path_of_data):    
            dataset = pd.read_csv(path_of_data + '/gait_data.csv', \
                              index_col = None, header=None)
        else:
            path_of_data = os.path.dirname(__file__) + '/demo_data' 
            dataset = pd.read_csv(path_of_data + '/gait_data.csv', \
                              index_col = None, header=None)
        
        setattr(self, "dataset", dataset)
        
         
    #labels must be the last column    
    def sampling(self, train_size=0.70):
        
        if self.dataset.empty:
            print('\nsampling(): Error loadign the dataframe object')
            return
        
        train_set = [] 
        test_set = []
        for label in self.dataset.iloc[:,-1].unique():
            subset = self.dataset[self.dataset.iloc[:,-1] == label]
            sample_size = int(train_size * len(subset))
        
            random.seed = 3
            train_idxs = random.sample(range(0, len(subset)), sample_size)
            test_idxs = [idx for idx in range(0, len(subset)) if idx not in train_idxs]
            
            train_set.append(subset.iloc[train_idxs,:])
            test_set.append(subset.iloc[test_idxs,:])
            
        train_set = pd.concat(train_set, ignore_index=True)
        test_set = pd.concat(test_set, ignore_index=True)
            
        return (train_set, test_set)