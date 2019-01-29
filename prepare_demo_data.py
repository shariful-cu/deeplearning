#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 11:33:36 2019


********* Preparing demo subset of the original dataset and save it *******

@author: Shariful
"""

#importing necessary python modules
import pandas as pd
import numpy as np


##======Human Gait Dataset======

#org_path = '/Users/Shariful/Documents/GitHubRepo/Datasets/GaitDataset'
#
#file_path = org_path + '/gallery_gait.csv'
#gallery = pd.read_csv(file_path, header=None, index_col=None)
#
#objIds = list(gallery.iloc[:2,-1])
#gallery = gallery.iloc[:2,:]
#
#file_path = org_path + '/probe_gait.csv'
#probe = pd.read_csv(file_path, header=None, index_col=None)
#probe = probe.loc[probe.iloc[:,-1].isin(objIds)]
#
#
#demo_df = pd.concat([gallery, probe], ignore_index=True)
#
#save_path = '/Users/Shariful/Documents/GitHubRepo/deeplearning/supervised/demo_gait_data/demo_gait_data.csv'
#demo_df.to_csv(save_path, header=None, index=False)

#=====End of Gait==== 

#===========ADFA-LD system calls dataset=====

org_path = '/Users/Shariful/Documents/GitHubRepo/deeplearning/supervised/data'

#   Load train normal sequences
file_path = org_path + '/train_normal.csv'
train_normal = pd.read_csv(file_path, index_col=None)
label = np.zeros([len(train_normal), 1], dtype=int)

#   Load train normal sequences
file_path = org_path + '/test_normal.csv'
test_normal = pd.read_csv(file_path, index_col=None)
label = np.vstack((label, np.zeros([len(test_normal), 1], dtype=int)))


#   Load test attack sequences
file_path = org_path + '/test_attack.csv'
test_attack = pd.read_csv(file_path, index_col=None)
label = np.vstack((label, np.ones([len(test_attack), 1], dtype=int)))

demo_df = pd.concat([train_normal, test_normal, test_attack], \
                    ignore_index=True)
demo_df['label'] = pd.DataFrame(label)

save_path = '/Users/Shariful/Documents/GitHubRepo/deeplearning/demo_data/adfa_ld.csv'
demo_df.to_csv(save_path, index=False)

#=====End of ADFA-LD==== 