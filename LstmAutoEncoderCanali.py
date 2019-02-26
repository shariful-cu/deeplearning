#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 23:43:07 2018

@author: Shariful
"""



import os, sys
path = os.path.abspath(__file__)
dir_path = os.path.dirname(path)
idx = dir_path.rfind('/')
if idx == -1:
    idx = dir_path.rfind('\\')
sys.path.append(dir_path[: -(len(dir_path) - idx)])


import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras_anomaly_detection.library.plot_utils import visualize_reconstruction_error
from keras_anomaly_detection.library.recurrent import LstmAutoEncoder
import numpy as np

from scipy.spatial import distance
from  sklearn import metrics
from matplotlib import pyplot as plt




def plot_ROC(test_labels, test_predictions):
    fpr, tpr, thresholds = metrics.roc_curve(
            test_labels, test_predictions, pos_label=1)
    auc = "%.2f" % metrics.auc(fpr, tpr)
    title = 'ROC Curve, AUC = '+str(auc)
    with plt.style.context(('ggplot')):
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, "#000099", label='ROC curve')
        ax.plot([0, 1], [0, 1], 'k--', label='Baseline')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right')
        plt.title(title)
    return fig 

def base_tf_idf(train_set, test_set, train_mode=True):
    if train_mode:
        dist_test = distance.cdist(test_set, train_set, metric='euclidean')
        test_predictions = dist_test.mean(axis=1)
        return test_predictions
    else:
#        load computed dist (TODO: )
        return True

def main():
#================read training dataset====================

    #    train_path = '/Users/Shariful/Documents/GitHubRepo/Datasets/ADFA-LD/n-gram/5-gram/train/5_gram.csv'
#    attack test path
#    test_path = '/Users/Shariful/Documents/GitHubRepo/Datasets/ADFA-LD/n-gram/5-gram/5_gram_attack_2.csv'
#    test_data = pd.read_csv(test_path, index_col=0, usecols=[0,1,2,3,4,5])
#    test_data_np = test_data.as_matrix()
#    normal test path
    
#    data_dir_path = '/Users/Shariful/Documents/GitHubRepo/Datasets/ecg_demo/data'
    data_dir_path = (r'/Users/Shariful/Documents/SysCallDataset/PreparedData'
                     r'/Canali_dataset/sliding_window_5')
#    model_dir_path = '/Users/Shariful/Documents/GitHubRepo/Datasets/ecg_demo/models'
    model_dir_path = (r'/Users/Shariful/Documents/GitHubRepo/deeplearning/'
                      r'syscall_anomaly/Canali/trained_models')
    
    score_dir_path = (r'/Users/Shariful/Documents/GitHubRepo/deeplearning/'
                      r'syscall_anomaly/Canali/scores')

    canali_data = pd.read_csv(data_dir_path + '/train_set.csv', header=None)
#    canali_data = pd.read_csv(data_dir_path + '/train_set.csv', \
#                           index_col=0, usecols=[0,1,2,3,4,5])

#==================Fit the LSTM model=====================
#    ['0','1','2','3','4']
#    canali_data = canali_data.iloc[:, 0:-1]
#    print(canali_data.head())
    canali_np_data = canali_data.as_matrix()
#    scaler = MinMaxScaler()
#    canali_np_data = scaler.fit_transform(canali_np_data)
#    print(canali_np_data.shape)

    ae = LstmAutoEncoder()

    # fit the data and save model into model_dir_path
    ae.fit(canali_np_data, model_dir_path=model_dir_path, batch_size=1000, \
           epochs=20, estimated_negative_sample_ratio=None)

#==========Load the saved model===========
    
    # load back the model saved in model_dir_path detect anomaly
    ae.load_model(model_dir_path)

#=============read test dataset===============
    
#    test data set
    test_idx_path = data_dir_path + '/test_set_index_range_label.csv'
    df_test_idx = pd.read_csv(test_idx_path, header = None)
    
    test_path = data_dir_path + '/test_set.csv'
    df_test = pd.read_csv(test_path, header = None)
    
    df_test_np = df_test.as_matrix()
#    df_test_np = df_test_np[0:123649,:]
    
    test_labels = np.array(df_test_idx.iloc[:,-1])
        
#    ecg_np_test_data = canali_np_data[0:43559, :]
#    test_data_np = np.vstack((ecg_np_test_data, test_data_np))

#================predict scores on testing set============

    
#    anomaly_information = ae.anomaly(canali_np_data[:23, :])
    anomaly_information = ae.anomaly(df_test_np, threshold=150)
#    reconstruction_error = []
    idx_out = 0
    max_scores = np.zeros((df_test_idx.shape[0]))
    for idx_in, (is_anomaly, dist) in enumerate(anomaly_information):
#        print('# ' + str(idx) + ' is ' + ('abnormal' if is_anomaly else 'normal') + ' (dist: ' + str(dist) + ')')
#        reconstruction_error.append(dist)

        #finding the maximum score out of all subsequences' scores
        if idx_in <= df_test_idx.loc[idx_out][:][1]:
            if max_scores[idx_out] < dist:
                max_scores[idx_out] = dist
        else:
            idx_out += 1
            max_scores[idx_out] = dist

#    visualize_reconstruction_error(reconstruction_error, ae.threshold)
    visualize_reconstruction_error(max_scores, ae.threshold)
    
    
#=============load and plot the computed scores on testing set==============  
    
#    max_scores = pd.read_csv('/Users/Shariful/Documents/GitHubRepo/deeplearning/syscall_anomaly/scores_on_testset/lstm_128_units.csv', \
#                            header = None)
#    visualize_reconstruction_error(max_scores, 150)
    
#    draw the roc curve
    plot_ROC(test_labels, max_scores)
    
#    save the computed scores
    np.savetxt(score_dir_path + '/lstm_128_units.csv', max_scores, delimiter=",")
    

#
if __name__ == '__main__':
    main()