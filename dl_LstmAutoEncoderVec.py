#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 23:43:07 2018

@author: Shariful
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras_anomaly_detection.library.plot_utils import visualize_reconstruction_error
from keras_anomaly_detection.library.recurrent import LstmAutoEncoder

def main():

#    data_dir_path = '/Users/Shariful/Documents/GitHubRepo/Datasets/ecg_demo/data'
    data_dir_path = '/Users/Shariful/Documents/GitHubRepo/deeplearning/demo_data'
#    model_dir_path = '/Users/Shariful/Documents/GitHubRepo/Datasets/ecg_demo/models'
    model_dir_path = '/Users/Shariful/Documents/GitHubRepo/Datasets/adfa_demo/models'

#    ecg_data = pd.read_csv(data_dir_path + '/ecg_discord_test.csv', header=None)
    ecg_data = pd.read_csv(data_dir_path + '/adfa_ld.csv', skiprows=1, \
                           index_col=None, header=None)
    ecg_data = ecg_data.iloc[:, 0:-1]
    print(ecg_data.head())
    ecg_np_data = ecg_data.as_matrix()
    scaler = MinMaxScaler()
    ecg_np_data = scaler.fit_transform(ecg_np_data)
    print(ecg_np_data.shape)

    ae = LstmAutoEncoder()

#    # fit the data and save model into model_dir_path
#    ae.fit(ecg_np_data[0:79, :], model_dir_path=model_dir_path, estimated_negative_sample_ratio=0.9)

    # load back the model saved in model_dir_path detect anomaly
    ae.load_model(model_dir_path)
#    anomaly_information = ae.anomaly(ecg_np_data[:23, :])
    anomaly_information = ae.anomaly(ecg_np_data, threshold=2.02)
    reconstruction_error = []
    for idx, (is_anomaly, dist) in enumerate(anomaly_information):
        print('# ' + str(idx) + ' is ' + ('abnormal' if is_anomaly else 'normal') + ' (dist: ' + str(dist) + ')')
        reconstruction_error.append(dist)

    visualize_reconstruction_error(reconstruction_error, ae.threshold)

#
if __name__ == '__main__':
    main()