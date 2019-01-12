#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 13:29:04 2019


****** Supervised Neural Network for *******
    
    - Classification Problem 
    - Data Type: Fixed-size vector

@author: Shariful
"""


#importing utility files
from deeplearning.data_ import Data

#importing necessary python modules
import numpy as np
import os
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
from keras.models import load_model
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

#load data
#default 'path' of the dataset is 'demo'
#for whole dataset: please set the 'path' of your dataset's location
data = Data()
train_set, test_set = data.sampling() #default train size is 70%

# Convert the target to categorical: target [TODO: IMPROVEMENT NEED]
lab_train = np.array(train_set.iloc[:,-1])
lab_train[lab_train==94] = 0
lab_train[lab_train==95] = 1
target = to_categorical(lab_train)

# normalizing predictors
predictors = train_set.iloc[:,0:-1].values
scaler = MinMaxScaler()
predictors = scaler.fit_transform(predictors)

#size of input_shape
n_cols = predictors.shape[1]

#get the architecture of CNN
def get_new_model(input_shape = (n_cols,)):
    # Set up the model
    model = Sequential()
    
    # Add the first layer
    model.add(Dense(100, activation = 'relu', input_shape = input_shape))
    
    # Add layer 2
    model.add(Dense(100, activation = 'relu'))
    
   # Add layer 3
    model.add(Dense(100, activation = 'relu'))
    
    # Add the output layer
    model.add(Dense(2, activation = 'softmax'))
    return (model)


early_stopping_monitor = EarlyStopping(patience=2)

model = get_new_model()
    
# Compile the model
model.compile(optimizer = 'sgd', loss = 'categorical_crossentropy', \
              metrics = ['accuracy'])
    
# Fit the model
model.fit(predictors, target, validation_split = 0.3, \
          callbacks = [early_stopping_monitor], epochs=10)
  
#save the model
dir_path=os.path.dirname(__file__)
filePath = dir_path + '/gait_dl2_model.h5'
model.save(filePath)
   
#load the model
my_model = load_model(filePath)
   
# normalizing test set
df_test = test_set.iloc[:,0:-1]
scaler = MinMaxScaler()
df_test = scaler.fit_transform(df_test)

# labeling test set       [TODO: IMPROVEMENT NEED]
lab_test = np.array(test_set.iloc[:,-1])
lab_test[lab_test==94] = 0
lab_test[lab_test==95] = 1

#predict scores on test set
predictions = model.predict(df_test)

#To get the prediction, you should get the one with the highest value. 
#You can interpret them as probability even if there are not technically. 
predicted_label = predictions.argmax(axis=1)
predicted_acc = ((predicted_label == lab_test).sum())/lab_test.size
print('The predicted accuracy is: ', predicted_acc)
