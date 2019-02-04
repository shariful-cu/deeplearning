#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 19:30:09 2019

@author: Shariful
"""

# Load layers
# Input/dense/output layers
from keras.layers import Input, Dense
input_tensor = Input(shape=(1,))
output_tensor = Dense(1)(input_tensor)
## Import the plotting function
#from keras.utils import plot_model
#import matplotlib.pyplot as plt

# Build the model
from keras.models import Model
model = Model(input_tensor, output_tensor)

# Compile the model
model.compile(optimizer='adam', loss='mean_absolute_error')


## Summarize the model
#model.summary()
#
## Plot the model
#plot_model(model, to_file='model.png')
#
## Display the image
#data = plt.imread('model.png')
#plt.imshow(data)
#plt.show()

30 y
400 uni scholls
12k ids
11k  