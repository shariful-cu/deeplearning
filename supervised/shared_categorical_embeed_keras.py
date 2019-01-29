#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 05:56:45 2019

****** Supervised Neural Network for *******
    
    - Advanced Deep Learning with Keras in Python
    - Embeeding and Shared layers for categorical input
    - Muliput input multiple outputs 
    - Single model for both regression and classification problem 
    - Data Type: basketball_data

@author: Shariful
"""
# Load layers
from keras.layers import Input, Dense

# Input layer
input_tensor = Input(shape=(1,))

# Dense layer
output_layer = Dense(1)

# Connect the dense layer to the input_tensor
output_tensor = output_layer(input_tensor)
