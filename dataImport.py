#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 20:24:38 2018

@author: jmobrecht
"""

#################################################
## Import Libraries and Folder Locations
#################################################

# Libraries 

import tensorflow as tf
print('\n' + 'Python Version: ' + tf.VERSION + '\n')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Folder Locations

folder = '/Users/jmobrecht/GitHub/NN'
file = 'listdata.csv'
path = folder + '/' + file
print('Folder Location: ' + path + '\n')

# Import Data w/ Pandas

my_data = pd.read_csv(path)
nrowsTot = np.size(my_data,0)
ncolsTot = np.size(my_data,1)

#################################################
## Assign Data & Variables
#################################################

# Assign data to Input, Target, and Other

dataInput   = my_data.iloc[:,0:13] # dataframes
dataTarget  = my_data.iloc[:,13:36]
dataOther   = my_data.iloc[:,36:40]
ncolsInput  = np.size(dataInput,1)
nrowsTarget = np.size(dataTarget,1)
nrowsOther  = np.size(dataOther,1)

# Alternative way to split the dataframe: split at cols 13 & 36!
dataInput2, dataTarget2, dataOther2 = np.split(my_data,[13,36],axis=1) 

# Alternative way to split the data into tensors: column widths 13, 23, 4!
dataInput3, dataTarget3, dataOther3 = tf.split(my_data,[13,23,4],1)

# Note: Tensorflow variables don't seem to show up in Spyder!  This code shows that... just for fun.
a = tf.constant([1.0], name="a")
b = tf.constant([2.0], name="b")
c = a + b
with tf.Session() as sess:
    print(sess.run(c))
    sess.close()

# Assign Wind Speed Arrays
    
arrMeanWind = np.arange(5.0,11.5,0.5)
arrPCWind = np.arange(3.0,26.0,1.0)

#################################################
## Split Data into Training & Test
#################################################

# Define test, validation, training split

trainSplit = 0.8
nrowsTrain = int(np.round(trainSplit*nrowsTot))
nrowsTest  = nrowsTot - nrowsTrain

# Define shuffle array

shuf = np.arange(nrowsTot)
np.random.shuffle(shuf)

# Split the data into Input, Target / Train & Test

dataInputTrain  = dataInput.loc[shuf[0:nrowsTrain]]
dataTargetTrain = dataTarget.loc[shuf[0:nrowsTrain]]
dataInputTest   = dataInput.loc[shuf[0:nrowsTest]]
dataTargetTest  = dataTarget.loc[shuf[0:nrowsTest]]

# Define learning rate

learningRate = 0.01

# Starting fuckin around with Keras

from tensorflow import layers
from keras.models import Sequential

model = tf.keras.Sequential()
# Adds a densely-connected layer with 64 units to the model:
model.add(layers.Dense(55, activation='relu'))
# Add another:
model.add(layers.Dense(35, activation='relu'))
# Add a softmax layer with 10 output units:
model.add(layers.Dense(10, activation='softmax'))







#################################################
## END
#################################################