#%% Wind Shear Prediction

# John Obrecht
# Dec. '18

#%% Import various libraries

from __future__ import absolute_import, division, print_function
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
#print(tf.__version__)
from tensorflow import keras
from sklearn.model_selection import train_test_split

#%% Input Variables

# Maximum number of files to import
numFilesMax = 50

# What pct of data should be used for the analysis?  Start small
# not working just yet
pctData = 0.01

# Number of input variables
numInput = 22

# Define node numbers of first and second hidden layers
numNodes1 = 100
numNodes2 = 60

# Number of target variables
numTarget = 4

# Percent of the Data for Testing & Validation
pctTest     = 0.1
pctValidate = 0.1

# Learning rate for model
pctLearn = 0.001

# Number of Epochs for Training
numEpochs = 10

#%% Import Data

# Choose the full path name of the project folder (Home) and data repository (Data)
dirHome = '/Users/jmobrecht/anaconda3/PYTHON/Projects/Rotor Sensing'
dirData = dirHome + '/Loop2'

# Read all csv files names from the Data directory except for the *red.csv files 
fileList = [os.path.join(dirData, f) for f in os.listdir(dirData) if f.endswith('red.csv')]
print ('Number of files for analysis:', len(fileList))

# Read data into a DataFrame (df)
df = pd.DataFrame()
list_ = []
i = 0
for file_ in fileList:
    i += 1
    # Break out of the loop after loading a limited number of files (randomize?)
    if i > numFilesMax:
        break
    # Read in data here
    df = pd.read_csv(file_, header=None)
    # Find a way of randomly selecting a certain pct of data (---to do---)
    # Append df to the list of DataFrames
    list_.append(df)
#    print (file_)
# Concatenate the list of data into a DataFrame
df = pd.concat(list_, ignore_index=True)
print ('df:', df.head())

#%% Massage the Data Into a Good Form for the NN

# Create numpy array from the panda data frame
dataArray = np.asarray(df[0:])

# Identify the input data
xData = dataArray[:,0:numInput]
#print('xData:', xData) 

# Identify the target data
yData = dataArray[:,numInput:numInput + numTarget]
#print('yData:', yData)

print('Length of data:', len(xData))

#%% Test-Train Split of the Data

# Split data to train and test
xTrain, xTest, yTrain, yTest = train_test_split(xData, yData, test_size = pctTest)

print('Length of training data:', len(xTrain))
print('Length of testing data:', len(xTest))

print('Training set: {}'.format(xTrain.shape))
print('Testing set:  {}'.format(xTest.shape))

# Normalize the input data: note only test data is used for stats (---fix?---)
xMean = xTrain.mean(axis = 0)
xStd = xTrain.std(axis = 0)
xTrain = (xTrain - xMean) / xStd
xTest = (xTest - xMean) / xStd

# Normalize the target data?
#yMean = yTrain.mean(axis = 0)
#yStd = yTrain.std(axis = 0)
#yTrain = (yTrain - yMean) / yStd
#yTest = (yTest - yMean) / yStd

#%% Neural Network Architecture

## Define the model, optimization, etc.
def build_model():
    model = keras.Sequential([
        keras.layers.Dense(numNodes1, activation = tf.nn.relu, input_shape = (numInput,)),
        keras.layers.Dense(numNodes2, activation = tf.nn.relu),
        keras.layers.Dense(numTarget)
    ])
    optimizer = tf.train.RMSPropOptimizer(pctLearn)
    model.compile(loss = 'mse', optimizer = optimizer, metrics = ['mae'])
    return model
model = build_model()
model.summary()

dirCheckPt = dirHome + '/Training/cp.ckpt'
dirCheckPtFile = os.path.dirname(dirCheckPt)

#%% Training

# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 10 == 0: print('')
        print('.', end='')

# Store training stats: Create checkpoint callback
cp_callback = tf.keras.callbacks.ModelCheckpoint(dirCheckPt, save_weights_only=True, verbose=1)

# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 20)

# Train the NN
history = model.fit(xTrain, yTrain, epochs = numEpochs,
                    validation_split = pctValidate, verbose = 0,
                    callbacks = [cp_callback, early_stop, PrintDot()])

#%% Plot & Print Outputs

# Figure folder
dirFigure = dirHome + '/Figures/Regression'

#%% Figure 1: Plot the cost function for test & validation

figHistory = dirFigure + '/figHistory_' + str(numTarget) + '_' + str(numNodes1) + '_' + str(numNodes2) + '.png'

def plot_history(history):
    plt.figure(1)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error')
    plt.plot(history.epoch, np.array(history.history['mean_absolute_error']), label='Train Loss')
    plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']), label = 'Val loss')
    plt.legend()
    plt.show()
    plt.savefig(figHistory)

plot_history(history)

[loss, mae] = model.evaluate(xTest, yTest, verbose=0)

print('Testing set Mean Abs Error:', mae)

#%% Figure 2: Plot the correlations between all output truth & predictions

plt.figure(2)

# Predictions
pTest = model.predict(xTest)
#print(yTest.shape)
#print(pTest.shape)

# Inverse normalization
#yTest = yStd * yTest + yMean
#pTest = yStd * pTest + yMean

# Define file names for saving figures
figTruePred = dirFigure + '/figTruePred_' + str(numTarget) + '_' + str(numNodes1) + '_' + str(numNodes2) + '.png'
figTrueDiff = dirFigure + '/figTrueDiff_' + str(numTarget) + '_' + str(numNodes1) + '_' + str(numNodes2) + '.png'
figDiff     = dirFigure + '/figDiff_'     + str(numTarget) + '_' + str(numNodes1) + '_' + str(numNodes2) + '.png'
figAllOut   = dirFigure + '/figAllOut_'   + str(numTarget) + '_' + str(numNodes1) + '_' + str(numNodes2) + '.png'

# Not sure why this is being changed to 32 bit...
yTest = yTest.astype('float32')

q = 0
for i in range(numTarget):
    
    q += 1
    plt.subplot(numTarget,3,q)
#    plt.clf()
    plt.scatter(yTest[:,i], pTest[:,i], s = 0.1)
    plt.xlabel('True')   
    plt.ylabel('Prediction')
    plt.axis('equal')
    #plt.xlim(plt.xlim())
    #plt.ylim(plt.ylim())
    #_ = plt.plot([-100, 100], [-100, 100])
#    plt.show()
#    plt.savefig(figTruePred)

    q += 1
    plt.subplot(numTarget,3,q)
#    plt.clf()
    plt.scatter(yTest[:,i], pTest[:,i] - yTest[:,i], s = 0.1)
    plt.xlabel('True')   
    plt.ylabel('Prediction - True')
    plt.axis('equal')
    #plt.xlim(plt.xlim())
    #plt.ylim(plt.ylim())
    #_ = plt.plot([-100, 100], [-100, 100])
#    plt.show()
#    plt.savefig(figTrueDiff)

    q += 1
    plt.subplot(numTarget,3,q)
#    plt.clf()
    error = pTest[:,i] - yTest[:,i]
    plt.hist(error, bins = 100)
    #plt.xlim([-1.,1.])
    plt.xlabel("Prediction - True")
#    plt.show()
#    plt.savefig(figDiff)
    
plt.savefig(figAllOut)

#%% END
