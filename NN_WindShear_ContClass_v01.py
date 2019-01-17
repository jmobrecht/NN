#%% Wind Shear Prediction

# John Obrecht
# Dec. '18

#%% Import various libraries

from __future__ import absolute_import, division, print_function
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
#print(tf.__version__)
from tensorflow import keras
from sklearn.model_selection import train_test_split

#%% Input Variables

# Maximum number of files to import
numFilesMax = 20

# What pct of data should be used for the analysis?  Start small
# not working just yet
pctData = 0.1

# Number of input variables
numInput = 22

# Define node numbers of first and second hidden layers
numNodes1 = 100
numNodes2 = 60

# Column for Shear Values
colShear = numInput + 1

# Percent of the Data for Testing & Validation
pctTest     = 0.2
pctValidate = 0.1

# Learning rate for model
pctLearn = 0.02

# Number of Epochs for Training
numEpochs = 20

#%% Import Data

# Choose the full path name of the project folder (Home) and data repository (Data)
dirHome = '/Users/jmobrecht/anaconda3/PYTHON/Projects/Rotor Sensing'
dirData = dirHome + '/Loop2'

# Read all csv files names from the Data directory except for the *red.csv files 
fileList = [os.path.join(dirData, f) for f in os.listdir(dirData) if f.endswith('red.csv')]
print ('Number of files for analysis:', len(fileList))

# Read data into a DataFrame (df)
#df = pd.DataFrame()

df = pd.DataFrame()

list_ = []
i = 0
for file_ in fileList:
    i += 1
    # Break out of the loop after loading a limited number of files (randomize?)
    if i > numFilesMax:
        break
    # Read in data here and randomly sample a certain percentage
    df = pd.read_csv(file_, header = None).sample(frac = pctData)

    # Append df to the list of DataFrames
    list_.append(df)
    # Code for showing progress
    if i % 100 == 0: 
        print('...' + str(i) + '...', end = '')
        print('')

# Concatenate the list of data into a DataFrame
df = pd.concat(list_, ignore_index=True)

# Randomizes the sample - could be combined with earlier, but isn't for memory reasons
df = df.sample(frac=1)

# Columns:
cols = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
df.columns = cols

print ('df:', df.head())

#%% Massage the Data Into a Good Form for the NN
# up until now, the code should be the same for regression or classification

# Create numpy array from the panda data frame
dataArray = np.asarray(df[0:])

# Identify the input data
xData = dataArray[:,0:numInput]
#print('xData:', xData) 

numPts = len(xData)
print('Length of data:', numPts)

# id shear data
shear = dataArray[:,colShear]

# create array of "bin centers" for classification of shear
shearArray = np.arange(-0.7,1.0,0.1)
shearArrayStep = shearArray[1] - shearArray[0]

# Number of outputs
numTarget = len(shearArray)

# find the point in the array that is closest to the shear value
yData = np.zeros([numPts,numTarget])
for i in range(numPts):
    tmp = abs(shearArray - shear[i])
    yData[i,(tmp == min(tmp))] = 1

#%% Test-Train Split of the Data
# No change for classification

# Split data to train and test
xTrain, xTest, yTrain, yTest = train_test_split(xData, yData, test_size = pctTest)

numTrain = len(xTrain)
print('Length of training data:', numTrain)
numTest  = len(xTest)
print('Length of testing data:', numTest)

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
#        keras.layers.Dense(numNodes1, activation = tf.nn.relu, input_shape = (numInput,)),
#        keras.layers.Dense(numNodes2, activation = tf.nn.relu),
        keras.layers.Dense(numNodes1, activation = tf.nn.sigmoid, input_shape = (numInput,)),
        keras.layers.Dense(numNodes2, activation = tf.nn.sigmoid),
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

#%% Post-Processing

# Import needed libraries
from numpy import exp
import scipy.optimize
from scipy.optimize import curve_fit, fmin

# Predictions
pTest = model.predict(xTest)

#%% Gaussian Fit

#banana = lambda x: 100*(x[1]-x[0]**2)**2+(1-x[0])**2
#xopt = scipy.optimize.fmin(func=banana, x0=[-1.2,1])

# Define Model: Gaussian
def gaussian(x, a, xc, s):
    return a * exp(-(x-xc)**2 /2 /s**2)

# Specify the fit data in x
xFit = shearArray

pTestVals = np.zeros([numTest,3])
shearFit  = np.zeros([numTest,1])
for i in range(numTest):
    # Specify the fit data in y
    yFit = pTest[i]
    # Eliminate negative values
    yFit[yFit<0] = 0
    
    shearFit[i] = shearArray[yTest[i,:]==1]
    
    for j in range(1):
        if j == 0:
            # Initial Guesses: Amplitude - max of yFit, Center - "CoG", Sigma - 0.2
            init_vals = [max(yFit), sum(xFit*yFit)/sum(yFit), max(yFit)/sum(yFit)*shearArrayStep]  # for [a, xc, s]
#            init_vals = [max(yFit), shearArray[yFit == max(yFit)], max(yFit)/sum(yFit)*shearArrayStep]  # for [a, xc, s]
        else:
            # Initial Guesses: Amplitude - max of yFit, Center - "CoG", Sigma - 0.2
            init_vals = [bestVals[0], bestVals[1], bestVals[2]]  # for [a, xc, s]

    # Fitting done here
    bestVals, covar = curve_fit(gaussian, xFit, yFit, p0 = init_vals)

    # Record the outputs from the fit
    pTestVals[i,0:3] = bestVals[0:3]

#%% Plot & Print Outputs

# Figure folder
dirFigure = dirHome + '/Figures/ContClass'

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

ind = 17

plt.plot(shearArray,pTest[ind],'-r.')
plt.xlabel('Shear Value ()')
plt.ylabel('Probability of Guess ()')

plt.plot([shear[ind],shear[ind]],[0,1],'-g')

xLine = np.arange(min(shearArray),max(shearArray),1E-2)
yLine = gaussian(xLine,pTestVals[ind,0],pTestVals[ind,1],pTestVals[ind,2])
plt.plot(xLine,yLine,'-k')

plt.show()
plt.hold('off')

#plt.savefig(figAllOut)

#%% Figure 3

plt.figure(3)

num3 = 100
x3 = np.random.randint(0,numTest,num3)

plt.subplot(3,1,1)
plt.plot(range(num3),shearFit[x3],'-b.')
plt.errorbar(range(num3),pTestVals[x3,1],pTestVals[x3,2],color='red')
plt.show()

resTest = np.zeros([num3,1])
for i in range(num3):
    resTest[i] = (shearFit[x3[i]]-pTestVals[x3[i],1])

plt.subplot(3,1,2)
plt.plot(range(num3),resTest,'-g.')

#%% Figure 4

x = shearFit[:,0]
y = pTestVals[:,1]

plt.figure(4)
f5 = sns.kdeplot(x, y)#, shade = 'true')
plt.plot([min(shearArray),max(shearArray)],[min(shearArray),max(shearArray)],'r-')
plt.axis('equal')
plt.show()

#plt.subplot(3,1,2)
#plt.plot(shear)

#%% END
