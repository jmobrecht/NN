# Code from SY that you adapted for predicting power curves based on AEP data
# SY ran the Keras model from Tensorflow

# Nov '18

#%% Import various libraries

from __future__ import absolute_import, division, print_function
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

#%% Import Data

# Choose the full path name of the folder where the data is kept
path = '/Users/jmobrecht/anaconda3/PYTHON/Projects/Power Curve/data/'

# Read all csv files names from the Data directory except for the *red.csv files 
csv_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.csv') and not f.endswith('red.csv')]
print ('Number of files for analysis:', len(csv_files))

# Read data ---- Johnny, check the limit on inum below.  Should it be 5?
df = pd.DataFrame()
list_ = []
inum = 0
for file_ in csv_files:
    inum += 1
    if inum > 5: # 100
        break
    df = pd.read_csv(file_, header=None)
    list_.append(df)
    print (file_)
df = pd.concat(list_, ignore_index=True)
print ('df:', df.head())

# Create numpy array from the panda data frame
data = np.asarray(df[1:])

# Use one or several operational parameters as output data (y_data)
number_output = 23

# Variable number - number of input parameters
variable_number = 13

# 
y_data = data[:,variable_number:variable_number+number_output]
print('ydata', y_data)

# 
x_data = data[:,0:variable_number]
print('xdata', x_data)


print('Length of data', len(x_data))

#%% Test-Train Split of the Data

# Split data to train and test samples
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)

print('Length of train data', len(x_train))
print('Length of test data', len(x_test))

#
print(tf.__version__)

print('Training set: {}'.format(x_train.shape))  
print('Testing set:  {}'.format(x_test.shape))

# Test data is not used when calculating the mean and std
# Add treatment
#x_mean = x_train.mean(axis=0)
#x_std = x_train.std(axis=0)
##print('mean', mean)
##print('std', std)
##exit()
#x_train = (x_train - x_mean) / x_std
#x_test = (x_test - x_mean) / x_std
#
## Test data is not used when calculating the mean and std
## Add treatment
##y_mean = y_train.mean(axis=0)
##y_std = y_train.std(axis=0)
##print('mean', mean)
##print('std', std)
##exit()
##y_train = (y_train - y_mean) / y_std
##y_test = (y_test - y_mean) / y_std
#
#print(x_train[0])  # Print normalized training sample
#
##exit()

# Define node numbers of first and second hidden layers
node_number_first = 100
node_number_second = 60

#%% Neural Network Architecture

# Define the model, optimization, etc.
def build_model():
    model = keras.Sequential([
        keras.layers.Dense(node_number_first,  activation=tf.nn.relu,
                           input_shape=(x_train.shape[1],)),
        keras.layers.Dense(node_number_second, activation=tf.nn.relu),
        keras.layers.Dense(number_output)
    ])
    optimizer = tf.train.RMSPropOptimizer(0.001)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
    return model

model = build_model()
model.summary()

checkpoint_path = path + 'training/cp.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)

# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 10 == 0: print('')
        print('.', end='')

EPOCHS = 1000

# Store training stats

# Create checkpoint callback
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, 
                                                 save_weights_only=True,
                                                 verbose=1)

# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

history = model.fit(x_train, y_train, epochs=EPOCHS,
                    validation_split=0.2, verbose=0,
                    callbacks=[cp_callback, early_stop, PrintDot()])

#%% Plot & Print Outputs

# Figure 1: Plot the cost function for test & validation

fig_history = 'fig_history' + str(variable_number) + '_' + str(node_number_first) + '_' + str(node_number_second) + '.png'

def plot_history(history):
    plt.figure(1)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error')
    plt.plot(history.epoch, np.array(history.history['mean_absolute_error']), label='Train Loss')
    plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']), label = 'Val loss')
    plt.legend()
    #plt.ylim([0, 5])
    #plt.show()
    plt.savefig(fig_history)

plot_history(history)

[loss, mae] = model.evaluate(x_test, y_test, verbose=0)

print('Testing set Mean Abs Error:', mae)

# Predictions
#test_predictions = model.predict(test_data).flatten()
p_test = model.predict(x_test)

print(y_test.shape)
print(p_test.shape)

# Inverse normalization
#y_test = y_std*y_test + y_mean
#p_test = y_std*p_test + y_mean

# Define file names for saving figures
fig_true_pred = 'fig_true_pred' + str(variable_number) + '_' + str(node_number_first) + '_' + str(node_number_second) + '.png'
fig_true_diff = 'fig_true_diff' + str(variable_number) + '_' + str(node_number_first) + '_' + str(node_number_second) + '.png'
fig_diff = 'fig_diff' + str(variable_number) + '_' + str(node_number_first) + '_' + str(node_number_second) + '.png'

y_test = y_test.astype('float32')

# Figure 2: Plot the correlations between all output truth & predictions

plt.figure(2)
q = 0
for i in range(number_output):
    
    q += 1
    plt.subplot(4,6,q)
#    plt.clf()
    plt.scatter(y_test[:,i], p_test[:,i], s=0.1)
    plt.xlabel('True')   
    plt.ylabel('Prediction')
    plt.axis('equal')
    #plt.xlim(plt.xlim())
    #plt.ylim(plt.ylim())
    #_ = plt.plot([-100, 100], [-100, 100])
    #plt.show()
    plt.savefig(fig_true_pred)

#    q += 1
#    plt.subplot(3,number_output,q)
#    plt.clf()
#    plt.scatter(y_test[:,i], p_test[:,i]-y_test[:,i], s=0.1)
#    plt.xlabel('True')   
#    plt.ylabel('Prediction - True')
#    #plt.axis('equal')
#    #plt.xlim(plt.xlim())
#    #plt.ylim(plt.ylim())
#    #_ = plt.plot([-100, 100], [-100, 100])
#    #plt.show()
#    plt.savefig(fig_true_diff)
#
#    q += 1
#    plt.subplot(3,number_output,q)
##    plt.clf()
#    error = p_test[:,i] - y_test[:,i]
#    plt.hist(error, bins = 100)
#    #plt.xlim([-1.,1.])
#    plt.xlabel("Prediction - True")
#    #plt.show()
#    plt.savefig(fig_diff)

# Figure 3: Plot an example power curve vs. predicted power curve

tn = 1
plt.figure(3)
plt.plot(np.arange(3,26),y_test[tn,:],'k.-')
plt.plot(np.arange(3,26),p_test[tn,:],'ro')

#%% End
