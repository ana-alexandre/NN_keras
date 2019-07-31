#NN
#keras

#Neural network for a regression problem
#Reads in reco and lep kinematics
#(plus checks if leptons are present and, if so, if they're an electron or a muon) of a b-jet
#Trains to predict the true kinematics of the jet

import numpy as np
import matplotlib.pyplot as plt
import os,sys
from Preprocess import datasets
#import train_test as tt
from read_h5 import read_input,save_jets
from hist import hist,stack_hist
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras import optimizers
import tensorflow.python.keras.backbend as K


#set parameters
#these should be varied in order to determine optimal values
n_epochs = 6 #number of epochs to train the network
learning_rate = 0.005
log_interval = 15
#drop_rate_1 = 0.5
#drop_rate_2 = 0.3


#set a standard random seed for reproducible results
seed = 42
np.random.seed(seed)


#define the train and test sets
train_input, test_input, train_target, test_target = datasets()

print('\ntraining data format: input: ',train_input.shape,'\ttarget:',train_target.shape)
print('\ntesting target format: input: ',test_input.shape,'\ttarget:',test_target.shape)

#define the network
network = Sequential()
network.add(Dense(20, input_dim=18, kernel_initializer='normal', activation='relu'))
network.add(Dense(10, activation='relu'))
network.add(Dense(1, activation='linear'))
network.summary()

#define the optimizer and the learning rate
adam = optimizers.Adam(lr=learning_rate)

#define the resolution metric
def resolution(y_true,y_pred):
    return (y_true-y_pred)/y_true


#compile the network
network.compile(loss='mse', 
                optimizer=adam,
                metrics=[resolution])


#fit the model
history = model.fit(train_input, train_target,
                    epochs=20,  verbose=1,
                    batch_size=128, validation_split = 0.2, shuffle=True)


print(history.history.keys())

#plot the loss history
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('MSE Loss')
plt.legend(['Train Loss','Validation Loss'], loc='upper right')
plt.show()


#test the network
output = network.predict(test_input)

#plot an histogram of the output
hist(output, Pt, title='Network output')

#save the recopt info in a numpy array
dset = read_input('input.h5')
data_points = dset.shape[0]
RecoPt = save_jets(dset,0,data_points)

#plot an histogram if the stacked resolutions
final = []
#calculate the final resolution
res1 = (np.asarray(test_target) - np.asarray(output))/np.asarray(test_target)
final.append(res1)#append to array
#calculate resolution between Reco and output
res2 = (RecoPt - np.asarray(output))/RecoPt
final.append(res2)
#calculate resolution between Reco and target
res3 = (RecoPt - np.asarray(test_target))/RecoPt
final.append(res3)

#plot the stacked histogram of all the resolutions
legend = ['Target-Output','RecoPt-Output','Reco-Target']
stack_hist(final, resolution, legend=legend, loc='upper left')





