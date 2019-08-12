#NN
#keras

#Neural network for a regression problem
#Reads in reco and lep kinematics
#(plus checks if leptons are present and, if so, if they're an electron or a muon) of a b-jet
#Trains to predict the true kinematics of the jet

import numpy as np
import matplotlib.pyplot as plt
import os,sys
from preprocess import datasets
from read_h5 import read_input,save_jets
from hist import hist,stack_hist
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras import optimizers
from tensorflow.python.keras import backend as K


#set parameters
#these should be varied in order to determine optimal values
n_epochs = 200 #number of epochs to train the network
learning_rate = 0.005
batch_size = 128
no_layers = 3
layer1 = 120
layer2 = 80
layer3 = 40
#drop_rate_1 = 0.5
#drop_rate_2 = 0.3


#set a standard random seed for reproducible results
seed = 42
np.random.seed(seed)


#define the train and test sets
train_input, test_input, train_target, test_target = datasets('input.h5')

print('\ntraining data format: input: ',train_input.shape,'\ttarget:',train_target.shape)
print('\ntesting target format: input: ',test_input.shape,'\ttarget:',test_target.shape)

#define the network
model = Sequential()
model.add(Dense(layer1, input_dim=18, kernel_initializer='normal', activation='relu'))
model.add(Dense(layer2, activation='relu'))
model.add(Dense(layer3, activation='relu'))
#network.add(Dense(layer4, activation='relu'))
model.add(Dense(1, activation='linear'))
model.summary()

#define the optimizer and the learning rate
adam = optimizers.Adam(lr=learning_rate)

#define the resolution metric
def resolution(y_true,y_pred):
    return (y_true-y_pred)/y_true


#compile the network
model.compile(loss='mse', 
                optimizer=adam,
                metrics=[resolution])


#fit the model
history = model.fit(train_input, train_target,
                    epochs=n_epochs,  verbose=1,
                    batch_size=batch_size, validation_split = 0.2, shuffle=True)


print(history.history.keys())

#plot the loss history
fig = plt.figure()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('MSE Loss')
plt.legend(['Train Loss','Validation Loss'], loc='upper right')
fig.savefig('loss_plot.png')
plt.show()


#test the network
output = model.predict(test_input)

#plot an histogram of the output
hist(output, 'Pt', title='Network output')


#save labels to csv files
root = 'ne_'+str(n_epochs)+'_lr_'+str(learning_rate)+'_nl_'+str(no_layers)+'_'+str(layer1)+'_'+str(layer2)+'_'+str(layer3)+'_'

# serialize model to JSON
model_json = model.to_json()
with open(root+"model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(root+"model.h5")
print("Saved model to disk")


fname_train_lss = root+'train_loss.csv'
fname_test_lss = root+'val_loss.csv'
fname_train_res = root+'train_res.csv'
fname_test_res = root+'val_res.csv'
fname_output = root+'output.csv'
fname_target = root+'target.csv'

np.savetxt(fname_train_lss,history.history['loss'],delimiter=',')
np.savetxt(fname_test_lss,history.history['val_loss'],delimiter=',')
np.savetxt(fname_train_res,history.history['resolution'],delimiter=',')
np.savetxt(fname_test_res,history.history['val_resolution'],delimiter=',')
np.savetxt(fname_output,np.asarray(output),delimiter=',')
np.savetxt(fname_target,np.asarray(test_target), delimiter=',')


del model
K.clear_session()
import gc
gc.collect()
