#plot the loss and accuracies for different dropout rates in the same plot
#plot the difference between test and train for different dropout rates

import sys,os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from hist import hist,stack_hist
from read_h5 import read_input,save_jets

n_epochs = 500
learning_rate = 0.001
no_layers = 4
layer1 = 120
layer2 = 80
layer3 = 40
#drop_rate = (0.5,0.3)


#filename
file = 'ne_'+str(n_epochs)+'_lr_'+str(learning_rate)+'_nl_'+str(no_layers)+'_'+str(layer1)+'_'+str(layer2)+'_'+str(layer3)+'_'

train_loss=np.loadtxt(file+"train_loss.csv",dtype='float')
test_loss=np.loadtxt(file+"val_loss.csv",dtype='float')
output = np.loadtxt(file+'output.csv',dtype='float')
train_res=np.loadtxt(file+"train_res.csv",dtype='float')
test_res=np.loadtxt(file+"val_res.csv",dtype='float')
targets = np.loadtxt(file+'target.csv',dtype='float')


#plot the test and train losses in one graph
fig = plt.figure()

plt.plot(test_loss, '--', color='olive', linewidth=1)
#plt.scatter(test_losses[i], marker=marker[i], s=25, facecolors='none', edgecolors='olive')
plt.plot(train_loss, color='darkblue', linewidth=1)
#plt.scatter(test_losses[i], marker=marker[i], s=25, facecolors='none', edgecolors='darkblue')
plt.legend(['Validation', 'Training'],loc='upper right')
plt.xlabel('Number of epochs')
plt.ylabel('MSE Loss')
fig.savefig('final_losses.png')
plt.show()


#plot the test and train resolution in one graph
fig = plt.figure()

plt.plot(test_res, '--', color='olive', linewidth=1)
#plt.scatter(test_losses[i], marker=marker[i], s=25, facecolors='none', edgecolors='olive')
plt.plot(train_res, color='darkblue', linewidth=1)
#plt.scatter(test_losses[i], marker=marker[i], s=25, facecolors='none', edgecolors='darkblue')
plt.legend(['Validation', 'Training'],loc='upper right')
plt.xlabel('Number of epochs')
plt.ylabel('Resolution')
fig.savefig('final_res.png')
plt.show()


#plot an histogram of the targets
hist(targets, 'Pt', title='Targets')

#plot an histogram of the output
hist(output, 'Pt', title='Network output')

#get the recopt in order to make the stacked histogram of all resolutions
dset=read_input('input.h5')
RecoPt = save_jets(dset,0,dset.shape[0])

#get the reco resolutions
res = []
res1 = (targets - output)/targets
res.append(res1)
res2 = (RecoPt[:29164] - output)/RecoPt[:29164]
res.append(res2)
res3 = (RecoPt[:29164] - targets)/RecoPt[:29164]
res.append(res3)

#draw the stacked histogram
stack_hist(res,'resolution',legend = ['Target - Output','Reco - Output','Reco - Target'])



