#plot the loss and accuracies for different dropout rates in the same plot
#plot the difference between test and train for different dropout rates

import sys,os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


#root filename
root = '~/NNkeras/results/ne_'


n_epochs = 100
learning_rate = ('5e-05','0.0001','0.0005','0.001','0.005','0.01','0.05','0.1')
no_layers = 2
layer1 = 20
layer2 = 10
#drop_rate = ('0.5','0.3')

#create empty arrays to store values
train_losses = []
test_losses = []
output = []
train_resol = []
test_resol = []
targets = []



#read the file
n = 0

for rate in range(len(learning_rate)):
    
    #filename
    file = 'ne_'+str(n_epochs)+'_lr_'+str(learning_rate[rate])+'_nl_'+str(no_layers)+'_'+str(layer1)+'_'+str(layer2)+'_'
 
    train_loss=np.loadtxt(file+"train_loss.csv",dtype='float')
    test_loss=np.loadtxt(file+"val_loss.csv",dtype='float')
    out = np.loadtxt(file+'output.csv',dtype='float')
    train_res=np.loadtxt(file+"train_res.csv",dtype='float')
    test_res=np.loadtxt(file+"val_res.csv",dtype='float')
    target = np.loadtxt(file+'target.csv',dtype='float')
        
    #now append the results for each learning rate value to a 3D vector
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    output.append(out)
    train_resol.append(train_loss)
    test_resol.append(test_loss)
    targets.append(target)

    n += 1


color = ['olive','indigo','orange','turquoise','black','crimson', 'coral','teal','purple','red','yellowgreen','black','gray','cyan']
marker = ['s', 'd','v','p','^','*','>','<']


#plot all the test losses in one graph
fig = plt.figure()

for i in range(0, n):
    plt.plot(test_losses[i],color=color[i], linewidth=0.8,label = learning_rate[i])
#    plt.scatter(test_losses[i], marker=marker[i], s=20, facecolors='none', edgecolors=color[i], label = learning_rate[i])

#plt.plot(test_losses[0],color=color[i], linewidth=0.8,label = learning_rate[i])
plt.legend(loc='upper right',title='Learning Rates')
plt.xlabel('Number of epochs')
plt.ylabel('MSE loss')
plt.ylim(700000000,1000000000)
#plt.xlim(0,100000)
fig.savefig('learning_rate_losses.png')
plt.show()


#plot all the train losses in one graph
fig = plt.figure()

for i in range(0, n):
    plt.plot(train_losses[i],color=color[i], linewidth=0.8,label = learning_rate[i])
#    plt.scatter(test_losses[i], marker=marker[i], s=20, facecolors='none', edgecolors=color[i], label = learning_rate[i])
plt.legend(loc='upper right',title='Learning Rates')
plt.xlabel('Number of epochs')
plt.ylabel('MSE loss')
plt.ylim(700000000,1000000000)
#plt.xlim(0,100000)
fig.savefig('learning_rate_train_losses.png')
plt.show()



