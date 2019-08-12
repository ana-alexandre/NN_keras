#plot the loss and accuracies for different dropout rates in the same plot
#plot the difference between test and train for different dropout rates

import sys,os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


n_epochs = 100
learning_rate = 0.001
layer1 = ('12','20','40','80','120','200','240','320')
layer2 = ('6','10','20','40','60','100','120','160')
#drop_rate = (0.5,0.3)
#arc = ('(4,8)','(10,20)','(20,40)','(6,12,24)','(4,8,12)','(6,12)')

#root filename
root = '~/NNkeras/results/ne_'


#create empty arrays to store values
train_losses = []
test_losses = []
train_resol = []
test_resol = [] 
targets = []
output = []


#read the files with 2 layers
n = 0

no_layers = 2
layer1 = ('12','20','40','80','120','200','240','320')
layer2 = ('6','10','20','40','60','100','120','160')
for no in range(len(layer1)):
    
    #filename
    file = 'ne_'+str(n_epochs)+'_lr_'+str(learning_rate)+'_nl_'+str(no_layers)+'_'+str(layer1[no])+'_'+str(layer2[no])+'_'
    
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

#read files with 3 layers
no_layers = 3
layer1 = ('12','30','80','120','240','320','300')
layer2 = ('8','20','40','80','120','160','200')
for no in range(len(layer1)):

    #filename
    file = 'ne_'+str(n_epochs)+'_lr_'+str(learning_rate)+'_nl_'+str(no_layers)+'_'+str(layer1[no])+'_'+str(layer2[no])+'_'

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

#read files with 4 layers
no_layers = 4
layer1 = ('32','80','120','320')
layer2 = ('16','40','80','160')
layer3 = ('8','20','40','80')
layer4 = ('4','10','20','40')
for no in range(len(layer1)):

    #filename
    file = 'ne_'+str(n_epochs)+'_lr_'+str(learning_rate)+'_nl_'+str(no_layers)+'_'+str(layer1[no])+'_'+str(layer2[no])+'_'+str(layer3[no])+'_'+str(layer4[no])+'_'

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


color = ['olive','indigo','orange','teal','green','crimson', 'turquoise','coral','purple','red','yellowgreen','black','gray','cyan','peru','darkgoldenrod','darkslateblue','firebrick','darkmagenta']
marker = ['s', '>','^','p','d','*','v','<']


legend = ('(12,6)','(20,10)','(40,20)','(80,40)','(120,60)','(200,100)','(240,120)','(320,160)','(12,8,4)','(30,20,10)','(80,40,20)','(120,80,40)','(240,120,60)','(320,160,80)','(300,200,100)','(32,16,8,4)','(80,40,20,10)','(120,80,40,20)','(320,160,80,40)')


#plot all the test losses in one graph
fig = plt.figure()

for i in range(0, n):
    plt.plot(test_losses[i],color=color[i], linewidth=1,label = legend[i])
    #plt.scatter(test_losses[i], marker=marker[i], s=25, facecolors='none', edgecolors=color[i], label = arc[i])

plt.legend(loc='upper right',title='Layers')
plt.xlabel('Number of epochs')
plt.ylabel('MSE Loss')
fig.savefig('layers2_losses.png')
plt.show()


#plot all the train losses in one graph
fig = plt.figure()

for i in range(0, n):
    plt.plot(train_losses[i],color=color[i], linewidth=1,label = legend[i])
    #plt.scatter(test_losses[i], marker=marker[i], s=25, facecolors='none', edgecolors=color[i], label = arc[i])

plt.legend(loc='upper right',title='Layers')
plt.xlabel('Number of epochs')
plt.ylabel('MSE Loss')
plt.ylim(700000000,900000000)
fig.savefig('layers2_train_losses.png')
plt.show()


#plot all the test losses in one graph
fig = plt.figure()

for i in range(7):
    plt.plot(test_losses[8:15][i],color=color[i], linewidth=1,label = legend[8:15][i])
    #plt.scatter(test_losses[i], marker=marker[i], s=25, facecolors='none', edgecolors=color[i], label = arc[i])

plt.legend(loc='upper right',title='Layers')
plt.xlabel('Number of epochs')
plt.ylabel('MSE Loss')
fig.savefig('layers3_losses.png')
plt.show()


#plot all the train losses in one graph
fig = plt.figure()

for i in range(7):
    plt.plot(train_losses[8:15][i],color=color[i], linewidth=1,label = legend[8:15][i])
    #plt.scatter(test_losses[i], marker=marker[i], s=25, facecolors='none', edgecolors=color[i], label = arc[i])

plt.legend(loc='upper right',title='Layers')
plt.xlabel('Number of epochs')
plt.ylabel('MSE Loss')
plt.ylim(700000000,900000000)
fig.savefig('layers3_train_losses.png')
plt.show()

#plot all the test losses in one graph
fig = plt.figure()

for i in range(4):
    plt.plot(test_losses[15:][i],color=color[i], linewidth=1,label = legend[15:][i])
    #plt.scatter(test_losses[i], marker=marker[i], s=25, facecolors='none', edgecolors=color[i], label = arc[i])

plt.legend(loc='upper right',title='Layers')
plt.xlabel('Number of epochs')
plt.ylabel('MSE Loss')
fig.savefig('layers4_losses.png')
plt.show()


#plot all the train losses in one graph
fig = plt.figure()

for i in range(4):
    plt.plot(train_losses[15:][i],color=color[i], linewidth=1,label = legend[15:][i])
    #plt.scatter(test_losses[i], marker=marker[i], s=25, facecolors='none', edgecolors=color[i], label = arc[i])

plt.legend(loc='upper right',title='Layers')
plt.xlabel('Number of epochs')
plt.ylabel('MSE Loss')
plt.ylim(700000000,900000000)
fig.savefig('layers4_train_losses.png')
plt.show()


#plot all the test losses in one graph
fig = plt.figure()

for i in range(4):
    plt.plot(test_losses[2:6][i],color=color[i], linewidth=1,label = legend[2:6][i])
    #plt.scatter(test_losses[i], marker=marker[i], s=25, facecolors='none', edgecolors=color[i], label = arc[i])
for i in range(3):
    plt.plot(test_losses[10:13][i],color=color[i+4], linewidth=1,label = legend[10:13][i])
    #plt.scatter(test_losses[i], marker=marker[i], s=25, facecolors='none', edgecolors=color[i], label = arc[i])
for i in range(2):
    plt.plot(test_losses[16:18][i],color=color[i+7], linewidth=1,label = legend[16:18][i])
    #plt.scatter(test_losses[i], marker=marker[i], s=25, facecolors='none', edgecolors=color[i], label = arc[i])
plt.legend(loc='upper left',title='Layers')
plt.xlabel('Number of epochs')
plt.ylabel('MSE Loss')
fig.savefig('layers_losses.png')
plt.show()


#plot all the train losses in one graph
fig = plt.figure()

for i in range(4):
    plt.plot(train_losses[2:6][i],color=color[i], linewidth=1,label = legend[2:6][i])
    #plt.scatter(test_losses[i], marker=marker[i], s=25, facecolors='none', edgecolors=color[i], label = arc[i])
for i in range(3):
    plt.plot(train_losses[10:13][i],color=color[i+4], linewidth=1,label = legend[10:13][i])
    #plt.scatter(test_losses[i], marker=marker[i], s=25, facecolors='none', edgecolors=color[i], label = arc[i])
for i in range(2):
    plt.plot(train_losses[16:18][i],color=color[i+7], linewidth=1,label = legend[16:18][i])
    #plt.scatter(test_losses[i], marker=marker[i], s=25, facecolors='none', edgecolors=color[i], label = arc[i])
plt.legend(loc='upper left',title='Layers')
plt.xlabel('Number of epochs')
plt.ylabel('MSE Loss')
plt.ylim(700000000,900000000)
fig.savefig('layers_train_losses.png')
plt.show()


