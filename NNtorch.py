
# coding: utf-8

# In[1]:


#NN

#Neural network for a regression problem
#Reads in reco and lep kinematics
#(plus checks if leptons are present and, if so, if they're an electron or a muon) of a b-jet
#Trains to predict the true kinematics of the jet

import numpy as np
import torch
import torchvision
import torch.utils.data as utils
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os,sys
from Preprocess import datasets,dataloaders
import train_test as tt
from Read_h5_hist import read_input,save_jets,hist


# In[2]:


#set parameters
#these should be varied in order to determine optimal values
n_epochs = 6 #number of epochs to train the network
learning_rate = 0.005
log_interval = 15
#drop_rate_1 = 0.5
#drop_rate_2 = 0.3


# In[3]:


#set a standard random seed for reproducible results
seed = 42
np.random.seed(seed)
torch.backends.cudnn.enabled = False
torch.manual_seed(seed)


# In[4]:


#set how many points to take for train and test set
total_size = 116655
train_size = 0.2*total_size
test_size = 0.8*total_size

#define the train and test sets
train_set, test_set = datasets(train_size,test_size) 

print('\ntraining data format: ',train_set)
print('\ntesting target format: ',test_set)


# In[5]:


#save the recopt info in a numpy array
dset = read_input('input.h5')
data_points = dset.shape[0]
RecoPt = save_jets(dset,0,data_points)


# In[6]:


#create train and test loaders
batch_size_train = 128
batch_size_test = 1000

#load the train loader
train_loader,test_loader = dataloaders(train_set,test_set,batch_size_train,batch_size_test)
#print info on the train and test loaders
print('\ntrain loader: ',train_loader,'\ntest loader: ',test_loader)


# In[7]:


#define the network as a class
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(18, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 1)

    def forward(self, x):

        #print(x.shape)
        x = F.leaky_relu(self.fc1(x))

        #print(x.shape)
        x = F.leaky_relu(self.fc2(x))

        x = self.fc3(x)
        return x #regression network: no softmax

network = Net()


# In[8]:


#define the optimizer and the learning rate
optimizer = optim.Adam(network.parameters(),
                      lr=learning_rate)


# In[9]:


#create empty arrays to keep track of losses
train_losses = []
train_acc = []
eval_losses = []
eval_acc = []
train_counter = []
test_losses = []
test_acc = []
final = []
outputs =[]
targets = []


# In[10]:


def train(epoch):

    for batch_idx, (data, target) in enumerate(train_loader):
        network.train()
        optimizer.zero_grad()
        output = network(data)
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()

        #calculate predictions and how many are correct
        #pred_train = output.data.max(1, keepdim=True)[1]

        if batch_idx % log_interval == 0:

            train_losses.append(loss.item())
            train_counter.append(
                    (batch_idx*batch_size_train) + ((epoch-1)*len(train_loader.dataset)))
            
            #calculate how well the network is doing
            
            #turn the torch output tensor into a numpy array
            out = []
            targets = []
            for i in range(output.shape[0]):
                out.append(output[i].detach().numpy()[0])
                targets.append(target[i].detach().numpy()[0])
            #calculate the accuracy for each element
            acc = np.divide((np.asarray(targets) - np.asarray(out)),np.asarray(targets))
            accuracy = np.average(acc)#take the average value
            train_acc.append(accuracy)#append to accuracy 
            
            print('\nTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} \tAvg resolution: {:.8f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item(),accuracy))

            train_eval()
            validate()


# In[11]:


def validate():
    #validate the network with a separate dataset
    # no network.eval() so that the dropped neurons are still dropped
    # to be able to compare the training and testing information
    test_loss = 0

    with torch.no_grad():
        for (data, target) in test_loader:
            output = network(data)

            test_loss += F.mse_loss(output, target, size_average=False).item()
            #pred = output.data.max(1, keepdim=True)[1]
            
            #turn the torch output tensor into a numpy array
            out = []
            targets = []
            for i in range(output.shape[0]):
                out.append(output[i].numpy()[0])
                targets.append(target[i].numpy()[0])
    #calculate the accuracy for each element
    acc = np.divide((np.asarray(targets) - np.asarray(out)),np.asarray(targets))
    accuracy = np.average(acc)#take the average value
    test_acc.append(accuracy)#append to accuracy array

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    print('\nTest set: Avg. loss: {:.4f}\tAvg resolution: {:.8f}'.format(test_loss,accuracy))


# In[12]:


def train_eval():
    #validate the network with a separate dataset
    # no network.eval() so that the dropped neurons are still dropped
    # to be able to compare the training and testing information
    #network.eval()
    train_loss = 0
    accuracy = 0
    
    with torch.no_grad():
        for (data, target) in train_loader:
            output = network(data)

            train_loss += F.mse_loss(output, target, size_average=False).item()
            #pred = output.data.max(1, keepdim=True)[1]
            
            #turn the torch output tensor into a numpy array
            out = []
            targets = []
            for i in range(output.shape[0]):
                out.append(output[i].numpy()[0])
                targets.append(target[i].numpy()[0])
    #calculate the accuracy for each element
    acc = np.divide((np.asarray(targets) - np.asarray(out)),np.asarray(targets))
    accuracy = np.average(acc)#take the average value
    eval_acc.append(accuracy)#append to accuracy array
            
    train_loss /= len(train_loader.dataset)
    eval_losses.append(train_loss)

    print('\nTrain set: Avg. loss: {:.4f}\tAvg resolution: {:.8f}'.format(train_loss,accuracy))


# In[13]:


def test():
    #test the network after it's trained
    network.eval()
    final_test_loss = 0
    #final = []
    accuracy = 0
    final_acc = []
    init_acc = []
    out = []
    outputs = []
    targets = []

    with torch.no_grad():
        for (data, target) in test_loader:
            output = network(data)

            final_test_loss += F.mse_loss(output, target, size_average=False).item()
            
            #turn the torch output tensor into a numpy array
            for i in range(output.shape[0]):
                outputs.append(output[i].numpy()[0])
                targets.append(target[i].numpy()[0])
            
            #pred = output.data.max(1, keepdim=True)[1]
            
    final_test_loss /= len(test_loader.dataset)
    final.append(final_test_loss)
    
    #calculate the accuracy for each element
    final_acc = (np.asarray(targets) - np.asarray(outputs))/np.asarray(targets)
    accuracy = np.average(final_acc)#take the average value
    final.append(accuracy)#append to accuracy array
    
    final.append(final_acc)#append all the accuracy values
    
    init_acc = (RecoPt - np.asarray(outputs))/RecoPt
    final.append(init_acc)#append all the accuracy values
    
    ideal_acc = (RecoPt - np.asarray(targets))/RecoPt
    final.append(ideal_acc)#append all the accuracy values
    
    print('\nTest set: Avg. loss: {:.4f}\tAvg resolution: {:.8f}'.format(final[0],final[1]))
    
    #hist(targets, 'Known truth Pt',title = 'Network Input')
    #hist(outputs, 'Predicted truth Pt',title = 'Network Output')
    #hist(RecoPt, 'Pt',title = 'Reco Pt')
    
    #hist(final[3],'(Reco-Pred):Reco')
    #hist(final[2], '(Truth-Pred):Truth')
    
    fig = plt.figure()
    
    plt.hist(final[2:], 500, stacked=True)
    plt.xlim(-10,2)
    plt.legend(['Target-output', 'Reco-output','Reco-target'], loc='upper right')
    fig.savefig('resolution_hist2.png')
    plt.show()
    


# In[14]:


#network begins training here
for epoch in range(1, n_epochs + 1): #loop over n epochs
    train(epoch)


# In[15]:


#plot the losses
fig = plt.figure()

#plot the mse loss
plt.plot(train_counter, train_losses, color='orange', linewidth=0.5)
#plt.scatter(train_counter, train_losses, marker="^", facecolors='none', edgecolors='orange')
plt.plot(train_counter, test_losses, color='red', linewidth=0.5)
#plt.scatter(train_counter, test_losses, marker="^", facecolors='none', edgecolors='red')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.title('Mean Square Error Loss')
plt.xlabel('Number of training examples seen')
plt.ylabel('MSE Loss')
plt.ylim(300000000,2000000000)
fig.savefig('loss_plot2.png')
plt.show()


# In[16]:


#finally, test the network with the full set of neurons learned
#i.e. no dropout
print('\nFinal result:')
test()


# In[17]:


hist(targets, 'Known truth Pt',title = 'Network Input')
hist(outputs, 'Predicted truth Pt',title = 'Network Output')
hist(RecoPt, 'Pt',title = 'Reco Pt')
    
plt.hist(final[2:], 500, stacked=True)
plt.xlim(-10,2)
plt.legend(['Target-output', 'Reco-output','Reco-target'], loc='upper right')
plt.show()


# In[ ]:


print (dir(network))
print (network.state_dict())


# In[ ]:


#save labels to csv files
root = '/Users/alex98/DESY/results/ne_'+str(n_epochs)+'_lg_'+str(log_interval)+'_lr_'+str(learning_rate)+'a1_'

fname_train_lss = root+'train_loss.csv'
fname_test_lss = root+'test_loss.csv'
fname_counter = root+'counter.csv'
fname_final = root+'final_result.csv'

np.savetxt(fname_counter,train_counter,delimiter=',')
np.savetxt(fname_train_lss,train_losses,delimiter=',')
np.savetxt(fname_test_lss,test_losses,delimiter=',')
np.savetxt(fname_final,final,delimiter=',')

