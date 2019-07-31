##NN - preprocessing
#for network in keras

#reads in the h5 file
#normalizes the information
#and turns it into a torch tensor

import numpy as np
from Read_h5_hist import read_input,save_jets,hist#####!!!!!!this will need updating!!!!!
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

def datasets():
    
    #read in the h5 file into a dataset
    #this will also print the shape of the file
    #and the column names and data types
    dset = read_input('input.h5')
    data_points = dset.shape[0]

    #save the total number of columns
    #and the min and max values for the target column index
    no_of_columns = 22
    min_target = 4
    max_target = 7

    #create empty arrays to store targets and input
    targets = np.array([]).reshape(116655,0)
    inputs = np.array([]).reshape(116655,0)

    #seperate the dataset into input and targets
    for i in range(no_of_columns):

        temp = save_jets(dset,i,data_points)
        temp = np.reshape(temp, (116655,1))

        if i==min_target: #if the column is the TruthPt column
            targets = np.concatenate((targets,temp),axis=1)

        if i<min_target or i>max_target:
            inputs = np.concatenate((inputs,temp),axis=1)


    print('\ntargets:\n',targets[2405:2408],'\ninputs:\n',inputs[2405:2408])

    #normalise the inputs
    #without normalizing the bolean columns
    #so normalize each part seperately
    inputs_norm1 = normalize(inputs[:, [0,1,2,3,4,5,6,7]], axis=0)
    inputs_norm2 = normalize(inputs[:, [11,12,13,14]], axis=0)
    #join the arrays together
    args = (inputs_norm1, inputs[:,[8,9,10]],inputs_norm2,inputs[:,[15,16,17]])
    inputs_ = np.concatenate(args, axis=1)

    #print a histogram of the target distribution (Truth Pt)
    hist(targets,'TruthPt',title='targets')
    #print a few of the targets and the corresponding inputs
    #(the numbers were chosen so that one of the targets has a lepton present)
    print('targets: ',targets[2405:2410],'inputs: ',inputs[2405:2410])


    x_train,x_test,y_train,y_test = train_test_split(inputs,targets)

    return x_train,x_test,y_train,y_test


