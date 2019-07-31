#read input.h5
#save data to separate arrays

import h5py
import numpy as np


def read_input(file_name):
    #this function will read the h5 file
    #and put each variable into one numpy array

    #get all the data into 'dset'
    f = h5py.File(file_name,'r')
    dset = np.asarray(f['jets'])

    #array = np.zeros( [f.get(self.

    #print shape and type of info
    data_points = dset.shape[0]
    data_type = dset.dtype
    print('\nNo of data points: ',data_points,'\nData type:',data_type)

    return dset

def save_jets(dset,column_no,data_points):
    #save the info from one colum of the jets file to a numpy array

    #create an empty array to store info
    columnx = []

    for i in range(data_points): #note:this part of the code only works for the specific 'jets' in input.h5 case
        columnx.append(dset[i][column_no])

    return columnx

