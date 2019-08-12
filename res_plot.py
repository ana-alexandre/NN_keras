#plot the resolution histograms


#read in code
#blah blah blah
output = []
target = []
#blah blah blah



#save the recopt info in a numpy array
dset = read_input('input.h5')
data_points = dset.shape[0]
RecoPt = save_jets(dset,0,data_points)

#plot an histogram if the stacked resolutions
final = []
#calculate the final resolution
res1 = (target - output)/target
final.append(res1)#append to array
#calculate resolution between Reco and output
res2 = (RecoPt - output)/RecoPt
final.append(res2)
#calculate resolution between Reco and target
res3 = (RecoPt - test_target)/RecoPt
final.append(res3)

#plot the stacked histogram of all the resolutions
legend = ['Target-Output','RecoPt-Output','Reco-Target']
stack_hist(final, 'resolution', legend=legend, loc='upper left')





