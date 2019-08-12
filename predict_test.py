#test the predict code
#which reads in kine,atics form root file
#and use keras trained model to predict the "truth" info
#but now with the h5 data used to train th network


from hist import hist
from read_h5 import read_input,save_jets
import uproot
import numpy as np
from tensorflow.python.keras import models
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from preprocess import datasets

#load the trained model
with open("ne_200_lr_0.005_nl_3_120_80_40_model.json") as fp:
    model_json = fp.read()
model = models.model_from_json(model_json)
#load the weights
model.load_weights("ne_200_lr_0.005_nl_3_120_80_40_model.h5")

#try with the h5 data
#define the train and test sets
train_input, test_input, train_target, test_target = datasets('input.h5')

#calculate the saved model 
out_train = model.predict(train_input)
out_test = model.predict(test_input)

hist(out_train,'pt (train)')
hist(out_test,'pt (test)')

#now try with the h5 file corresponding to the root file
#define the train and test sets
train_input, test_input, train_target, test_target = datasets('test_input.h5')

#calculate the saved model 
out_train = model.predict(train_input)
out_test = model.predict(test_input)

hist(out_train,'pt (train_h5)')
hist(out_test,'pt (test_h5)')


#get the data form the ROOT file
t = uproot.open("input.root")["AnaTree"]
print(t.keys())

#get the kinematics for reco and leptons to apply the model to 
#and normalize if necessary
recopt1 = np.reshape(t.array("BQuark1JetRecoPt"),(35652,1))
recoeta1 = np.reshape(t.array("BQuark1JetRecoEta"),(35652,1))
recophi1 = np.reshape(t.array("BQuark1JetRecoPhi"),(35652,1))
recom1 = np.reshape(t.array("BQuark1JetRecoMass"),(35652,1))
lep1pt1 = np.reshape(t.array("BQuark1JetLep1Pt"),(35652,1))
lep1eta1 = np.reshape(t.array("BQuark1JetLep1Eta"),(35652,1))
lep1phi1 = np.reshape(t.array("BQuark1JetLep1Phi"),(35652,1))
lep1m1 = np.reshape(t.array("BQuark1JetLep1Mass"),(35652,1))
lep1ele1 = np.reshape(t.array("BQuark1JetLep1IsElectron"),(35652,1))
lep1mu1 = np.reshape(t.array("BQuark1JetLep1IsMuon"),(35652,1))
lep1pr1 = np.reshape(t.array("BQuark1JetLep1IsPresent"),(35652,1))
lep2pt1 = np.reshape(t.array("BQuark1JetLep2Pt"),(35652,1))
lep2eta1 = np.reshape(t.array("BQuark1JetLep2Eta"),(35652,1))
lep2phi1 = np.reshape(t.array("BQuark1JetLep2Phi"),(35652,1))
lep2m1 = np.reshape(t.array("BQuark1JetLep2Mass"),(35652,1))
lep2ele1 = np.reshape(t.array("BQuark1JetLep2IsElectron"),(35652,1))
lep2mu1 = np.reshape(t.array("BQuark1JetLep2IsMuon"),(35652,1))
lep2pr1 = np.reshape(t.array("BQuark1JetLep2IsPresent"),(35652,1))

inp1 = np.concatenate((recopt1,recoeta1,recophi1,recom1,lep1pt1,lep1eta1,lep1phi1,lep1m1), axis=1)
inp2 = np.concatenate((lep2pt1,lep2eta1,lep2phi1,lep2m1), axis=1)

inp1 = normalize(inp1, axis=0)
inp2 = normalize(inp2, axis=0)

hist(inp1[0],'recopt')
hist(inp1[1],'recoeta')
hist(inp1[2],'recophi')
hist(inp1[3],'recom')
hist(inp1[4],'lep1pt')
hist(inp1[5],'lep1eta')
hist(inp1[6],'lep1phi')
hist(inp1[7],'lep1m')
hist(lep1ele1,'lep1ism')
hist(lep1mu1,'lep1isp')
hist(lep1pr1,'lep1ise')
hist(inp2[0],'lep2pt')
hist(inp2[1],'lep2eta')
hist(inp2[2],'lep2phi')
hist(inp2[3],'lep2m')
hist(lep2ele1,'lep2ism')
hist(lep2mu1,'lep2isp')
hist(lep2pr1,'lep2ise')


#second jet
recopt2 = np.reshape(t.array("BQuark2JetRecoPt"),(35652,1))
recoeta2 = np.reshape(t.array("BQuark2JetRecoEta"),(35652,1))
recophi2 = np.reshape(t.array("BQuark2JetRecoPhi"),(35652,1))
recom2 = np.reshape(t.array("BQuark2JetRecoMass"),(35652,1))
lep1pt2 = np.reshape(t.array("BQuark2JetLep1Pt"),(35652,1))
lep1eta2 = np.reshape(t.array("BQuark2JetLep1Eta"),(35652,1))
lep1phi2 = np.reshape(t.array("BQuark2JetLep1Phi"),(35652,1))
lep1m2 = np.reshape(t.array("BQuark2JetLep1Mass"),(35652,1))
lep1ele2 = np.reshape(t.array("BQuark2JetLep1IsElectron"),(35652,1))
lep1mu2 = np.reshape(t.array("BQuark2JetLep1IsMuon"),(35652,1))
lep1pr2 = np.reshape(t.array("BQuark2JetLep1IsPresent"),(35652,1))
lep2pt2 = np.reshape(t.array("BQuark2JetLep2Pt"),(35652,1))
lep2eta2 = np.reshape(t.array("BQuark2JetLep2Eta"),(35652,1))
lep2phi2 = np.reshape(t.array("BQuark2JetLep2Phi"),(35652,1))
lep2m2 = np.reshape(t.array("BQuark2JetLep2Mass"),(35652,1))
lep2ele2 = np.reshape(t.array("BQuark2JetLep2IsElectron"),(35652,1))
lep2mu2 = np.reshape(t.array("BQuark2JetLep2IsMuon"),(35652,1))
lep2pr2 = np.reshape(t.array("BQuark2JetLep2IsPresent"),(35652,1))

inpu1 = np.concatenate((recopt2,recoeta2,recophi2,recom2), axis=1)
inpu1 = normalize(inpu1, axis=0)

inpu2 = np.concatenate((lep1pt2,lep1eta2,lep1phi2,lep1m2), axis=1)
inpu2 = normalize(inpu2, axis=0)

inpu3 = np.concatenate((lep2pt2,lep2eta2,lep2phi2,lep2m2), axis=1)
inpu3 = normalize(inpu3, axis=0)


args1 = (inp1,lep1ele1,lep1mu1,lep1pr1,inp2,lep2ele1,lep2mu1,lep2pr1)
args2 = (inpu1,inpu2,lep1ele2,lep2mu2,lep1pr2,inpu3,lep2ele2,lep2mu2,lep2pr2)

in1 = np.concatenate(args1, axis=1)
in2 = np.concatenate(args2, axis=1)

#hist(recopt1, 'pt (root_reco)')

print(in1[:20])

#use model to predict output pt
out1 = model.predict(in1)
out2 = model.predict(in2)

print(out1[:20])

hist(out1,'pt')
hist(out2,'pt2')


