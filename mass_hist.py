#draws ai mass histogram from the kinematics

import ROOT
from ROOT.Math import PtEtaPhiMVector as LV
import numpy as np

out1=np.loadtxt("BQuark1JetOutPt.csv",dtype='float')
out2=np.loadtxt("BQuark2JetOutPt.csv",dtype='float')

#get the tree from the ROOT file
f = ROOT.TFile("input.root")
t = f.Get("AnaTree")

# Prepare the histogram to fill
h_truthM = ROOT.TH1F("truthM", ";p_{T} (Truth) [MeV]", 100, 0, 200e3)
h_recoM = ROOT.TH1F("recoM", ";p_{T} (Reco) [MeV]", 100, 0, 200e3)
h_outM = ROOT.TH1F("outM", ";p_{T} (NN Output) [MeV]", 100, 0, 200e3)

# GetEntries returns the total number of entries in the tree
for iEntry in range(t.GetEntries() ):
    # Now fill the histogram from the value from the TTree
    t.GetEntry(iEntry)
    
    #get the truth kinematics for both jets
    tpt1 = t.BQuark1JetTruthPt
    teta1 = t.BQuark1JetTruthEta
    tphi1 = t.BQuark1JetTruthPhi
    tm1 = t.BQuark1JetTruthMass
    p4truth1 = LV(tpt1,teta1,tphi1,tm1)
    
    tpt2 = t.BQuark2JetTruthPt
    teta2 = t.BQuark2JetTruthEta
    tphi2 = t.BQuark2JetTruthPhi
    tm2 = t.BQuark2JetTruthMass
    p4truth2 = LV(tpt2,teta2,tphi2,tm2)
    
    #sum the 4-vectors from the two jets
    p4truth = p4truth1 + p4truth2
    #fill the histogram with the mass from the combined 4-vector
    h_truthM.Fill(p4truth.M())

    #get the rceo kinematics for both jets
    rpt1 = t.BQuark1JetRecoPt
    reta1 = t.BQuark1JetRecoEta
    rphi1 = t.BQuark1JetRecoPhi
    rm1 = t.BQuark1JetRecoMass
    p4reco1 = LV(rpt1,reta1,rphi1,rm1)

    rpt2 = t.BQuark2JetRecoPt
    reta2 = t.BQuark2JetRecoEta
    rphi2 = t.BQuark2JetRecoPhi
    rm2 = t.BQuark2JetRecoMass
    p4reco2 = LV(rpt2,reta2,rphi2,rm2)

    #sum the 4-vectors from the two jets
    p4reco = p4reco1 + p4reco2
    #fill the histogram with the mass from the combined 4-vector
    h_recoM.Fill(p4reco.M())
    
    #use model to predict outPt
    opt1 = out1[iEntry]
    p4out1 = LV(opt1,reta1,rphi1,rm1)
    opt2 = out2[iEntry]
    p4out2 = LV(opt2,reta2,rphi2,rm2)
    
    if iEntry % 1500 == 0:
        print('\n',opt1)
        print(tpt1)
        print(rpt1)

    #sum the 4-vectors from the two jets
    p4out = p4out1 + p4out2
    #fill the histogram with the mass from the combined 4-vector
    h_outM.Fill(p4out.M())

c = ROOT.TCanvas()
h_truthM.Draw()
c.Print("truthM_hist.pdf")
c = ROOT.TCanvas()
h_recoM.Draw()
c.Print("recoM_hist.pdf")
c = ROOT.TCanvas()
h_outM.Draw()
c.Print("outM_hist.pdf")


