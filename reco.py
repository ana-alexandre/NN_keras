#get reco pt hist

from hist import hist
from read_h5 import read_input,save_jets


#get recopt
dset=read_input('input.h5')
RecoPt = save_jets(dset,0,dset.shape[0])

hist(RecoPt,'recoPt',title = 'Reco Pt')


