from parameters import *
import numpy as np
class redatuming():
    def __init__(self,model,MRA1,MRA2,t):
        self.model=model
        self.MRA1=MRA1
        self.MRA2=MRA2
        self.t=t
        X1=MRA1.X
        X2=MRA2.X
        coherent_1=model.sym_encoder.predict(X1)
        coherent_2=model.sym_encoder.predict(X2)
        nuisance_1=model.nui_encoder.predict(X1)
        nuisance_2=model.nui_encoder.predict(X2)
        merger_2_1 = model.latentcat(coherent_1,nuisance_2)
        merger_1_2 = model.latentcat(coherent_2,nuisance_1)
        Y21 = model.decoder.predict(merger_2_1)
        Y12 = model.decoder.predict(merger_1_2)
        Y11=model.predict(MRA1.X)
        Y22=model.predict(MRA2.X)
        self.C1_N1_input=MRA1.X[0,t,:]
        self.C1_N1_output=Y11[0,t,:]
        self.C2_N2_input=MRA2.X[0,t,:]
        self.C2_N2_output=Y22[0,t,:]
        self.C1_N2_virtual=Y21[0,t,:]
        self.C1_N2_synthetic=np.roll(MRA1.thetas[0,:],int(-MRA2.shifts[0,t]))
        self.C2_N1_virtual=Y12[0,t,:]
        self.C2_N1_synthetic=np.roll(MRA2.thetas[0,:],int(-MRA1.shifts[0,t]))
    def MSE(self):
        self.MSE_C1_N2=np.mean((self.C1_N2_virtual-self.C1_N2_synthetic)**2)
        self.MSE_C2_N1=np.mean((self.C2_N1_virtual-self.C2_N1_synthetic)**2)
        return self.MSE_C1_N2,self.MSE_C2_N1
        
    