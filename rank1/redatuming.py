from parameters import *
import numpy as np
from tqdm import trange
def redatuming(model,data):
    X = data.X
    X_states = data.X_states
    X_nuisances = data.X_nuisances
    nx, nt, _ = X.shape
    Z=np.empty((nx,nx,nt))
    Z_hat=np.empty((nx,nx,nt))
    Cs=model.sym_encoder.predict(X, verbose=0)
    Ns=model.nui_encoder.predict(X, verbose=0)
    for i in trange(5):
        for i_prime in range(5):
            coherent_i_prime=Cs[i_prime:i_prime+1,:]
            nuisance_i_j=Ns[i_prime:i_prime+1,:]
            merger = model.latentcat(coherent_i_prime,nuisance_i_j)
            Z_hat[i,i_prime,:]=model.decoder.predict(merger, verbose=0)[0,:,0]
            Z[i,i_prime,:]=X_states[i_prime]*X_nuisances[i,:] 
    return Z,Z_hat