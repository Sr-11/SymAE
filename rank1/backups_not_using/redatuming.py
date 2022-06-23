import numpy as np
from tqdm import trange
def redatuming(model,data,i,i_prime):
    '''
    Parameters
    -----------
    model : SymAE defined in symae_codel.py
    data : generate defined in generated.py
    i : int
    i_prime : int
    
    Returns
    ----------
    Z_hat : np.ndarray
        Z_hat.shape = ()
    '''
    X = data.X
    X_states = data.X_states
    X_nuisances = data.X_nuisances
    nx, nt, _ = X.shape
    Z=np.empty((nx,nx,nt))
    Z_hat=np.empty((nx,nx,nt))
    Cs=model.sym_encoder.predict(X, verbose=0)
    Ns=model.nui_encoder.predict(X, verbose=0)
    coherent_i_prime=Cs[i_prime:i_prime+1,:]
    nuisance_i_j=Ns[i_prime:i_prime+1,:]
    merger = model.latentcat(coherent_i_prime,nuisance_i_j)
    Z = X_states[i_prime]*X_nuisances[i,:] 
    Z_hat = model.decoder.predict(merger, verbose=0)[0,:,0]
    return  Z_hat, Z