from parameters import *
import numpy as np
def latent(model,MRA):
    '''
    Obtain latent coherent code and latent nuisance code of data
    
    Parameters
    ----------
    model : class SymAE
        The SymAE
    MRA : class MRA_generate
        MRA.X is the input to SymAE
    
    Returns
    ----------
    C[0,:] : np.array
        1-dimensional 
    N[0,:] : np.array
        1-dimensional
    '''
    X=MRA.X
    C=model.sym_encoder.predict(X)
    N=model.nui_encoder.predict(X)
    return C[0,:],N[0,:]

