from parameters import *
import numpy as np
def redatuming(model,X1,X2,t):
    '''
    Redatuming

    Parameters
    ----------
    model : SymAE
        tensorflow model defined in symae_model
    X1 : np.ndarray
        X1.shape=(d,nt,1)
    X2 : np.ndarray
        X2.shape=(d,nt,1)
    t : int
        The \tau in the paper.
        We focus on X1[:,t,0] and X2[:,t,0]
        
    Returns
    ----------
    Y_2_1 : np.ndarray
        Y_2_1.shape=(d,nt,1)
        gengerated by coherent code from X1 and nuisance code from X2
    Y_1_2 : np.ndarray
        gengerated by coherent code from X2 and nuisance code from X1
    '''
    coherent_1=model.sym_encoder.predict(X1)
    coherent_2=model.sym_encoder.predict(X2)
    nuisance_1=model.nui_encoder.predict(X1)
    nuisance_2=model.nui_encoder.predict(X2)
    merger_2_1 = model.latentcat(coherent_1,nuisance_2)
    merger_1_2 = model.latentcat(coherent_2,nuisance_1)
    Y_2_1 = model.decoder.predict(merger_2_1)
    Y_1_2 = model.decoder.predict(merger_1_2)
    return Y_2_1,Y_1_2
    
    