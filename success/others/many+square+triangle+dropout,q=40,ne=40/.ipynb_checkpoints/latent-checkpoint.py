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
    C : np.array
        C.shape = N*p
    N : np.array
        N.shape = N*(q*nt)
    '''
    X=MRA.X
    C=model.sym_encoder.predict(X,verbose=0)
    N=model.nui_encoder.predict(X,verbose=0)
    return C,N

