from parameters import *
import numpy as np
def redatuming(model,X_states,X_nuisances,X):
    nx,_,nt=X.shape
    Z=np.empty((nx,nx,nt))
    Z_hat=np.empty((nx,nx,nt))
    for i in range(nx):
        for i_prime in range(nx):
            for j in range(nt):
                coherent_i_prime=model.sym_encoder.predict(X[i_prime,:])
                nuisance_i_j=model.nui_encoder.predict(X[i,:])[j]
                merger = model.latentcat(coherent_i_prime,nuisance_i_j)
                Z_hat[i,i_prime,j]=model.decoder.predict(merger)
                Z[i,i_prime,j]=X_states[i_prime]*X_nuisances[j]    
    return Z,Z_hat