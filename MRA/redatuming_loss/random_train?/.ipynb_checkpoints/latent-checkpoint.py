from parameters import *
def latent(model, X):
    return model.encoder(X)[:, p:p+q*nt].reshape(-1,q)
