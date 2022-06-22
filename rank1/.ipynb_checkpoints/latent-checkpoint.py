def latent(model,X):
    C=model.sym_encoder.predict(X)
    N=model.nui_encoder.predict(X)
    return C,N
