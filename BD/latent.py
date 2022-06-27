def latent(model,MRA):
    X=MRA.X
    C=model.sym_encoder.predict(X,verbose=0)
    N=model.nui_encoder.predict(X,verbose=0)
    return C,N

