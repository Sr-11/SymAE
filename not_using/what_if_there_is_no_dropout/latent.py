from parameters import *
import numpy as np
def latent(model,MRA):
    X=MRA.X
    C=model.sym_encoder.predict(X)
    N=model.nui_encoder.predict(X)
    return C,N

