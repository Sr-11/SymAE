import numpy as np
def rho(theta,tau):
    assert len(theta)==len(tau)
    smallest=np.inf
    for l in range(len(theta)):
        roll_tau=np.roll(tau, l)
        smallest=min([smallest,np.linalg.norm(theta-roll_tau)])
    return smallest