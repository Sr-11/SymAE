import numpy as np
def rho(theta,tau):
    assert len(theta)==len(tau)
    smallest=np.inf
    l_flag=0
    for l in range(len(theta)):
        roll_tau=np.roll(tau, l)
        if smallest > np.linalg.norm(theta-roll_tau):
            smallest = np.linalg.norm(theta-roll_tau)
            l_flag = l
    return smallest, l_flag