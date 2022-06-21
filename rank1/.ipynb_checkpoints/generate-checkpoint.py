import numpy as np
import math
import matplotlib.pyplot as plt
import random
class generate():
    def __init__(self,states,nuisances,nx,nt,outer_replace=False):
        states = np.copy(states)
        nuisances = np.copy(nuisances)
        n1 = len(states)
        n2 = len(nuisances)
        X = np.empty((nx,nt,1))
        X_states = np.empty(nx, dtype='int')
        X_nuisances = np.empty((nx,nt), dtype='int')
        D = np.tensordot(states,nuisances,0)
        waiting_samples = [list(range(n2)) for i in range(n1)]
        waiting_states = list(range(n1))
        for i in range(nx):
            e = np.random.choice(waiting_states)
            X_states[i] = e
            for t in range(nt):
                c = np.random.randint(0,len(waiting_samples[e]))
                j = waiting_samples[e][c]
                X_nuisances[i,t] = j
                X[i,t,0] = D[e,j]
                if outer_replace == False:
                    waiting_samples[e].pop(c)
            if len(waiting_samples[e]) < nt:
                 waiting_states.remove(e)
        self.D=D
        self.X = X
        self.X_states = X_states
        self.X_nuisances = X_nuisances
        self.waiting_samples=waiting_samples