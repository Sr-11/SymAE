import numpy as np
import math
import matplotlib.pyplot as plt
import random
class generate():
    def __init__(self, states, nuisances, nx, nt, replace=1):
        '''
        replace : int
            If replace==0, without replacement everywhere. Must $n_x*n_t <= n_1*n_2$ and $n_t <= n_2$.  
            If replace==1, X[i,:] have different nuisances, but each block in D can appear multiple times in X.  
            If replace==2, with replacement everywhere.
        '''
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
        selected_times = np.zeros((n1,n2), dtype='int')
        if replace == 0:
            assert nx*nt<=n1*n2 and nt<=n2, 'X is larger than D'
            for i in range(nx):
                e = np.random.choice(waiting_states)
                X_states[i] = e
                for t in range(nt):
                    c = np.random.randint(0,len(waiting_samples[e]))
                    j = waiting_samples[e][c]
                    X_nuisances[i,t] = j
                    X[i,t,0] = D[e,j]
                    selected_times[e,j] += 1
                    waiting_samples[e].pop(c)
                if len(waiting_samples[e]) < nt:
                     waiting_states.remove(e)
        if replace == 1:
            for i in range(nx):
                e = np.random.choice(waiting_states)
                X_states[i] = e
                c = np.random.choice(range(n2),nt,replace=False)
                for t in range(nt):
                    j = c[t]
                    X_nuisances[i,t] = j
                    X[i,t,0] = D[e,j]
                    selected_times[e,j] += 1
        if replace == 2:
            for i in range(nx):
                e = np.random.choice(range(n1))
                X_states[i] = e
                for t in range(nt):
                    c = np.random.randint(0,n2)
                    X_nuisances[i,t] = c
                    X[i,t,0] = D[e,c]        
                    selected_times[e,c] += 1
        self.D=D
        self.X = X
        self.X_states = X_states
        self.X_nuisances = X_nuisances
        self.waiting_samples=waiting_samples
        self.selected_times=selected_times