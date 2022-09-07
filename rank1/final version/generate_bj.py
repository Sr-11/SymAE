import numpy as np
import math
import matplotlib.pyplot as plt
import random
class generate():
    def __init__(self, states, nuisances, nx, nt, replace=1):
        '''
        Parameters
        ------------
        state : np.ndarray
            states.shape=(n1,)
            All states.
        nuisances : np.ndarray
            nuisances.shape=(n2,)
            All nuisances.
        nx : int
            n_X in the paper.
        nt : int
            n_tau in the paper.
        replace : int
            If replace==0, without replacement everywhere. Must have $n_x*n_t <= n_1*n_2$ and $n_t <= n_2$.  
            If replace==1, X[i,:] have different nuisances, but each block in D can appear multiple times in X.  
            If replace==2, with replacement everywhere. ???

        
        Yields
        -----------
        self.X : np.ndarray
            X.shape = (nx,nt,1)
        self.D : np.ndarray
            D.shape = (n1,n2)
        self.X_states : np.ndarray
            X_states.shape= (nx,)
            Store the subscripts of states in X. 
            i.e. X_i has state a_{X_states[i]}
        self.X_nuisances : np.ndarray    
            X_nuisances.shape= (nx,nt)
            Store the subscripts of nuisances in X. 
            i.e. X_i[j] has nuisance b_{X_nuisances[i,j]}       
        '''
        
        # states and nuisance lists
        states = np.copy(states)
        nuisances = np.copy(nuisances)
        n1 = len(states)
        n2 = len(nuisances)
        
        # initializer X
        X = np.empty((nx,nt,1))
        X_states = np.empty(nx, dtype='int')
        X_nuisances = np.empty((nx,nt), dtype='int')
        
        # on fait D = states*nuisances^T (outer product), car 0 en argument
        # Donc D c'est les données.
        D = np.tensordot(states, nuisances, 0)
        
        # ???
        # waiting_samples = [[k for k in range(n2)] for i in range(n1)]
        # liste de listes, où chaque ligne correspond à une liste des indices [n2]
        # rappel: les lignes de X sont des états. Donc, pour chaque etat dans X, 
        # on a un vecteur de nuisances. Et ces nuisances il y en a au plus n2, et sont
        # donc piochés parmi les n2 nuisances totales. Donc, pour chaque ligne dans X, 
        # on aura nt nuisances, qui vont devoir etre piochés aleatoirement. On doit suivre
        # qui a été pioché, et donc c'est pour ça qu'on crée une liste waiting_samples pour suivre
        # lesquelles des nuisances sont toujours disponibles.
        # Donc, waiting samples ce sont des listes de listes de INDICES correspondant initialement
        # aux indices de nuissances disponibles dans D.
        # ensuite, au fur et a mesure, vu qu'on va piocher dans D, ces listes d'indices vont se retrecir.
        waiting_samples = [list(range(n2)) for i in range(n1)]
        waiting_states = list(range(n1))
        selected_times = np.zeros((n1,n2), dtype='int')
        
        # without replacement of blocks per coherent state
        if replace == 0:
            # pour toute ligne i dans X, on pioche epsilon_i dans [n1]
            for i in range(nx):
                e = np.random.choice(waiting_states)     # e = epsilon_i
                X_states[i] = e                          # juste pour documenter
                
                # maintenant, on va piocher des nuisances
                for t in range(nt):      # t c'est tau, dans [n_tau], indice des nuisances
                    
                    # on pioche un indice dans waiting_samples[e]. Notons que 
                    # initalement (t=0), waiting_samples[e] correpond a tt les indices de la ligne e de D (il y en a n2 donc).
                    # ensuite on va vider cette liste waiting_samples avec chaque pioche.
                    c = np.random.randint(0, len(waiting_samples[e]))
                    j = waiting_samples[e][c]
                    
                    # on store la correspondence entre X et D, en plus de X_states
                    # j c'est un element dans [n2]
                    X_nuisances[i, t] = j
                    
                    # on remplit X[i, t]
                    X[i, t, 0] = D[e, j]
                    selected_times[e, j] += 1     # pour documenter
                    # ce qui fait le "sans replacement"
                    waiting_samples[e].pop(c)
                    
                if len(waiting_samples[e]) < nt:
                     waiting_states.remove(e)
        
        # with replacement of blocks per coherent state
        if replace == 1:
            for i in range(nx):
                e = np.random.choice(waiting_states)
                X_states[i] = e
                
                # on pioche une sous-liste de n2, aleatoire, de taille nt, avec des indices non-uniques!
                c = np.random.choice(range(n2), nt, replace=False)
                
                for t in range(nt):
                    j = c[t]
                    X_nuisances[i, t] = j
                    X[i, t, 0] = D[e, j]
                    selected_times[e, j] += 1
          
        #######
        # ...?
        if replace == 2:
            for i in range(nx):
                e = np.random.choice(range(n1))
                X_states[i] = e
                for t in range(nt):
                    c = np.random.randint(0,n2)
                    X_nuisances[i,t] = c
                    X[i,t,0] = D[e,c]        
                    selected_times[e,c] += 1
        ####### 
            
        self.D = D
        self.X = X
        self.X_states = X_states
        self.X_nuisances = X_nuisances
        self.waiting_samples = waiting_samples
        self.selected_times = selected_times