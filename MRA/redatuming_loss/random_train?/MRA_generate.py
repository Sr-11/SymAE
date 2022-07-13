import numpy as np
import math
import matplotlib.pyplot as plt
import random
from tqdm import trange
def g0(n,x):
    return np.random.rand(1)
class MRA_generate():
    '''
    Parameters
    ----------
    d : int
        The dimensions of each instance, i.e. d=dim Xi[j] (e.g. d=28*28 for mnist).
    nt : int
        The number of instances in each X_i, i.e. n_tau in the paper (Xi[1]...Xi[nt]).
    N : int
        Cardinality of the data set X, i.e. n_X in the paper.
    sigma : float
        This is not currently being used, skip it.
        The standard deviation of the noise of normal distribution.
    ne : int
        The number of different states, i.e. n_epsilon in the paper
    g : function
        g(n,x) n=0,1..., x∈(0,1)
    replace : int
        If replace==0, without replacement everywhere. Must $n_x*n_t <= n_1*n_2$ and $n_t <= n_2$.  
        If replace==1, X[i,:] have different nuisances, but each block in D can appear multiple times in X.  
        If replace==2, with replacement everywhere.

    Yields
    ----------
    X : numpy.ndarray
        Return the generated data set X, a N*nt*d numpy tensor.
        X.shape=(N,nt,d)
    states : numpy.ndarray
        states.shape=(N,)
        state[i] denotes the subscript of state of X_i.
    thetas : numpy.ndarray
        theta.shape=(N,d)
        thetas[i] denotes the state of X_i[t].
    shifts : numpy.ndarray
        shifts.shape=(N,nt)
        shifts[i,t] denotes the shift of X_i[t].

    '''
    def __init__(self,d=100,nt=20,N=1000,sigma=0,ne=20,g=g0,replace=1, continuous=False):
        self.d=d
        self.nt=nt
        self.N=N
        self.sigma=sigma
        self.ne=ne
        self.g=g
        self.replace=replace
        self.X = np.empty((N,nt,d), dtype = np.float32, order = 'C')
        self.states = np.empty((N), dtype = int, order = 'C')
        self.thetas = np.empty((N,d), dtype = np.float32, order = 'C')
        self.shifts = np.empty((N,nt), dtype = int, order = 'C')
        self.waiting_samples = [list(range(d)) for i in range(ne)]
        self.waiting_states = list(range(ne))
        self.select_times = np.zeros((ne,d))
        self.continuous = continuous
    def generate_default(self):
        d = self.d
        nt = self.nt
        N = self.N
        sigma = self.sigma
        ne = self.ne
        g = self.g
        replace = self.replace
        X = self.X
        states = self.states
        thetas = self.thetas
        shifts = self.shifts
        waiting_samples = [list(range(d)) for i in range(ne)]
        self.waiting_samples = waiting_samples
        waiting_states = list(range(ne))
        self.waiting_states = waiting_states
        select_times = np.zeros((ne,d))
        self.select_times = select_times
        if self.continuous == True:
            self.shifts = np.empty((N,nt), dtype = np.float32, order = 'C')
            for i in trange(N):
                e=np.random.choice(waiting_states)
                states[i]=e
                ls=np.random.uniform(0, d, nt)
                for j in range(nt):
                    l=ls[j]
                    self.shifts[i,j]=l
                    select_times[e,int(l)] += 1
                    for k in range(d):
                        X[i,j,k]=g(e,(k+l)%d/d)+sigma*np.random.normal()
            
        elif self.continuous == False:
            if replace == 0:
                for i in range(N):
                    if len(waiting_states) == 0:
                        waiting_states = list(range(ne))
                        waiting_samples = [list(range(d)) for i in range(ne)]
                    e=np.random.choice(waiting_states)
                    states[i]=e
                    thetas[i,:]=[g(e,k/d) for k in range(d)]
                    ls=np.random.choice(waiting_samples[e],replace=False,size=nt)
                    for l in ls:
                        waiting_samples[e].remove(l)
                    if len(waiting_samples[e])<nt:
                        waiting_states.remove(e)
                    for j in range(nt):
                        l=ls[j]
                        shifts[i,j]=l
                        select_times[e,l] += 1
                        for k in range(d):
                            X[i,j,k]=thetas[i,(k+l)%d]+sigma*np.random.normal()
            if replace == 1:
                for i in range(N):
                    e=np.random.choice(waiting_states)
                    states[i]=e
                    thetas[i,:]=[g(e,k/d) for k in range(d)]
                    ls=np.random.choice(waiting_samples[e],replace=False,size=nt)
                    for j in range(nt):
                        l=ls[j]
                        shifts[i,j]=l
                        select_times[e,l] += 1
                        for k in range(d):
                            X[i,j,k]=thetas[i,(k+l)%d]+sigma*np.random.normal()
            if replace == 2:
                for i in range(N):
                    e=np.random.choice(waiting_states)
                    states[i]=e
                    thetas[i,:]=[g(e,k/d) for k in range(d)]
                    ls=np.random.choice(waiting_samples[e],replace=True,size=nt)
                    for j in range(nt):
                        l=ls[j]
                        shifts[i,j]=l
                        select_times[e,l] += 1
                        for k in range(d):
                            X[i,j,k]=thetas[i,(k+l)%d]+sigma*np.random.normal()
        return X
        
    """
    def generate_random(self):
        '''
        Generate a Multireference Alignment (MRA) data set X. 
        X[i,j,:] is the j-th instance of the i-th data. 
        The value X[i,j,k] randomly samples from a uniform distribution over [0, 1).
        X[i,j,k]~U(0,1)+noise, iid, where k=1,2...d (given 1<=i<=N and 1<=j<=nt).
        '''
        d=self.d;nt=self.nt;N=self.N;sigma=self.sigma;ne=self.ne;
        X=self.X;SNR=self.SNR;thetas=self.thetas;shifts=self.shifts
        for i in range(N):
            thetas[i,:]=np.random.rand(d)
            if sigma!=0:
                SNR[i]=(np.linalg.norm(thetas[i,:],2)/sigma)**2
            else:
                SNR[i]=np.inf
            for j in range(nt):
                l=np.random.randint(d)
                shifts[i,j]=l
                for k in range(d):
                    X[i,j,k]=thetas[i,(k+l)%d]+sigma*np.random.normal()
        return X
    def generate_trigonometric(self):
        '''
        Generate a Multireference Alignment (MRA) data set X using sin(nx) and cos(nx)
        '''
        d=self.d;nt=self.nt;N=self.N;sigma=self.sigma;ne=self.ne;states=self.states
        X=self.X;SNR=self.SNR;thetas=self.thetas;shifts=self.shifts
        def f(n,x):
            return math.cos(n*x/d*2*math.pi)
        for i in range(N):
            e=np.random.randint(ne)
            states[i]=e
            thetas[i,:]=[f(e,k) for k in range(d)]
            if sigma!=0:
                SNR[i]=(np.linalg.norm(thetas[i,:],2)/sigma)**2
            else:
                SNR[i]=np.inf
            for j in range(nt):
                l=np.random.randint(d)
                shifts[i,j]=l
                for k in range(d):
                    X[i,j,k]=thetas[i,(k+l)%d]+sigma*np.random.normal()
        return X
    def generate_smooth(self):
        '''
        Generate a Multireference Alignment (MRA) data set X. 
        X[i,j,:] is the j-th instance of the i-th data. 
        X[i,j,k] is "smooth" wrt k (though X[i,j,:] is discrete).
        X[i,j,:] is generated by conbination of Sin and Cos (g(m,x) in the code).
        '''
        d=self.d;nt=self.nt;N=self.N;sigma=self.sigma;ne=self.ne;states=self.states
        X=self.X;SNR=self.SNR;thetas=self.thetas;shifts=self.shifts
        def f(n,x):
            if n%2==0:
                return math.cos(n/2*x/d*2*math.pi)
            else:
                return math.sin((n+1)/2*x/d*2*math.pi)
        def g(n,x):
            return sum([f(k,x)/(k+1) for k in range(n+1)])
        for i in range(N):
            e=np.random.randint(ne)
            states[i]=e
            thetas[i,:]=[g(e,k) for k in range(d)]
            if sigma!=0:
                SNR[i]=(np.linalg.norm(thetas[i,:],2)/sigma)**2
            else:
                SNR[i]=np.inf
            for j in range(nt):
                l=np.random.randint(d)
                shifts[i,j]=l      
                for k in range(d):
                    X[i,j,k]=thetas[i,(k+l)%d]+sigma*np.random.normal()
        return X

    def generate_smooth_no_replacement(self):
        '''
        Generate a Multireference Alignment (MRA) data set X. 
        X[i,j,:] is the j-th instance of the i-th data. 
        X[i,j,k] is "smooth" wrt k (though X[i,j,:] is discrete).
        The difference between generate_smooth_no_replacement and generate_smooth is that
        this function guarantees the cyclic shifts related to different j's are all different, 
        while generate_smooth does not. 
        '''
        d=self.d;nt=self.nt;N=self.N;sigma=self.sigma;ne=self.ne;states=self.states
        X=self.X;SNR=self.SNR;thetas=self.thetas;shifts=self.shifts
        def f(n,x):
            if n%2==0:
                return math.cos(n/2*x/d*2*math.pi)
            else:
                return math.sin((n+1)/2*x/d*2*math.pi)
        def g(n,x):
            return sum([f(k,x)/(k+1) for k in range(n+1)])
        for i in range(N):
            e=np.random.randint(ne)
            states[i]=e
            thetas[i,:]=[g(e,k) for k in range(d)]
            if sigma!=0:
                SNR[i]=(np.linalg.norm(thetas[i,:],2)/sigma)**2
            else:
                SNR[i]=np.inf
            l=random.sample(range(d),nt)
            for j in range(nt):
                shifts[i,j]=l
                for k in range(d):
                    X[i,j,k]=thetas[i,(k+l[j])%d]+sigma*np.random.normal()
        return X
    """