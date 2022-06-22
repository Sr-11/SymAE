import numpy as np
import math
import matplotlib.pyplot as plt
import random
import timeit
def g0(n,x):
    return np.random.rand(1)
class MRA_generate():
    '''
    Parameters
    ----------
    d : int
        The dimensions of each instance, say d=dim Xi[j] (e.g. d=28*28 for mnist).
    nt : int
        The number of instances in each X_i, say n_tau in the paper (Xi[1]...Xi[nt]).
    N : int
        Cardinality of the data set X, say n_X in the paper.
    sigma : float
        The standard deviation of the noise of normal distribution.
    ne : int
        For convinence, remaining consistent with other generate_ functions.
    g : function
        g(n,x) n=0,1..., x∈(0,1)
    replace : bool
        if replace==True, Xi[j] (i fixed, j different) may correspond to same block in D.
    outer_replace : bool
        if outer_replace==False, Xi[j] are all different (for all i,j).

    Yields
    ----------
    X : numpy.ndarray
        Return the generated data set X, a N*nt*d numpy tensor.
        X.shape=(N,nt,d)
    SNR : numpy.ndarray
        SNR.shape=(N,)
        Signal-to-noise ratio
    thetas : numpy.ndarray
        theta.shape=(N,d)
    shifts : numpy.ndarray
        shifts.shape=(N,nt)
    states : numpy.ndarray
        states.shape=(N,)
        states is used to mark how many different states are in X
        doesn't have uniform format
    '''
    def __init__(self,d=100,nt=20,N=10000,sigma=0,ne=20,g=g0,replace=False,outer_replace=True):
        self.d=d
        self.nt=nt
        self.N=N
        self.sigma=sigma
        self.ne=ne
        self.g=g
        self.replace=replace
        self.outer_replace=outer_replace
    def generate_default(self):
        d = self.d
        nt = self.nt
        N = self.N
        sigma = self.sigma
        ne = self.ne
        g = self.g
        replace = self.replace
        outer_replace = self.outer_replace
        X=np.empty((N,nt,d), dtype = float, order = 'C')
        self.X=X
        states=np.empty(N, dtype = int)
        self.states=states
        thetas=np.empty((N,d), dtype = float, order = 'C')
        self.thetas=thetas
        SNR=np.empty(N)
        self.SNR=SNR
        shifts=np.empty((N,nt), dtype = float, order = 'C')
        self.shifts=shifts
        waiting_samples = [list(range(d)) for i in range(ne)]
        self.waiting_samples=waiting_samples
        waiting_states = list(range(ne))
        self.waiting_states=waiting_states
        select_times = np.zeros((ne,d))
        self.select_times = select_times
        for i in range(N):
            e=np.random.choice(waiting_states)
            states[i]=e
            thetas[i,:]=[g(e,k/d) for k in range(d)]
            if sigma!=0:
                SNR[i]=(np.linalg.norm(thetas[i,:],2)/sigma)**2
            else:
                SNR[i]=np.inf
            ls=np.random.choice(waiting_samples[e],replace=replace,size=nt)
            if outer_replace==False:
                for l in ls:
                    waiting_samples[e].remove(l)
                if len(waiting_samples[e])<nt:
                    waiting_states.remove(e)
            for j in range(nt):
                #l=np.random.randint(d)
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