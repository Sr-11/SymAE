import numpy as np
import math
import matplotlib.pyplot as plt
def generate(d=100,nt=20,N=10000,sigma=0):
    X=np.empty((N,nt,d), dtype = float, order = 'C')
    for i in range(N):
        theta=np.random.rand(d)
        theta=theta/np.linalg.norm(theta,2)*np.sqrt(d)
        for j in range(nt):
            l=np.random.randint(d)
            for k in range(d):
                X[i,j,k]=theta[(k+l)%d]+sigma*np.random.normal()
    return X
def generate_smooth(d=100,nt=20,N=10000,ne=10,sigma=0):
    PI=math.pi
    sin=math.sin
    cos=math.cos
    # We hope the data to be smooth
    # But it's not essential, just for human beings to feel what's happening
    # We expect a set of functions $f_i(x)$ defined on S1
    # A basis can be sin(nx) & cos(nx)
    # Let f(2n,x)=cos(nx), f(2n-1,x)=sin(nx)
    # Let f(0,x)=1
    # But we don't want any extra cycle(period?) smaller than 2pi, at least now
    # d is the length of the signal, \theta\in\mathbb{R}^d
    def f(n,x):
        if n%2==0:
            return cos(n/2*x/d*2*PI)
        else:
            return sin((n+1)/2*x/d*2*PI)
    # While we don't like perodic
    def g(n,x):
        return sum([f(k,x)/(k+1) for k in range(n+1)])
    '''
    for n in range(6):
        plt.plot(range(d),[g(n,x) for x in range(d)],label='g%d'%n)
    plt.xlabel('$x$ label');plt.ylabel('$y$ label');plt.title("Simple Plot")
    plt.legend();plt.show()
    '''
    # Now we generate g_n, n=0,1,2...
    # Then, generate D first
    # ? D_j^\epsilon is a "measurement", where e in 1:n_e, j in 1:n_msmt
    # D is a n_msmt * n_e block "matrix", where each block is d*1
    # Ok, D is a 3-dimensional tensor?
    # No, won't construct D. D is like a Sample Space.
    # What's the shape of X in Pawan???
    # X.shape=(nX,nt,k)=(N,nt,d)
    X=np.empty((N,nt,d), dtype = float, order = 'C')
    ne=10 # Only use g0,g1...g9
    for i in range(N):
        e=np.random.randint(ne)
        theta=[g(e,k) for k in range(d)]
        theta=theta/np.linalg.norm(theta,2)*np.sqrt(d)
        for j in range(nt):
            l=np.random.randint(d) # move to the left
            for k in range(d):
                X[i,j,k]=theta[(k+l)%d]+sigma*np.random.normal()
    return X

##### Test #####
def test():
    X=generate_smooth(100,20,1000,10,0.01)
    for j in range(5):
        plt.plot(range(100),X[0,j,:],label='%d'%j)
    plt.legend()
    plt.show()
test()
##### Correct #####
