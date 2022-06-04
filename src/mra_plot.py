import MRA_generate as generate
from parameters import *
import matplotlib.pyplot as plt
import numpy as np
from saveplots import savefig
def mra_plot(model):
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
        
    Returns
    -------
    numpy.ndarray
        Return the generated data set X, a N*nt*d numpy tensor.
        X.shape=(N,nt,d)
    '''
    test_X=generate.generate_smooth(d,nt,1,ne,0)
    test_Y=model.predict(test_X)
    plt.rc('font', size=fontsize)
    fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, figsize=figsize)
    fig.suptitle('p=%d, q=%d'%(p,q))
    axs[0].set_title('input')
    axs[1].set_title('output')
    for j in range(J):
        axs[0].plot(range(d),test_X[0,j,:],label='%d'%j)
    for j in range(J):
        axs[1].plot(range(d),test_Y[0,j,:],label='%d'%j)
    for ax in axs.flat:
        ax.grid(True)
        ax.set(xlabel='x',ylabel='value')
        ax.legend()
    fig.savefig('result.png')
    savefig(fig)
    print("===== End =====")