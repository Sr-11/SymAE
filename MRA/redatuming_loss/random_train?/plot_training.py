import MRA_generate as generate
from parameters import *
import matplotlib.pyplot as plt
def plot_training(model,test_X):
    '''
    Draw a plot of some (X, model.predict(X))
    
    Parameters
    ----------
    model : A tensorflow model (SymAE)
    test_X : np.ndarray
    
    Return
    ----------
    fig : A plt figure
    '''
    rows=3
    test_Y=model.predict(test_X)
    plt.rc('font', size=25)
    fig, axs = plt.subplots(rows, 2, sharex=True, sharey=True, figsize=(16,10))
    fig.suptitle('p=%d, q=%d'%(p,q))
    fig.subplots_adjust(wspace=0, hspace=0)
    fig.subplots_adjust(top=0.9,left=0.1,bottom=0.05,right=0.95)
    #fig.supylabel('value')
    axs[rows-1,0].set(xlabel='input')
    axs[rows-1,1].set(xlabel='output')
    axs[0,0].set_title('input')
    axs[0,1].set_title('output')
    for t in range(rows):
        axs[t,0].plot(range(d),test_X[0,t,:],label='%d'%t)
        axs[t,1].plot(range(d),test_Y[0,t,:],label='%d'%t)
        axs[t,0].grid(True)
        axs[t,1].grid(True)
        axs[t,0].set(ylabel='t=%d'%t)
    return fig