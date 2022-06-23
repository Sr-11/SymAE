from parameters import *
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from redatuming import redatuming
def plot_redatuming(redatuming_object):
    '''
    Plot a 2*2 figure, showing the redatuming result
    
    Parameters
    ----------
    redatuming_object : class redatuming
    
    Returns
    -------
    fig : A plt figure object
    '''
    redatum=redatuming_object
    # Plot redatuming
    plt.rc('font', size=25)
    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(16,12))
    fig.suptitle('Redatuming \n p=%d, q=%d '%(p,q))
    fig.subplots_adjust(wspace=0, hspace=0)
    fig.subplots_adjust(top=0.9,left=0.15,bottom=0.1,right=0.95)
    # N2 | [0,0] | [0,1]
    #----|-------+-------
    # N1 | [1,0] | [1,1] 
    #    |  C1   |  C2
    axs[1,0].plot(range(d),redatum.C1_N1_input,color='C0')
    axs[1,0].plot(range(d),redatum.C1_N1_output,color='C1')
    
    axs[0,1].plot(range(d),redatum.C2_N2_input,color='C0')
    axs[0,1].plot(range(d),redatum.C2_N2_output,color='C1')
    
    axs[0,0].plot(range(d),redatum.C1_N2_virtual,color='C2')
    axs[0,0].plot(range(d),redatum.C1_N2_synthetic,color='C3')
    
    axs[1,1].plot(range(d),redatum.C2_N1_virtual,color='C2')
    axs[1,1].plot(range(d),redatum.C2_N1_synthetic,color='C3')


    axs[1,0].set(xlabel='Coherent 1 \n State=%d'%redatum.MRA1.states[0])
    axs[1,1].set(xlabel='Coherent 2 \n State=%d'%redatum.MRA2.states[0])
    axs[1,0].set(ylabel='Nuisance 1 \n Shift=%d'%(redatum.MRA1.shifts[0,redatum.t]))
    axs[0,0].set(ylabel='Nuisance 2 \n Shift=%d'%(redatum.MRA2.shifts[0,redatum.t]))
    for ax in axs.flat:
        ax.grid(True)
        ax.legend()
    blue_patch = mpatches.Patch(color='C0', label='input')
    orange_patch = mpatches.Patch(color='C1', label='output')
    green_patch = mpatches.Patch(color='C2', label='virtual')
    red_patch = mpatches.Patch(color='C3', label='synthetic')
    fig.legend(handles=[blue_patch,orange_patch,green_patch,red_patch])
    return fig