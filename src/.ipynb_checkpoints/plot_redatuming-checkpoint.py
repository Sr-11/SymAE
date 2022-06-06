from parameters import *
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
def plot_redatuming(X1,X2,Y11,Y22,Y21,Y12,t):
    '''
    Plot a 2*2 figure, showing the redatuming result
    
    Parameters
    ----------
    X1 : np.array
        X1.shape=(1,nt,d)
        The first input of redatuming
    X2 : np.array
        X1.shape=(1,nt,d)
        The second input of redatuming
    Y11 : np.array
        Y11.shape=(1,nt,d)
        Y11=model.predict(X1)
    Y22 : np.array
        Y22.shape=(1,nt,d)
        Y22=model.predict(X2)
    Y21 : np.array
        Y21.shape=(1,nt,d)
        gengerated by coherent code from X1 and nuisance code from X2
    Y12 : np.array
        Y12.shape=(1,nt,d)
        gengerated by coherent code from X2 and nuisance code from X1
        
    Returns
    -------
    fig : A plt figure object
    '''
    top_left_corner_input=X1[0,t,:]
    top_left_corner_output=Y11[0,t,:]
    bottom_right_corner_input=X2[0,t,:]
    bottom_right_corner_output=Y22[0,t,:]
    top_right_corner_virtual=Y12[0,t,:]
    bottom_left_corner_virtual=Y21[0,t,:]
    # Plot redatuming
    plt.rc('font', size=25)
    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(16,16))
    fig.suptitle('Redatuming')
    fig.subplots_adjust(wspace=0, hspace=0)
    fig.subplots_adjust(top=0.92,left=0.1,bottom=0.05,right=0.95)
    axs[0,0].plot(range(d),top_left_corner_input)
    axs[0,0].plot(range(d),top_left_corner_output)
    #axs[0,0].set_title('C1+N1')
    axs[1,1].plot(range(d),bottom_right_corner_input)
    axs[1,1].plot(range(d),bottom_right_corner_output)
    #axs[1,1].set_title('C2+N2')
    axs[0,1].plot(range(d),top_right_corner_virtual,color='C2')
    #axs[0,1].set_title('C2+N1')
    axs[1,0].plot(range(d),bottom_left_corner_virtual,color='C2')
    axs[1,0].set(xlabel='Coherent 1')
    axs[1,1].set(xlabel='Coherent 2')
    axs[0,0].set(ylabel='Nuisance 1')
    axs[1,0].set(ylabel='Nuisance 2')
    for ax in axs.flat:
        ax.grid(True)
        ax.legend()
    blue_patch = mpatches.Patch(color='C0', label='input')
    orange_patch = mpatches.Patch(color='C1', label='output')
    green_patch = mpatches.Patch(color='C2', label='virtual')
    fig.legend(handles=[blue_patch,orange_patch,green_patch])
    return fig