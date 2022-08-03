from parameters import *
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
def movie_redatum(g, model):
    #plt.rc('font', size=25)
    figsize = (10*1.4,ne*1.5)
    fig, axs = plt.subplots(ne, 10, sharex=True, sharey=True, figsize=figsize)
    fig.suptitle('Redatuming')
    #fig.subplots_adjust(top=0.9,left=0.15,bottom=0.1,right=0.95)
    # N2 | [0,0] | [0,1]
    #----|-------+-------
    # N1 | [1,0] | [1,1] 
    #    |  C1   |  C2
    Xs = np.empty((1,nt,d))
    Xn = np.empty((1,nt,d))
    for i in range(ne):
        for j in range(10):
            l = d*j/10
            ax = axs[i,j]
            real = [g(i,((k+l)%d)/d) for k in range(d)]
            ax.plot(range(d), real, color='C0')
            
            ax.plot(range(d), model(Xs)[0,0,:], color='C1')
            
            Xs[0,0,:] = [g(i,((k+l)%d)/d) for k in range(d)]
            for t in range(1,nt):
                l = np.random.randint(d)
                Xs[0,t,:] = [g(i,((k+l)%d)/d) for k in range(d)]
                
            c = np.random.randint(ne)    
            Xn[0,0,:] = [g(c,((k+l)%d)/d) for k in range(d)]
            for t in range(1,nt):
                l = np.random.randint(d)
                Xn[0,t,:] = [g(c,((k+l)%d)/d) for k in range(d)]
    
            Zs = model.sym_encoder(Xs)
            Zn = model.nui_encoder(Xn)
            merger = model.latentcat(Zs, Zn, training=False)
            redatum = model.decoder(merger)
            ax.plot(range(d), redatum[0,0,:], color='C2')
            
     
    for j in range(10):
        axs[0,j].set(title='shift %d'%(d*j/10))
    for i in range(ne):
        axs[i,0].set(ylabel='state %d'%(i+1))
    for ax in axs.flat:
        ax.grid(True)
        ax.set_xlim([0,100])
        #ax.legend()
    blue_patch = mpatches.Patch(color='C0', label='real signal')
    orange_patch = mpatches.Patch(color='C1', label='reconstruct')
    green_patch = mpatches.Patch(color='C2', label='redatuming')
    fig.legend(handles=[blue_patch, orange_patch, green_patch])
    fig.tight_layout()
    fig.subplots_adjust(wspace=0, hspace=0)
    plt.xticks(np.arange(0,d,d/2))
    return fig