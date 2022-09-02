from parameters import *
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from tqdm import tqdm
def movie_redatum(g, model):
    plt.rc('font', size=15)
    M = 10
    figsize = (M*1.4,ne*1.5)
    fig, axs = plt.subplots(ne, M, sharex=True, sharey=True, figsize=figsize)
    fig.suptitle('Redatuming')
    
    Xs = np.empty((1,nt,d,1))
    Xn = np.empty((1,nt,d,1))
    for i in tqdm(range(ne)):
        for j in range(M):
            ax = axs[i,j]
            l = d*j/M
            
            Xs[0,0,:,0] = [g(i,((k+l)%d)/d) for k in range(d)]
            for t in range(1,nt):
                l = np.random.randint(d)
                Xs[0,t,:,0] = [g(i,((k+l)%d)/d) for k in range(d)]
           # Xs+=np.random.normal(size=Xs.shape,scale=0.1)
            
            ##### real #####
            l = d*j/M
            real = Xs[0,0,:,0] 
            #ax.scatter(range(d), real, color='C2', s=2.3)
            ax.plot(range(d), real, color='C2', linewidth=2.3)
            ##### reconstruct #####
            l = d*j/M

            #ax.scatter(range(d), model(Xs)[0,0,:,0], color='C1', s=3.0)
            ax.plot(range(d), model(Xs)[0,0,:,0], color='C1', linewidth=3.0)
            ##### redatum #####
            c=0
            if i==0:
                c=1
            elif i==1:
                c=2
            elif i==2:
                c=0
            elif i==3:
                c=5
            elif i==4:
                c=3
            elif i==5:
                c=4
            l = d*j/M
            Xn[0,0,:,0] = [g(c,((k+l)%d)/d) for k in range(d)]
            for t in range(1,nt):
                l = np.random.randint(d)
                Xn[0,t,:,0] = [g(c,((k+l)%d)/d) for k in range(d)]
         #   Xn+=np.random.normal(size=Xs.shape,scale=0.1)
            
            Zs = model.sym_encoder(Xs)
            Zn = model.nui_encoder(Xn)
            merger = model.latentcat(Zs, Zn, training=False)
            redatum = model.decoder(merger)
            #ax.scatter(range(d), redatum[0,0,:,0], color='C0', s=2.0)
            ax.plot(range(d), redatum[0,0,:,0], color='C0', linewidth=2.2)
            
    for j in range(M):
        axs[0,j].set(title='shift %d'%(d*j/M))
    for i in range(ne):
        axs[i,0].set(ylabel='state %d'%(i+1))
    for ax in axs.flat:
        ax.grid(True)
        ax.set_xlim([0,100])
        #ax.legend()
    blue_patch = mpatches.Patch(color='C2', label='real signal')
    orange_patch = mpatches.Patch(color='C1', label='reconstruct')
    green_patch = mpatches.Patch(color='C0', label='redatuming')
    fig.legend(handles=[blue_patch, orange_patch, green_patch])
    fig.tight_layout()
    fig.subplots_adjust(wspace=0, hspace=0)
    plt.xticks(np.arange(0,d,d/2))
    return fig