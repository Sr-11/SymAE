import numpy as np
from redatuming import redatuming
def plot_complexity(model,MRA1,MRA2,t):
    '''
    Currently not used
    '''
    Y11=model.predict(MRA1.X)
    Y22=model.predict(MRA2.X)
    Y21,Y12=redatuming(model,X1,X2,t)

    X=MRA_data.generate_smooth()
    X_clean=MRA_data.X_clean
    theta=X_clean[0,0,:]
    data_SNR=MRA_data.SNR
    Y=model.predict(X)
    reconstruction_SNR=np.linalg.norm(theta,2)/sigma)**2
    plt.rc('font', size=25)
    fig, axs = plt.subplots(rows, 2, sharex=True, sharey=True, figsize=(16,20))
    fig.suptitle('p=%d, q=%d'%(p,q))
    fig.subplots_adjust(wspace=0, hspace=0)
    fig.subplots_adjust(top=0.95,left=0.1,bottom=0.05,right=0.95)
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