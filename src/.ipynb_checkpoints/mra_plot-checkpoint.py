import MRA_generate as generate
from parameters import *
import matplotlib.pyplot as plt
import numpy as np
from saveplots import savefig
def mra_plot(model):
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