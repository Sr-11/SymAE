import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
def plot_training(X,Y,i=0):
    plt.figure(figsize=(5,3),dpi=200)
    plt.plot(range(len(X[i,:,0])),X[i,:,0])
    plt.plot(range(len(Y[i,:,0])),Y[i,:,0])
    blue_patch = mpatches.Patch(color='C0', label='$X_i[i]$')
    orange_patch = mpatches.Patch(color='C1', label='$\hat{X}_i[j]$')
    plt.legend(handles=[blue_patch,orange_patch])
    plt.xlabel('j')
    plt.ylabel('value')
    fig=plt.gcf()
    plt.tight_layout()
    return fig