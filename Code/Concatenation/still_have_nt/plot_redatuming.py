import matplotlib.pyplot as plt
import numpy as np
def plot_redatuming(Z,Z_hat,X_states,i,i_prime):
    Input = Z[i,i_prime,:]
    nx,_,nt = Z.shape
    #print( X_states[i_prime] )
    #print( X_nuisances[i,j] )
    #print( CEnc(Input,j) )
    #print( NEnc(Input,j) )
    #print( Z[i,i_prime,j] )
    #print( np.floor(Z_hat[i,i_prime,j]) )
    fig=plt.figure(figsize=(5, 3), dpi=500)
    plt.plot(range(nt),Z[i,i_prime,:],label='${X}_{s=%d \mapsto s=%d}[j]$'%(X_states[i],X_states[i_prime]))
    plt.plot(range(nt),Z_hat[i,i_prime,:],label='$\hat{X}_{s=%d \mapsto s=%d}[j]$'%(X_states[i],X_states[i_prime]))
    plt.legend()
    plt.title('$\hat{X}_{s=%d \mapsto s=%d}[j]$'%(X_states[i],X_states[i_prime]))
    plt.ylabel('value')
    plt.xlabel('j')
    plt.tight_layout()
    return plt.gcf()