import numpy as np
def fourier(theta):
    '''
    Fourier transformation on Zd
    
    Parameters
    ----------
    theta : list
        The signal you want to transform
        
    Returns
    ----------
    Fourier transformation of theta
    '''
    return np.fft.fft(theta)/len(theta)