import numpy as np
def fourier(theta):
    d=len(theta)
    return np.fft.fft(theta)/d