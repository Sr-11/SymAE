import numpy as np
import math
import matplotlib.pyplot as plt
import random
import timeit
class rank1_generate():
    def __init__(self,n1=1000,n2=1000,nx=100,nt=10):
        '''
        D in n1*n2
        X in nx*t
        '''
        self.n1=n1
        self.n2=n2
        self.nx=100
        self.nt=10
    def generate_default(self):
        a=np.random.randint(10000, size=(n1,1))
        b=np.random.randint(10000, size=(n2,1))
        D=a*b.T
        self.D=D
        
        return self.X