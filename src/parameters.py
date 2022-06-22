# This .py file defines most of the parameters used everywhere

# Parameters related to data itself
N=100 # N is the Cardinality of the data set X, say n_X in the paper
nt=20 # nt is the number of instances in each X_i, say n_tau in the paper (Xi[1]...Xi[nt])
d=100 # d is the dimensions of each "fundamental" data, say d=dim Xi[j] (e.g. d=28*28 for mnist)

# Parameters related to SymAE
p=10
q=20

# Parameters related to NeuralNets
kernel_size=5 # How to convolute, the size of the kernel
filters=64 # How many filters in each convolution layer
dropout_rate=0.8 # GaussianDropout

# Parameters related to my specific algorithm of generating MRA data
ne=4
sigma=0.0 # The intensity of the noise5 # GaussianDropouttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt