# This .py file defines most of the parameters used everywhere

# Parameters related to data itself
N = 1000 # N is the Cardinality of the data set X, i.e. n_X in the paper
nt = 20 # nt is the number of instances in each X_i, i.e. n_tau in the paper (Xi[1]...Xi[nt])
d = 100 # d is the dimensions of image.


# Parameters related to SymAE
p = 100 # Dimensions of the coherent latent space
q = 3 # Dimensions of the nuisant latent space

# Parameters related to NeuralNets
kernel_size = 5 # How to convolute, the size of the kernel
filters = 64 # How many filters in each convolution layer
dropout_rate = 0.0 # GaussianDropout

# Parameters related to my specific algorithm of generating MRA data
ne = 50 # The number of states, i.e. n_epsilon in the paper
sigma = 0.0 # The intensity of the noise, currently it is not used, always set sigma=0