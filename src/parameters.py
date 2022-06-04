# Parameters related to data itself
N=1000 # N is the Cardinality of the data set X, say n_X in the paper
nt=20 # nt is the number of instances in each X_i, say n_tau in the paper (Xi[1]...Xi[nt])
d=100 # d is the dimensions of each "fundamental" data, say d=dim Xi[j] (e.g. d=28*28 for mnist)

# Parameters related to SymAE
p=8 # symmetric encoder latent dimensions, p=dim Ci (Coherent Code)
q=22 # nuisance encoder latent dimensions, q=dim Ni[j] (Dissimilar Code)

# Parameters related to NeuralNets
kernel_size=5 # How to convolute, the size of the kernel
filters=64 # How many filters in each convolution layer

# Parameters related to my specific algorithm of generating MRA data
ne=10 # Only use g0,g1...g9
sigma=0.0 # The intensity of the noise

# Parameters related to training
epochs=10 # Epochs

# Parameters related to matplotlib
fontsize=25 # Font size
figsize=(24,10) #(12,6)# Figure size
I=0 # Check X[I,:,:]
J=5 # Check X[I,1:J,:]

# Parameters related to existed weights
save=0 # 0:not save, 1:save
load=1 # 0:not load, 1:load