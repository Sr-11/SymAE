import generate
import symae
import os
import numpy as np
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
tfk = tf.keras
tfkl = tf.keras.layers
tfkltd= tf.keras.layers.TimeDistributed
gpus = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(gpus))
print(tf.__version__)

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
sigma=0.1 # The intensity of the noise

# Build symmetric encoder
sym_enc_input=tfk.Input(shape=(nt,d,1), dtype='float32', name='sym_enc_input')
sym_enc_output=symae.SymmetricEncoder1D(kernel_size,filters,[2,2,2,2],[2,2,2,2],p)(sym_enc_input)
sym_encoder=tfk.Model(sym_enc_input, sym_enc_output, name='sym_encoder')
sym_encoder.summary()

# Build nuisance encoder
nui_enc_input=tfk.Input(shape=(nt,d,1), dtype='float32', name='sym_enc_input')
nui_enc_output=symae.NuisanceEncoder1D(kernel_size,filters,[2,2,2],[2,2,2,2],q)(nui_enc_input)
nui_enc_flatten=tfkl.Flatten()(nui_enc_output)
nui_encoder=tfk.Model(nui_enc_input, nui_enc_flatten, name='nui_encoder')
nui_encoder.summary()

# Build distribute in decoder
distribute_input = tfk.Input(shape=(p+q*nt), name='latentcode')
distribute_output=symae.DistributeZsym(nt, p, q)(distribute_input)
distzsym = tfk.Model(distribute_input, distribute_output, name='distzsym')
distzsym.summary()

#Build mixer in decoder
mixer_input = tfk.Input(shape=(nt,p+q), name='mixer_input')
mixer_output=symae.Mixer1D(kernel_size,filters,10,d)(mixer_input)
mixer = tfk.Model(mixer_input, mixer_output, name='mixer') 
mixer.summary()

# Build encoder
encoder_input=tfk.Input(shape=(nt,d,1), dtype='float32', name='encoder_input')
znuisance=nui_encoder(encoder_input)
zsym=sym_encoder(encoder_input)
latentcat=symae.LatentCat(0.65) #0.4 droupout
encoder_output=latentcat(zsym,znuisance)
encoder=tfk.Model(encoder_input, encoder_output, name="encoder")
encoder.summary()

# Build decoder
decoder_input = tfk.Input(shape=(p+q*nt), name='latentcode')
decoder_output=mixer(distzsym(decoder_input))
decoder=tfk.Model(decoder_input,decoder_output, name="decoder") 
decoder.summary()

#Build SymAE
model=tf.keras.Model(encoder_input, decoder(encoder_output) , name='autoencoder_clone')
model.summary()
model.compile(optimizer='adam', loss='mse')

X=generate.generate_smooth(d,nt,N,ne,sigma)
import matplotlib.pyplot as plt
for j in range(5):
    plt.plot(range(100),X[0,j,:],label='%d'%j)
plt.legend()
plt.show()

history=model.fit(X,X,epochs=50,shuffle=True)