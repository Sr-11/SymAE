import MRA_generate as generate
import symae_core as symae
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from symae_model import SymAE
from parameters import *
from mra_plot import mra_plot

tfk = tf.keras
tfkl = tf.keras.layers
tfkltd= tf.keras.layers.TimeDistributed
gpus = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(gpus))
print("TensorFlow Version: ", tf.__version__)

# Generate MRA Data and Train SymAE
# Build SymAE model
autoencoder_input=tfk.Input(shape=(nt,d,1), dtype='float32', name='autoencoder_input')
model=tf.keras.Model(autoencoder_input,SymAE(N,nt,d,p,q,kernel_size,filters)(autoencoder_input),name='SymAE_model_in_MRA')
model.compile(optimizer='adam', loss='mse')
model.summary()

# Generate MRA data
X=generate.generate_smooth(d,nt,N,ne,sigma)

# Train the NeuralNet
if load==1:
    model.load_weights('./checkpoints/')
history=model.fit(X,X,epochs=epochs)
Y=model.predict(X)
if save==1:
    model.save_weights('./checkpoints/')

# Plot part of X,Y
mra_plot(model)