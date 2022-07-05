#%% Load packages
import tensorflow as tf
tfk = tf.keras
tfkl = tf.keras.layers
tfkltd= tf.keras.layers.TimeDistributed
from parameters import *
import os 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="6,7"

gpus = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(gpus))
print("TensorFlow Version: ", tf.__version__)

class NuisanceEncoder1D(tf.keras.Model):
  def __init__(self, kernel_size, filter,  fstep=[2,4,8], tdown=[1,1,2,2], latent_dim=512):
    super(NuisanceEncoder1D, self).__init__(name='')
    k1=kernel_size

    self.c1=tfkltd(tfkl.Conv1D(filter,((k1)),padding='same',activation='elu'))
    self.c2=tfkltd(tfkl.Conv1D(filter,((k1)),padding='same',activation='elu'))
    self.mp1=tfkltd(tfkl.MaxPool1D(pool_size=(tdown[0])))
    self.c3=tfkltd(tfkl.Conv1D(filter//fstep[0],((k1)),padding='same',activation='elu'))
    self.c4=tfkltd(tfkl.Conv1D(filter//fstep[0],((k1)),padding='same',activation='elu'))
    self.mp2=tfkltd(tfkl.MaxPool1D(pool_size=(tdown[1])))
    self.c5=tfkltd(tfkl.Conv1D(filter//fstep[1],((k1)),padding='same',activation='elu'))
    self.c6=tfkltd(tfkl.Conv1D(filter//fstep[1],((k1)),padding='same',activation='elu'))
    self.mp3=tfkltd(tfkl.MaxPool1D(pool_size=(tdown[2])))
    self.c7=tfkltd(tfkl.Conv1D(filter//fstep[2],((k1)),padding='same',activation='elu'))
    self.c8=tfkltd(tfkl.Conv1D(filter//fstep[2],((k1)),padding='same'))
    self.bn=tfkltd(tfkl.BatchNormalization())
    self.a=tfkltd(tfkl.Activation('elu'))
    self.mp4=tfkltd(tfkl.MaxPool1D(pool_size=(tdown[3])))
    self.f=tfkltd(tfkl.Flatten())
    self.d=tfkltd(tfkl.Dense(latent_dim))
    
    #self.f2=tfkl.Flatten()
    self.bn2=tfkltd(tfkl.BatchNormalization(activity_regularizer=tf.keras.regularizers.L2(0.1)))
    #self.bn3=tfkl.BatchNormalization()
    
    #self.ln1=tfkl.LayerNormalization(axis=2)
    
        
  def call(self, input_tensor, training=False):
    # n, ntau, nr, nt, nc = input_tensor.get_shape()
    x=self.c1(input_tensor)
    x=self.c2(x)
    x=self.mp1(x)
    x=self.c3(x)
    x=self.c4(x)
    x=self.mp2(x)
    x=self.c5(x)
    x=self.c6(x)
    x=self.mp3(x)
    x=self.c7(x)
    x=self.c8(x)
    x=self.bn(x, training=training)
    x=self.a(x)
    x=self.mp4(x)
    x=self.f(x)
    out=self.d(x)
    
    #out=self.f2(out)
    #out=self.bn2(out, training=training)
    #out=self.bn3(out, training=training)  
    #out=tf.reshape(out, [-1, nt, q])
    
    return out
  def model(self, x):
    return tfk.Model(inputs=x, outputs=self.call(x))

class SymmetricEncoder1D(tf.keras.Model):
  def __init__(self, kernel_size, filters, fstep=[2,4,8], tdown=[2,4,4,4],latent_dim=8):
    super(SymmetricEncoder1D, self).__init__(name='')
    k1=kernel_size
    self.c11=tfkltd(tfkl.Conv1D(filters,(k1),padding='same',activation='elu'))
    self.c12=tfkltd(tfkl.Conv1D(filters//fstep[0],(k1),padding='same',activation='elu'))
    self.mp11=tfkltd(tfkl.MaxPool1D(pool_size=(tdown[0])))
    self.c13=tfkltd(tfkl.Conv1D(filters//fstep[1],(k1),padding='same',activation='elu'))
    self.c14=tfkltd(tfkl.Conv1D(filters//fstep[2],(k1),padding='same',activation='elu'))
    self.mp12=tfkltd(tfkl.MaxPool1D(pool_size=(tdown[1])))

    self.c21=tfkl.Conv1D(filters,(k1),padding='same',activation='elu')
    self.c22=tfkl.Conv1D(filters//fstep[0],(k1),padding='same',activation='elu')
    self.mp21=tfkl.MaxPool1D(pool_size=(tdown[2]))
    self.c23=tfkl.Conv1D(filters//fstep[1],(k1),padding='same',activation='elu')
    self.c24=tfkl.Conv1D(filters//fstep[2],(k1),padding='same')
    self.bn=tfkl.BatchNormalization()
    self.a=tfkl.Activation('elu')
    self.mp22=tfkl.MaxPool1D(pool_size=(tdown[3]))
    self.f=tfkl.Flatten()
    self.d=tfkl.Dense(latent_dim)

  def call(self, input_tensor, training=False):
    # n, ntau, nr, nt, nc = input_tensor.get_shape()
    x=self.c11(input_tensor)
    x=self.c12(x)
    x=self.mp11(x)
    x=self.c13(x)
    x=self.c14(x)
    x=self.mp12(x)
    x=tf.math.reduce_mean(x,axis=1)
    x=self.c21(x)
    x=self.c22(x)
    x=self.mp21(x)
    x=self.c23(x)
    x=self.c24(x)
    x=self.bn(x, training=training)
    x=self.a(x)
    x=self.mp22(x)
    x=self.f(x)
    out=self.d(x)
    return out

  def model(self, x):
    return tfk.Model(inputs=x, outputs=self.call(x))

class DistributeZsym(tf.keras.Model):
  def __init__(self, ntau, nz0, nzi):
    super(DistributeZsym, self).__init__(name='')

    self.nz0=nz0
    self.nzi=nzi
    self.ntau=ntau
    self.ri=tfkl.Reshape(target_shape=(ntau,nzi))
    self.repeat=tfkl.RepeatVector(ntau)

  def call(self, z, training=False):

    z0,zi=tf.split(z,[self.nz0, self.ntau*self.nzi],axis=1)
    zi=self.ri(zi)
    z0=self.repeat(z0)
    out=tfkl.concatenate([z0, zi],axis=2)
    return out
  def model(self, x):
    return tfk.Model(inputs=x, outputs=self.call(x))


class LatentCat(tf.keras.Model):
  def __init__(self, alpha=1.0):
    super(LatentCat, self).__init__(name='')
    #self.drop = tf.keras.layers.GaussianNoise(alpha)
    #self.drop = tfkl.GaussianDropout(alpha)
    self.drop=tfkl.Dropout(alpha)

  def call(self, zsym, znuisance,training=False):
    znuisance=self.drop(znuisance,training=training)
    z=tfkl.concatenate([zsym, znuisance])
    return z

class Mixer1D(tf.keras.Model):
  def __init__(self, kernel_size, filters, upfact, nt):
    super(Mixer1D, self).__init__(name='')
    (k1)=kernel_size
    tup=upfact

    self.d1=tfkltd(tfkl.Dense(units=((nt//tup)*1),activation='elu'))
    self.r2=tfkltd(tfkl.Reshape(target_shape=((nt//tup),1)))
    self.c1=tfkltd(tfkl.Conv1D(filters,((k1)),padding='same',activation='elu'))
    self.us1=tfkltd(tfkl.UpSampling1D(size=(tup)))
    self.c2=tfkltd(tfkl.Conv1D(filters,((k1)),padding='same',activation='elu'))
    self.c3=tfkltd(tfkl.Conv1D(filters,((k1)),padding='same'))
    self.bn=tfkltd(tfkl.BatchNormalization())
    self.a=tfkltd(tfkl.Activation('elu'))
    self.c4=tfkltd(tfkl.Conv1D(filters,((k1)),padding='same',activation='elu'))
    self.c5=tfkltd(tfkl.Conv1D(filters,((k1)),padding='same',activation='elu'))
    self.c6=tfkltd(tfkl.Conv1D(1,((k1)),padding='same'))

  def call(self, z, training=False):
    x=self.d1(z)
    x=self.r2(x)
    x=self.c1(x)
    x=self.us1(x)
    x=self.c2(x)
    x=self.c3(x)
    x=self.bn(x, training=training)
    x=self.a(x)
    x=self.c4(x)
    x=self.c5(x)
    out=self.c6(x)
    return out
  def model(self, x):
    return tfk.Model(inputs=x, outputs=self.call(x))

