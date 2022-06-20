#%% Load packages
import tensorflow as tf
tfk = tf.keras
tfkl = tf.keras.layers
tfkltd= tf.keras.layers.TimeDistributed

import os 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="6,7"

gpus = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(gpus))
print("TensorFlow Version: ", tf.__version__)

class NuisanceEncoderDense0D(tf.keras.Model):
  def __init__(self, nt_out, nt_in):
    super(NuisanceEncoderDense0D, self).__init__(name='')
    n=max(nt_in,nt_out)
    self.d1=tfkl.Dense(n,activation='elu')
    self.d2=tfkl.Dense(n*2,activation='elu')
    self.d3=tfkl.Dense(n*6,activation='elu')
    self.d4=tfkl.Dense(n*3,activation='elu')
    self.d5=tfkl.Dense(n,)
    self.a=tfkl.Activation('elu')
    self.d6=tfkl.Dense(nt_out)
    self.bn=tf.keras.layers.BatchNormalization()
  def call(self, input_tensor, training=False):
    x=self.d1(input_tensor)
    x=self.d2(x)
    x=self.d3(x)
    x=self.d4(x)
    x=self.d5(x)
    x=self.bn(x, training=training)
    x=self.a(x)
    out=self.d6(x)
    return out
  def model(self, x):
    return tfk.Model(inputs=x, outputs=self.call(x))

class SymmetricEncoderDense0D(tf.keras.Model):
  def __init__(self, nt_out, nt_in):
    super(SymmetricEncoderDense0D, self).__init__(name='')
    n=max(nt_in,nt_out)
    self.nt_out=nt_out
    self.nt_in=nt_in
    self.d1=tfkl.Dense(n,activation='elu')
    self.d2=tfkl.Dense(n*2,activation='elu')
    self.d3=tfkl.Dense(n*6,activation='elu')
    self.d4=tfkl.Dense(n*3,activation='elu')
    self.d7=tfkl.Dense(n,)
    self.d8=tfkl.Dense(nt_out,)
    self.bn=tfkl.BatchNormalization()
    self.a=tfkl.Activation('elu')
  def call(self, input_tensor, training=False):
    x=self.d1(input_tensor)
    x=self.d2(x)
    x=self.d3(x)
    x=self.d4(x)
    x=tf.math.reduce_mean(x,axis=1)
    x=self.d7(x)
    x=self.bn(x, training=training)
    x=self.a(x)
    out=self.d8(x)
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
    self.drop=tfkl.GaussianDropout(alpha)
  def call(self, zsym, znuisance,training=False):
    znuisance=self.drop(znuisance,training=training)
    z=tfkl.concatenate([zsym, znuisance])
    return z

class MixerDense0D(tf.keras.Model):
  def __init__(self, nt_out, nt_in):
    super(MixerDense0D, self).__init__(name='')
    self.nt_out=nt_out
    self.nt_in=nt_in
    n=max(nt_out,nt_in)
    self.d1=tfkl.Dense(n,activation='elu')
    self.d2=tfkl.Dense(n*2,activation='elu')
    self.d3=tfkl.Dense(n*6,activation='elu')
    self.d4=tfkl.Dense(n*3,activation='elu')
    self.d7=tfkl.Dense(n,)
    self.d8=tfkl.Dense(nt_out,)
    self.bn=tfkl.BatchNormalization()
    self.a=tfkl.Activation('elu')
    self.bn=tfkl.BatchNormalization()
    self.a=tfkl.Activation('elu')
  def call(self, z, training=False):
    x=self.d1(z)
    x=self.d2(x)
    x=self.d3(x)
    x=self.d4(x)
    x=self.d7(x)
    x=self.bn(x, training=training)
    x=self.a(x)
    out=self.d8(x)
    return out
  def model(self, x):
    return tfk.Model(inputs=x, outputs=self.call(x))