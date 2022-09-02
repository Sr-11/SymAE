import tensorflow as tf
tfk = tf.keras
tfkl = tf.keras.layers
tfkltd= tf.keras.layers.TimeDistributed

class NuisanceEncoder1D(tf.keras.Model):
  def __init__(self, latent_dim):
    super(NuisanceEncoder1D, self).__init__(name = 'nui_encoder')
    self.c3=tfkltd(tfkl.Conv1D(32, 5, padding='same', activation='elu'))
    self.c4=tfkltd(tfkl.Conv1D(32, 5, padding='same', activation='elu'))
    self.mp2=tfkltd(tfkl.MaxPool1D(pool_size = 2))
    self.c5=tfkltd(tfkl.Conv1D(32, 5, padding='same', activation='elu'))
    self.c6=tfkltd(tfkl.Conv1D(32, 5, padding='same', activation='elu'))
    self.mp3=tfkltd(tfkl.MaxPool1D(pool_size = 2))
    self.c7=tfkltd(tfkl.Conv1D(32, 5, padding='same', activation='elu'))
    self.c8=tfkltd(tfkl.Conv1D(32, 5, padding='same', activation='elu'))
    self.bn=tfkltd(tfkl.BatchNormalization())
    self.a=tfkltd(tfkl.Activation('elu'))
    self.mp4=tfkltd(tfkl.MaxPool1D(pool_size = 2))
    self.f=tfkltd(tfkl.Flatten())
    self.d=tfkltd(tfkl.Dense(latent_dim))
    
  def call(self, x, training=False):
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
    x=self.d(x)
    return x

class SymmetricEncoder1D(tf.keras.Model):
  def __init__(self, latent_dim):
    super(SymmetricEncoder1D, self).__init__(name = 'sym_encoder')
    self.c13=tfkltd(tfkl.Conv1D(32, 5, padding='same',activation='elu'))
    self.c14=tfkltd(tfkl.Conv1D(32, 5, padding='same',activation='elu'))
    self.mp12=tfkltd(tfkl.MaxPool1D(pool_size = 2))

    self.c23=tfkl.Conv1D(32, 5, padding='same',activation='elu')
    self.c24=tfkl.Conv1D(32, 5, padding='same',activation='elu')
    self.bn=tfkl.BatchNormalization()
    self.a=tfkl.Activation('elu')
    self.mp22=tfkl.MaxPool1D(pool_size = 2)
    self.f=tfkl.Flatten()
    self.d=tfkl.Dense(latent_dim)

  def call(self, x, training=False):
    x=self.c13(x)
    x=self.c14(x)
    x=self.mp12(x)
    x=tf.math.reduce_mean(x,axis=1)
    x=self.c23(x)
    x=self.c24(x)
    x=self.bn(x, training=training)
    x=self.a(x)
    x=self.mp22(x)
    x=self.f(x)
    x=self.d(x)
    return x

class Mixer1D(tf.keras.Model):
  def __init__(self, nt):
    super(Mixer1D, self).__init__(name = 'mixer')
    self.d1=tfkltd(tfkl.Dense(units=10, activation='elu'))
    self.r=tfkltd(tfkl.Reshape((10,1)))
    self.us1=tfkltd(tfkl.UpSampling1D(size=(10)))
    self.c1=tfkltd(tfkl.Conv1D(64, 5, padding='same',activation='elu'))
    self.c2=tfkltd(tfkl.Conv1D(64, 5, padding='same',activation='elu'))
    self.c3=tfkltd(tfkl.Conv1D(64, 5, padding='same',activation='elu'))
    self.c6=tfkltd(tfkl.Conv1D(1, 5, padding='same'))

  def call(self, x, training=False):
    x=self.d1(x)
    x=self.r(x)
    x=self.c1(x)
    x=self.us1(x)
    x=self.c2(x)
    x=self.c3(x)
    x=self.c6(x)
    return x

class DistributeZsym(tf.keras.Model):
  def __init__(self, ntau, nz0, nzi):
    super(DistributeZsym, self).__init__(name = 'dist')
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
