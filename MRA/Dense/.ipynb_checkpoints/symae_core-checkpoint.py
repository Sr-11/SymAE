import tensorflow as tf
tfk = tf.keras
tfkl = tf.keras.layers
tfkltd= tf.keras.layers.TimeDistributed

class NuisanceEncoder1D(tf.keras.Model):
  def __init__(self, latent_dim):
    super(NuisanceEncoder1D, self).__init__(name = 'nui_encoder')
    self.c1=tfkltd(tfkl.Conv1D(32, 5, padding='same', activation='elu'))
    self.c2=tfkltd(tfkl.Conv1D(32, 5, padding='same', activation='elu'))
    self.mp1=tfkltd(tfkl.MaxPool1D(pool_size = 2))
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
    x=self.c1(x)
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
    x=self.d(x)
    return x

class SymmetricEncoder1D(tf.keras.Model):
  def __init__(self, latent_dim):
    super(SymmetricEncoder1D, self).__init__(name = 'sym_encoder')
    self.c11=tfkltd(tfkl.Conv1D(32, 5, padding='same',activation='elu'))
    self.c12=tfkltd(tfkl.Conv1D(32, 5, padding='same',activation='elu'))
    self.mp11=tfkltd(tfkl.MaxPool1D(pool_size = 2))
    self.c13=tfkltd(tfkl.Conv1D(32, 5, padding='same',activation='elu'))
    self.c14=tfkltd(tfkl.Conv1D(32, 5, padding='same',activation='elu'))
    self.mp12=tfkltd(tfkl.MaxPool1D(pool_size = 2))

    self.c21=tfkl.Conv1D(32, 5, padding='same',activation='elu')
    self.c22=tfkl.Conv1D(32, 5, padding='same',activation='elu')
    self.mp21=tfkl.MaxPool1D(pool_size = 2)
    self.c23=tfkl.Conv1D(32, 5, padding='same',activation='elu')
    self.c24=tfkl.Conv1D(32, 5, padding='same',activation='elu')
    self.bn=tfkl.BatchNormalization()
    self.a=tfkl.Activation('elu')
    self.mp22=tfkl.MaxPool1D(pool_size = 2)
    self.f=tfkl.Flatten()
    self.d=tfkl.Dense(latent_dim)

#     self.d1=tfkl.Dense(100)
#     self.d2=tfkl.Dense(1000)
#     self.d3=tfkl.Dense(1000)
#     self.d4=tfkl.Dense(1000)
#     self.d5=tfkl.Dense(1000)
#     self.d6=tfkl.Dense(1000)
#     self.d7=tfkl.Dense(d)
    
  def call(self, x, training=False):
    x=self.c11(x)
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
    x=self.d(x)
    #x1=tf.reduce_mean(x**2,axis=[1,2])
    #x2=tf.reduce_mean(x,axis=[1,2])
    #x=tf.concat([x1, x2], -1)
    # x=self.d1(x)
    # x=self.d2(x)
    # x=self.d3(x)
    # x=self.d4(x)
    # x=self.d5(x)
    # x=self.d6(x)
    # x=self.d7(x)
    return x

class Mixer1D(tf.keras.Model):
  def __init__(self, nt):
    super(Mixer1D, self).__init__(name = 'mixer')
    # self.d1=tfkltd(tfkl.Dense(units=10, activation='elu'))
    # self.r=tfkltd(tfkl.Reshape((10,1)))
    # self.c1=tfkltd(tfkl.Conv1D(64, 5, padding='same',activation='elu'))
    # self.us1=tfkltd(tfkl.UpSampling1D(size=(10)))
    # self.c2=tfkltd(tfkl.Conv1D(64, 5, padding='same',activation='elu'))
    # self.c3=tfkltd(tfkl.Conv1D(64, 5, padding='same',activation='elu'))
    # self.bn=tfkltd(tfkl.BatchNormalization())
    # self.a=tfkltd(tfkl.Activation('elu'))
    # self.c4=tfkltd(tfkl.Conv1D(64, 5, padding='same',activation='elu'))
    # self.c5=tfkltd(tfkl.Conv1D(64, 5, padding='same',activation='elu'))
    # self.c6=tfkltd(tfkl.Conv1D(1, 5, padding='same'))
    
    self.d1=tfkltd(tfkl.Dense(1000))
    self.d2=tfkltd(tfkl.Dense(1000))
    self.d3=tfkltd(tfkl.Dense(2000))
    self.d4=tfkltd(tfkl.Dense(3000))
    self.d5=tfkltd(tfkl.Dense(2000))
    self.d6=tfkltd(tfkl.Dense(1000))
    self.d7=tfkltd(tfkl.Dense(1000))
    self.d8=tfkltd(tfkl.Dense(100))
    
  def call(self, x, training=False):
    #x=self.d1(x)
    # x=self.r(x)
    # x=self.c1(x)
    # x=self.us1(x)
    # x=self.c2(x)
    # x=self.c3(x)
    # x=self.bn(x, training=training)
    # x=self.a(x)
    # x=self.c4(x)
    # x=self.c5(x)
    # x=self.c6(x)
    
    x=self.d1(x)
    x=self.d2(x)
    x=self.d3(x)
    x=self.d4(x)
    x=self.d5(x)
    x=self.d6(x)
    x=self.d7(x)
    x=self.d8(x)
    x=tf.expand_dims(x,-1)
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
