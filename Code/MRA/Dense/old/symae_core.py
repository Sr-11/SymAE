import tensorflow as tf
tfk = tf.keras
tfkl = tf.keras.layers
tfkltd= tf.keras.layers.TimeDistributed

class NuisanceEncoderDense1D(tf.keras.Model):
  def __init__(self, q, w=100):
    super(NuisanceEncoderDense1D, self).__init__(name='nui_encoder')
    self.d1=tfkltd(tfkl.Dense(w,activation=tf.keras.layers.LeakyReLU()))
    self.d2=tfkltd(tfkl.Dense(q))
  def call(self, x, training=False):
    x=self.d1(x)
    x=self.d2(x)
    return x

class SymmetricEncoderDense1D(tf.keras.Model):
  def __init__(self, p, w=100):
    super(SymmetricEncoderDense1D, self).__init__(name='sym_encoder')
    self.d1=tfkltd(tfkl.Dense(w,activation=tf.keras.layers.LeakyReLU()))
    self.d2=tfkltd(tfkl.Dense(w))
    self.d3=tfkl.Dense(w,activation=tf.keras.layers.LeakyReLU())
    self.d4=tfkl.Dense(p)
  def call(self, x, training=False):
    x=self.d1(x)
    x=self.d2(x)
    x=tf.math.reduce_mean(x,axis=1)
    x=self.d3(x)
    x=self.d4(x)
    return x

class DistributeZsym(tf.keras.Model):
  def __init__(self, ntau, nz0, nzi):
    super(DistributeZsym, self).__init__(name='dist')
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
    super(LatentCat, self).__init__(name='latentcat')
    # self.drop=tfkl.GaussianDropout(alpha)
    self.drop=tfkl.Dropout(alpha)
  def call(self, zsym, znuisance,training=False):
    znuisance=self.drop(znuisance,training=training)
    z=tfkl.concatenate([zsym, znuisance])
    return z
                   
class MyLatentCat(tf.keras.Model):
    def __init__(self, alpha=1.0):
        super(MyLatentCat, self).__init__(name='latentcat')
        self.drop = tfkl.Dropout(alpha)
    def call(self, zsym, znuisance,training=False):
        znuisance = self.drop(znuisance, training=training)
        znuisance = tfkl.Flatten()(znuisance)
        z = tfkl.concatenate([zsym, znuisance])
        return z

class MixerDense1D(tf.keras.Model):
  def __init__(self, d, w=100):
    super(MixerDense1D, self).__init__(name='mixer')
    self.d1=tfkltd(tfkl.Dense(w,activation=tf.keras.layers.LeakyReLU()))
    self.d2=tfkltd(tfkl.Dense(d))
  def call(self, x, training=False):
    x=self.d1(x)
    x=self.d2(x)
    return x
