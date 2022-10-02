import tensorflow as tf
tfk = tf.keras
tfkl = tf.keras.layers
tfkltd= tf.keras.layers.TimeDistributed
layers = tf.keras.layers
from keras.utils import control_flow_util
from keras import backend
from keras.engine import base_layer
class Dropout(base_layer.BaseRandomLayer):
    def __init__(self, rate, seed=None, **kwargs):
        super(Dropout, self).__init__(seed=seed, **kwargs)
        self.supports_masking = True
        self.rate = rate
        self.seed = seed
    def call(self, inputs, training=None):
        def noised():
            stddev = np.sqrt(self.rate / (1.0 - self.rate))
            return inputs * tf.random.normal(
                shape=tf.shape(inputs),
                mean=1.0,
                stddev=stddev,
                dtype=inputs.dtype)
        return backend.in_train_phase(noised, inputs, training=training)
    def get_config(self):
        config = {'rate': self.rate, 'seed': self.seed}
        base_config = super(Dropout, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    def compute_output_shape(self, input_shape):
        return input_shape


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
    return x

class DecDense1(tf.keras.Model):
  def __init__(self):
    super(DecDense1, self).__init__(name = 'DecDense1')
    self.d1=tfkltd(tfkl.Dense(1000, activation='elu'))
    self.d2=tfkltd(tfkl.Dense(1000, activation='elu'))
    self.r1=tfkltd(tfkl.Reshape(target_shape=(-1,1)))
    self.c1=tfkltd(tfkl.Conv1D(64,5,padding='same',activation='elu'))
    self.c2=tfkltd(tfkl.Conv1D(16,5,padding='same',activation='elu'))
    self.c3=tfkltd(tfkl.Conv1D(16,5,padding='same',activation='elu'))
    self.c4=tfkltd(tfkl.Conv1D(1,5,padding='same'))

  def call(self, x, training=False):
    x=self.d1(x)
    x=self.d2(x)
    x=self.r1(x)
    x=self.c1(x)
    x=self.c2(x)
    x=self.c3(x)
    x=self.c4(x)
    x=x[:,:,:,0]
    return x

class DecDense2(tf.keras.Model):
  def __init__(self):
    super(DecDense2, self).__init__(name = 'DecDense2')
    self.d1=tfkltd(tfkl.Dense(100))
    self.d2=tfkltd(tfkl.Dense(3))
    self.d3=tfkltd(tfkl.Dense(10))
    self.drop1=tfkl.Dropout(0.5)
    self.d4=tfkltd(tfkl.Dense(100))
    self.d5=tfkltd(tfkl.Dense(3))
    self.d6=tfkltd(tfkl.Dense(10))
    self.drop2=tfkl.Dropout(0.5)
    self.d7=tfkltd(tfkl.Dense(100))
    self.d8=tfkltd(tfkl.Dense(3))
    self.d9=tfkltd(tfkl.Dense(10))
    self.drop3=tfkl.Dropout(0.5)
    self.d10=tfkltd(tfkl.Dense(1000))
    self.d11=tfkltd(tfkl.Dense(1000))
    
  def call(self, x, training=False):
    x=self.d1(x)
    x=self.d2(x)
    x=self.d3(x)
    x=self.drop1(x)
    x=self.d4(x)
    x=self.d5(x)
    x=self.d6(x)
    x=self.drop2(x)    
    x=self.d7(x)
    x=self.d8(x)
    x=self.d9(x)
    x=self.drop3(x)    
    x=self.d10(x)
    x=self.d11(x)
    return x

class DecDense(tf.keras.Model):
  def __init__(self):
    super(DecDense, self).__init__(name = 'DecDense')
    self.d1=tfkltd(tfkl.Dense(1000, activation='elu'))
    self.d2=tfkltd(tfkl.Dense(100, activation='elu'))
    self.r1=tfkltd(tfkl.Reshape(target_shape=(100,1)))
    self.c1=tfkltd(tfkl.Conv1D(64,7,padding='same',activation='elu'))
    self.c2=tfkltd(tfkl.Conv1D(64,7,padding='same',activation='elu'))
    self.c3=tfkltd(tfkl.Conv1D(64,7,padding='same',activation='elu'))
    self.c4=tfkltd(tfkl.Conv1D(1,7,padding='same'))

  def call(self, x, training=False):
    x=self.d1(x)
    x=self.d2(x)
    x=self.r1(x)
    x=self.c1(x)
    x=self.c2(x)
    x=self.c3(x)
    x=self.c4(x)
    return x