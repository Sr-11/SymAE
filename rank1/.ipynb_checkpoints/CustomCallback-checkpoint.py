from IPython.display import clear_output
import tensorflow as tf
class CustomCallback(tf.keras.callbacks.Callback):
    '''
    Fix a bug of tensorflow
    When using model.fit(verbose=1 or 2 or 3), 
    Keras training progress keeps writing a new line in terminal on each epoch.
    Solved by using model.fit(verbose=0,callbacks=[CustomCallback()])
    '''
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        print("For epoch {:d}, loss is {:f}.".format(epoch, logs["loss"]))