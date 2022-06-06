import tensorflow as tf
import symae_core as symae
tfk = tf.keras
tfkl = tf.keras.layers
tfkltd= tf.keras.layers.TimeDistributed
class SymAE(tf.keras.Model):
    def __init__(self,N=1000,nt=20,d=100,p=8,q=22,kernel_size=5,filters=64): 
        '''
        class SymAE is the complete AutoEncoder

        Parameters
        ----------
        # Parameters related to data itself
        N : int
            N is the Cardinality of the data set X, say n_X in the paper
        nt : int
            nt is the number of instances in each X_i, say n_tau in the paper (Xi[1]...Xi[nt])
        d : int
            d is the dimensions of each "fundamental" data, say d=dim Xi[j] (e.g. d=28*28 for mnist)
        # Parameters related to SymAE
        p : int
            Symmetric encoder latent dimensions, p=dim Ci (Coherent Code)
        q : int
            Nuisance encoder latent dimensions, q=dim Ni[j] (Dissimilar Code)
        # Parameters related to NeuralNets
        kernel_size : int
            The size of the convolution window
        filters : int 
            How many filters in each convolution layer
        '''
        super(SymAE, self).__init__()
        # Build symmetric encoder
        sym_enc_input=tfk.Input(shape=(nt,d,1), dtype='float32', name='sym_enc_input')
        sym_enc_output=symae.SymmetricEncoder1D(kernel_size,filters,[2,2,2,2],[2,2,2,2],p)(sym_enc_input)
        sym_encoder=tfk.Model(sym_enc_input, sym_enc_output, name='sym_encoder')
        self.sym_encoder=sym_encoder
        # Build nuisance encoder
        nui_enc_input=tfk.Input(shape=(nt,d,1), dtype='float32', name='sym_enc_input')
        nui_enc_output=symae.NuisanceEncoder1D(kernel_size,filters,[2,2,2],[2,2,2,2],q)(nui_enc_input)
        nui_enc_flatten=tfkl.Flatten()(nui_enc_output)
        nui_encoder=tfk.Model(nui_enc_input, nui_enc_flatten, name='nui_encoder')
        self.nui_encoder=nui_encoder
        # Build distribute in decoder
        distribute_input = tfk.Input(shape=(p+q*nt), name='latentcode')
        distribute_output=symae.DistributeZsym(nt, p, q)(distribute_input)
        distzsym = tfk.Model(distribute_input, distribute_output, name='distzsym')
        #Build mixer in decoder
        mixer_input = tfk.Input(shape=(nt,p+q), name='mixer_input')
        mixer_output=symae.Mixer1D(kernel_size,filters,10,d)(mixer_input)
        mixer = tfk.Model(mixer_input, mixer_output, name='mixer') 
        # Build encoder
        encoder_input=tfk.Input(shape=(nt,d,1), dtype='float32', name='encoder_input')
        znuisance=nui_encoder(encoder_input)
        zsym=sym_encoder(encoder_input)
        latentcat=symae.LatentCat(0.65) #0.4 droupout
        self.latentcat=latentcat
        encoder_output=latentcat(zsym,znuisance)
        encoder=tfk.Model(encoder_input, encoder_output, name="encoder")
        self.encoder=encoder
        # Build decoder
        decoder_input = tfk.Input(shape=(p+q*nt), name='latentcode')
        decoder_output=mixer(distzsym(decoder_input))
        decoder=tfk.Model(decoder_input,decoder_output, name="decoder") 
        self.decoder=decoder
        #Build SymAE
        self.symae=tf.keras.Model(encoder_input, decoder(encoder_output) , name='autoencoder_clone')
    def call(self, input_tensor):
        return self.symae(input_tensor)
    def model(self, x):
        return tfk.Model(inputs=x, outputs=self.call(x))
