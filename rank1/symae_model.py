import tensorflow as tf
import symae_core as symae
tfk = tf.keras
tfkl = tf.keras.layers
tfkltd= tf.keras.layers.TimeDistributed

class SymAE(tf.keras.Model):
    def __init__(self,N=1000,nt=20,p=8,q=22,dropout_rate=0.65): 
        super(SymAE, self).__init__()
        
        # Build symmetric encoder
        sym_enc_input=tfk.Input(shape=(nt,1), dtype='float32', name='sym_enc_input')
        sym_enc_output=symae.SymmetricEncoderDense0D(p,nt)(sym_enc_input)
        sym_encoder=tfk.Model(sym_enc_input, sym_enc_output, name='sym_encoder')
        self.sym_encoder=sym_encoder
        
        # Build nuisance encoder
        nui_enc_input=tfk.Input(shape=(nt,1), dtype='float32', name='sym_enc_input')
        nui_enc_output=symae.NuisanceEncoderDense0D(q,nt)(nui_enc_input)
        nui_enc_flatten=tfkl.Flatten()(nui_enc_output)
        nui_encoder=tfk.Model(nui_enc_input, nui_enc_flatten, name='nui_encoder')
        self.nui_encoder=nui_encoder
        
        # Build distribute in decoder
        distribute_input = tfk.Input(shape=(p+q*nt), name='latentcode')
        distribute_output=symae.DistributeZsym(nt, p, q)(distribute_input)
        distzsym = tfk.Model(distribute_input, distribute_output, name='distzsym')
        self.distzsym = distzsym
        
        #Build mixer in decoder
        mixer_input = tfk.Input(shape=(nt,p+q), name='mixer_input')
        mixer_output=symae.MixerDense0D(1,p+q)(mixer_input)
        mixer = tfk.Model(mixer_input, mixer_output, name='mixer') 
        self.mixer = mixer
        
        # Build encoder
        encoder_input=tfk.Input(shape=(nt,1), dtype='float32', name='encoder_input')
        znuisance=nui_encoder(encoder_input)
        zsym=sym_encoder(encoder_input)
        latentcat=symae.LatentCat(dropout_rate) #0.4 droupout
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
