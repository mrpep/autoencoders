import tensorflow as tf
import tensorflow.keras.layers as tfkl
from helpers import AudioGenerator, load_wavs_from_folders, setup_tf, load_model
from losses import SpectralLoss
from save_checkpoints import SaveCheckpoints
from IPython import embed
import importlib

def make_autoencoder(model_name, mode='load', model_weights=None):

    setup_tf()
    model_module = importlib.import_module('models.{}'.format(model_name))
    model_fn = getattr(model_module,'build_model')
    ae_model, enc_model, dec_model = model_fn()
    
    if mode=='train':
        optimizer = tf.keras.optimizers.RMSprop(clipnorm=0.1,learning_rate=0.001)
        ae_model.compile(optimizer=optimizer,loss=SpectralLoss())
        ae_model.summary()

        df_drums = load_wavs_from_folders(['../data/drums'])
        ae_model.fit(AudioGenerator(df_drums,input_size=WAV_SIZE,batch_size=3),epochs=100,callbacks=[SaveCheckpoints(monitor_metric='loss',save_best=True)])

    elif mode=='load':
        load_model([ae_model,enc_model,dec_model],model_weights)
        return ae_model, enc_model, dec_model

