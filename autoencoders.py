import tensorflow as tf
import tensorflow.keras.layers as tfkl
from helpers import AudioGenerator, load_wavs_from_folders, setup_tf, load_model
from losses import SpectralLoss
from save_checkpoints import SaveCheckpoints
from IPython import embed
import importlib

def make_autoencoder(model_name, mode='load', model_weights=None, train_df=None, train_params=None, model_params=None):

    setup_tf()
    train_params_ = {
        'batch_size': 3,
        'input_size': 16384,
        'epochs': 100,
        'loss': SpectralLoss(),
        'optimizer': tf.keras.optimizers.RMSprop(clipnorm=0.1,learning_rate=0.001)
        }

    if train_params:
        train_params_.update(train_params)

    model_module = importlib.import_module('models.{}'.format(model_name))
    model_fn = getattr(model_module,'build_model')
    if model_params:
        ae_model, enc_model, dec_model = model_fn(**model_params)
    else:
        ae_model, enc_model, dec_model = model_fn()
    
    if mode=='train':
        #optimizer = tf.keras.optimizers.RMSprop(clipnorm=0.1,learning_rate=0.001)
        ae_model.compile(optimizer=train_params_['optimizer'],loss=train_params_['loss'])
        ae_model.summary()

        #df_drums = load_wavs_from_folders(['../data/drums'])
        ae_model.fit(AudioGenerator(train_df,input_size=train_params_['input_size'],batch_size=train_params_['batch_size']),epochs=train_params_['epochs'],callbacks=[SaveCheckpoints(monitor_metric='loss',save_best=True)])

    elif mode=='load':
        load_model([ae_model,enc_model,dec_model],model_weights)
        return ae_model, enc_model, dec_model

