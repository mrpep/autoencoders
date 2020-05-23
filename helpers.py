from pathlib import Path
import pandas as pd
import soundfile as sf
import numpy as np
import tensorflow as tf
import librosa
from tensorflow.python.framework.ops import disable_eager_execution
import importlib
import joblib
import sys

def load_wavs_from_folders(paths):
  all_rows = []

  if not isinstance(paths,list):
    paths = [paths]

  for path in paths:
    for wavfile in Path(path).rglob('*.wav'):
      try:
        metadata = sf.info(wavfile)
        row_i = {'path': wavfile,
                'sr': metadata.samplerate,
                'channels': metadata.channels,
                'duration': metadata.duration,
                'folder': path}
        all_rows.append(row_i)
      except:
        print('Could not load {}'.format(wavfile))
  return pd.DataFrame(all_rows)

def setup_tf():
    disable_eager_execution()

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
      try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
          tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
      except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

def load_model(models,weight_file):
    weights = joblib.load(weight_file)
    for model in models:
        for layer in model.layers:
            if layer.name in weights:
                layer.set_weights(weights[layer.name])

class AudioGenerator(tf.keras.utils.Sequence):
  def __init__(self,df,batch_size=16,input_size=65536,sr=44100,shuffle=True):
    self.data = df
    self.batch_size = batch_size
    self.sr = sr
    self.input_size = input_size
    self.shuffle = shuffle
    self.idxs = np.array(df.index)
    self.on_epoch_end()
    

  def __getitem__(self,idx):
    batch_idxs = np.take(self.idxs,np.arange(idx*self.batch_size,(idx+1)*self.batch_size),mode='wrap')
    batch_audios = self.df_to_audio(self.data.loc[batch_idxs])
    batch_audios = np.stack(batch_audios.values)

    return (batch_audios,batch_audios)

  def df_to_audio(self,df):

    def load_audios(row, fixed_size = 65536, sr = 44100):
      x,fs = librosa.core.load(row['path'],sr=sr,mono=True)
      if len(x)>fixed_size:
        y = x[:fixed_size]
      else:
        y = np.zeros((fixed_size,))
        y[:len(x)] = x
    
      return y
    
    return df.apply(load_audios,axis=1,fixed_size=self.input_size,sr = self.sr)

  def on_epoch_end(self):
    if self.shuffle:
      self.idxs = np.random.permutation(self.idxs)

  def __len__(self):
    return len(self.idxs)//self.batch_size
