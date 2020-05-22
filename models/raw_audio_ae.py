import tensorflow as tf
import tensorflow.keras.layers as tfkl

def build_model(wav_size=16384,chunk_size=256,chunk_hop=128,emb_size=32):
    N_WIN = (wav_size-chunk_size)//chunk_hop + 1

    hann_win = tf.signal.hann_window(chunk_size)
    wav_input = tf.keras.Input(shape=(wav_size,))

    encoder = tf.signal.frame(wav_input,chunk_size,chunk_hop,axis=1)
    encoder = tf.multiply(encoder,hann_win)
    encoder = tf.expand_dims(encoder,axis=-1)
    encoder = tfkl.TimeDistributed(tfkl.Conv1D(64,8,padding='SAME'))(encoder)
    encoder = tfkl.TimeDistributed(tfkl.Conv1D(64,8,padding='SAME',strides=8))(encoder)
    encoder = tfkl.TimeDistributed(tfkl.Conv1D(64,8,padding='SAME',strides=4))(encoder)
    encoder = tfkl.TimeDistributed(tfkl.Conv1D(64,8,padding='SAME',strides=2))(encoder)
    encoder = tfkl.TimeDistributed(tfkl.Flatten())(encoder)
    encoder = tfkl.GRU(emb_size,return_sequences=True)(encoder) #Global LSTM

    encoder_model = tf.keras.Model(wav_input,encoder)

    decoder_input = tf.keras.Input(shape=(N_WIN,emb_size), name='encoded_img')

    decoder = tfkl.GRU(256,return_sequences=True)(decoder_input)
    decoder = tfkl.TimeDistributed(tfkl.Reshape(target_shape=(4,64)))(decoder)
    decoder = tfkl.TimeDistributed(tfkl.Conv1D(64,8,padding='SAME'))(decoder)
    decoder = tfkl.TimeDistributed(tfkl.UpSampling1D(2))(decoder)
    decoder = tfkl.TimeDistributed(tfkl.Conv1D(64,8,padding='SAME'))(decoder)
    decoder = tfkl.TimeDistributed(tfkl.UpSampling1D(4))(decoder)
    decoder = tfkl.TimeDistributed(tfkl.Conv1D(64,8,padding='SAME'))(decoder)
    decoder = tfkl.TimeDistributed(tfkl.UpSampling1D(8))(decoder)
    decoder = tfkl.TimeDistributed(tfkl.Conv1D(1,8,padding='SAME',activation='tanh',dtype=tf.float32))(decoder)
    decoder = tfkl.Reshape(target_shape = (N_WIN,chunk_size))(decoder)   
    decoder = tf.multiply(decoder,hann_win)
    decoder = tf.signal.overlap_and_add(decoder,chunk_hop)

    decoder_model = tf.keras.Model(decoder_input,decoder)

    ae_input = tf.keras.Input(shape=(wav_size,))
    encoder_out = encoder_model(ae_input)
    decoder_out = decoder_model(encoder_out)
    
    ae_model = tf.keras.Model(inputs=ae_input,outputs=decoder_out)

    return ae_model, encoder_model, decoder_model
