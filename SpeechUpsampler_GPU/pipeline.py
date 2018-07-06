import tensorflow as tf
import scipy.io.wavfile as wav
from tensorflow.python.lib.io import file_io
import numpy as np

def decode_csv(line):

    parsed_line = tf.decode_csv(line, [['string'], ['string']])
    feat = parsed_line[0]
    lab = parsed_line[-1]

    return feat, lab

def audiofile_to_input_vector(truth_file):

#    Load wav files (1st column)
#    binary_truth = file_io.FileIO(truth_file, 'b')
    fs_truth, audio_truth = wav.read(truth_file)    
    return audio_truth.astype( np.float32 ).reshape(-1,1) 


def input_parser(truth_file, ds_file):
 
    truth_audio_array = tf.py_func(audiofile_to_input_vector,[truth_file], [tf.float32])
    ds_audio_array = tf.py_func(audiofile_to_input_vector,[ds_file], [tf.float32])     
    
    return truth_audio_array[0] , ds_audio_array[0] 


# Training pipeline

def train_input(data_filepath, batch_size, num_epoch):
   

#   num_parallel_calls = cpu_count()
    dataset = tf.data.TextLineDataset(data_filepath).map(decode_csv, num_parallel_calls = 1)
    dataset = dataset.map(input_parser,num_parallel_calls=4)
#    dataset = dataset.apply(tf.contrib.data.ignore_errors())
    dataset = dataset.repeat(num_epoch)
    dataset = dataset.padded_batch( batch_size, padded_shapes=( [None,1], [None,1] ) )
#     dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    upsampled, downsampled = iterator.get_next()
    return upsampled, downsampled


# Inference pipeline
def decode_csv_inference(line):

    parsed_line = tf.decode_csv(line, [['string']])
    wave_name = parsed_line[0]

    return wave_name



def audiofile_to_input_vector_inference(truth_file):

#    Load wav files (1st column)
#    binary_truth = file_io.FileIO(truth_file, 'b')
    fs_truth, audio_truth = wav.read(truth_file)    
    return audio_truth.astype( np.float32 ).reshape(-1,1) 


def input_parser_inference(truth_file):
 
    truth_audio_array = tf.py_func(audiofile_to_input_vector_inference,[truth_file], [tf.float32])    
    
    return truth_audio_array[0]


# Training pipeline

def inference_input(data_filepath):
   

#   num_parallel_calls = cpu_count()
    dataset = tf.data.TextLineDataset(data_filepath).map(decode_csv_inference, num_parallel_calls = 1)
    dataset = dataset.map(input_parser_inference,num_parallel_calls=4)
    iterator = dataset.make_one_shot_iterator()
    waveform = iterator.get_next()
    return waveform
