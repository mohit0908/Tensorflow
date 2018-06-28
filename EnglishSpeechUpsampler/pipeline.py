import tensorflow as tf



def decode_csv(line):
    """
    Args:
    Line : Line item from a csv file
    Returns:
    A parsed csv line 
    """
#   Taking 1st column only
    parsed_line = tf.decode_csv(line, [['string'], ['string']])
    return parsed_line[0]


def audiofile_to_input_vector(truth_file):
    """
    Calculates a feature vector given a filename

    Args:
    audio_filename : Name of the audio file in wav format
    numcep : Number of cepstral coeffs to return

    Returns:
    A feature vector of size [num of frames, numcep]
    """
    # Load truth wav files (1st column)
    binary_truth = file_io.FileIO(truth_file, 'r')
    fs_truth, audio_truth = wav.read(binary)
    print(tf.size(audio_truth))
    
    return audio_truth


def input_parser(truth_file):
    
    """
    Helper function for feature extraction in tensorflow
    """
    audio_array = tf.py_func(audiofile_to_input_vector,[truth_file], [tf.float32])
    print('Details of audio file:',audio_array[0].shape, type(audio_array))
    return audio_array


def train_input(data_filepath, batch_size, num_epoch):
    """
    Function to create batches for inference using Tensorflow
    """
#     dataset = tf.data.TextLineDataset(data_filepath).map(decode_csv,num_parallel_calls=cpu_count())
    dataset = tf.data.TextLineDataset(data_filepath).map(decode_csv)
#     dataset = dataset.map(input_parser,num_parallel_calls=cpu_count())

    dataset = dataset.map(input_parser)
    dataset = dataset.apply(tf.contrib.data.ignore_errors())
    dataset = dataset.repeat(num_epoch)
    # How to pad file
    dataset = dataset.padded_batch(batch_size, padded_shapes=([],[],[,None],[]))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(2)
    iterator = dataset.make_one_shot_iterator()
    batch_feats = iterator.get_next()
    return batch_feats

path = 'filename.csv'


train_input(path, 4,5)

