import os
import json
import numpy as np
import librosa
from inputs import get_bit_rates_and_waveforms
from inputs import get_truth_ds_filename_pairs
import tensorflow as tf
from models import deep_residual_network
import pandas as pd


data_settings_file = 'settings/data_settings.json'
model_settings_file = 'settings/model_settings.json'
upsampling_settings_file = 'settings/upsampling_settings.json'

data_settings = json.load(open(data_settings_file))
model_settings = json.load(open(model_settings_file))
upsampling_settings = json.load(open(upsampling_settings_file))

file_name_lists_dir = os.getcwd() + data_settings['output_dir_name_base'].replace('..','')
upsample_csv = os.getcwd() + (upsampling_settings['input_file']).replace('..','') + upsampling_settings['filename']

source_dir = os.path.split(upsample_csv)[0]


END_OFFSET = data_settings['end_time']
upsampling_factor = 2
INPUT_SIZE = int(data_settings['downsample_rate'])*upsampling_factor



model_checkpoint_file_name = os.getcwd() + upsampling_settings['model_checkpoint_file']


# Iterate through complete upsampling list and convert
df = pd.read_csv(upsample_csv, header = None)
for index in range(len(df)):
    # Load 8k file
    true_wf, true_br = librosa.load(df.iloc[index,0], sr=None, mono=True)

    # Upsampled file. This will be fed into Deep Neural Network
    us_wf = librosa.core.resample(true_wf, true_br, true_br*upsampling_factor)
    us_br = true_br*upsampling_factor
    number_of_reco_iterations = int(us_wf.size/INPUT_SIZE)


    # ################
    # MODEL DEFINITION
    # ################
    tf.reset_default_graph()
    train_flag, x, model = deep_residual_network(true_wf.dtype,
                                                 np.reshape(us_wf, (-1,1))[:us_br].shape,
                                                 **model_settings)

    # ################
    # ################


    # Add ops to restore all the variables.
    saver = tf.train.Saver()

    # create session and restore model
    sess = tf.Session()
    saver.restore(sess, model_checkpoint_file_name)

    # ###################
    # RECONSTRUCTION LOOP
    # ###################

    reco_wf = np.empty(us_wf.size)

    # Calculate number of samples in test file
    for i in range(number_of_reco_iterations):
        print('Segement {} of {}'.format(i + 1, number_of_reco_iterations))
    # Feeding 
        example = us_wf[i*INPUT_SIZE:(i + 1)*INPUT_SIZE]
#        print('File ', df.iloc[index,0], 'getting processed')
        reco_wf[i*INPUT_SIZE:(i + 1)*INPUT_SIZE] = model.eval(feed_dict={train_flag: False,x: example.reshape(1, -1, 1)},session=sess).flatten()

    file_name_base = df.iloc[index,0].split('/')[-1]
#    librosa.output.write_wav(os.path.join(source_dir, 'true_' + file_name_base),
#                             y=true_wf.flatten(), sr=true_br)
    try:librosa.output.write_wav(os.path.join(source_dir, 'reco_' + file_name_base),y=reco_wf.flatten(), sr=us_br)
    except Exception:pass
