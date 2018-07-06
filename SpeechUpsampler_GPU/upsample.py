import os
import json
import numpy as np
import librosa
import tensorflow as tf
import pandas as pd
import models_inference as mi


data_settings_file = 'settings/data_settings.json'
model_settings_file = 'settings/model_settings.json'
upsampling_settings_file = 'settings/upsampling_settings.json'

data_settings = json.load(open(data_settings_file))
model_settings = json.load(open(model_settings_file))
upsampling_settings = json.load(open(upsampling_settings_file))

upsample_csv = upsampling_settings['input_file']+ upsampling_settings['filename']

source_dir = os.path.split(upsample_csv)[0]


END_OFFSET = data_settings['end_time']
upsampling_factor = 2
INPUT_SIZE = int(data_settings['downsample_rate'])*upsampling_factor



model_checkpoint_file_name = os.getcwd() + upsampling_settings['model_checkpoint_file']


# Iterate through complete upsampling list and convert
df = pd.read_csv(upsample_csv, header = None)

true_wf, true_br = librosa.load(df.iloc[0,0], sr=None, mono=True)

us_wf = librosa.core.resample(true_wf, true_br, true_br*upsampling_factor)
us_br = true_br*upsampling_factor

train_flag, x, model = mi.deep_residual_network(true_wf.dtype,
                                                 np.reshape(us_wf, (-1,1))[:us_br].shape,
                                                 **model_settings)
saver = tf.train.Saver()

corrupt_file = []
with tf.Session() as sess:
        # create session and restore model
    saver.restore(sess, model_checkpoint_file_name)

    for index in range(len(df)):
        # Load 8k file
        true_wf, true_br = librosa.load(df.iloc[index,0], sr=None, mono=True)

        # Upsampled file. This will be fed into Deep Neural Network
        us_wf = librosa.core.resample(true_wf, true_br, true_br*upsampling_factor)
        us_br = true_br*upsampling_factor
        number_of_reco_iterations = int(us_wf.size/INPUT_SIZE)



        # ###################
        # RECONSTRUCTION LOOP
        # ###################

        reco_wf = np.empty(us_wf.size)
        print('Processing:',df.iloc[index,0])
        # Calculate number of samples in test file
        for i in range(number_of_reco_iterations):
            print('Segement {} of {}'.format(i + 1, number_of_reco_iterations))
        # Feeding 
            example = us_wf[i*INPUT_SIZE:(i + 1)*INPUT_SIZE]
            reco_wf[i*INPUT_SIZE:(i + 1)*INPUT_SIZE] = model.eval(feed_dict={train_flag: False,x: example.reshape(1, -1, 1)},session=sess).flatten()

        file_name_base = df.iloc[index,0].split('/')[-1]
        
        try:
            librosa.output.write_wav(os.path.join(source_dir, 'reco_' + file_name_base),y=reco_wf, sr=us_br)
        except Exception:
            corrupt_file.append(df.iloc[index,0])
            pass
        
df = pd.DataFrame(corrupt_file)
corrupt_file.to_csv('currupt_files.csv', header = None, index = False)
