import numpy as np
import os
import json
import librosa
import tensorflow as tf
import losses
from optimizers import make_variable_learning_rate, setup_optimizer
import models
import models_exp
import pandas as pd
import datetime
import pipeline
import time
from tensorflow.python.saved_model import tag_constants

#custom_shuffle_module = tf.load_op_library('src/shuffle_op.so')
#shuffle = custom_shuffle_module.shuffle

try:
    os.makedirs('./aux/checkpoint')
except Exception: 
    pass

data_settings_file = 'settings/data_settings.json'
training_settings_file = 'settings/training_settings.json'
model_settings_file = 'settings/model_settings.json'

data_settings = json.load(open(data_settings_file))
training_settings = json.load(open(training_settings_file))
model_settings = json.load(open(model_settings_file))

# Constants describing the training process.
# Samples per batch.
BATCH_SIZE = training_settings['batch_size']
# Number of epochs to train.
NUMBER_OF_EPOCHS = training_settings['number_of_epochs']
# Epochs after which learning rate decays.
NUM_EPOCHS_PER_DECAY = training_settings['num_epochs_per_decay']
# Learning rate decay factor.
LEARNING_RATE_DECAY_FACTOR = training_settings['learning_rate_decay_factor']
# Initial learning rate.
INITIAL_LEARNING_RATE = training_settings['initial_learning_rate']

example_number = 0
write_tb = False

file_name_lists_dir = data_settings['output_dir_name_base']
train_filepath =  './data/preprocessed/train_files.csv'
validation_filepath = './data/preprocessed/validation_files.csv'

def read_csv(filepath):
    df = pd.read_csv(filepath)
    return len(df)



SAMPLES_PER_EPOCH_TRAIN = read_csv(train_filepath)
SAMPLES_PER_EPOCH_VALID = read_csv(validation_filepath)


 #### my code #### 
current_data = []



with tf.Graph().as_default():
    global_step = tf.train.get_or_create_global_step()
    saver = tf.train.Saver() 
    with tf.Session() as sess:
               
#         Training pipeline
        inp = pipeline.train_input( train_filepath, BATCH_SIZE, NUMBER_OF_EPOCHS )


        up_audio , down_audio = inp

        up_audio_eval =  sess.run(up_audio)
        
        print('Model called')
        train_flag, x , up_audio_return_y = models.train( input_data = up_audio , 
                                                                input_shape = up_audio_eval.shape)
        print('Model created')
#         Calculate loss.

        loss = losses.mse("loss",up_audio, up_audio_return_y)

#         Variable that affect learning rate.
        num_batches_per_epoch_train = int(SAMPLES_PER_EPOCH_TRAIN/BATCH_SIZE)
        print('num_batches_per_epoch:',num_batches_per_epoch_train)
        decay_steps = int(num_batches_per_epoch_train * NUM_EPOCHS_PER_DECAY)


        # Decay the learning rate based on the number of steps.
        lr, global_step = make_variable_learning_rate(INITIAL_LEARNING_RATE,
                                                      decay_steps,
                                                      LEARNING_RATE_DECAY_FACTOR,
                                                      False)

        min_args = {'global_step': global_step}

#             Defining optimizer
        train_step = setup_optimizer(lr, loss, tf.train.AdamOptimizer,
                                     using_batch_norm=True,
                                     min_args=min_args)
        
        
#             Training Loop
        try:
        
            sess.run(tf.global_variables_initializer())
            training_loss = open('train_loss.txt', 'w')
            valid_loss = open('validation_loss.txt', 'w')
            for i in range(NUMBER_OF_EPOCHS):
                for j in range(num_batches_per_epoch_train):
                    avg_loss = 0
                    try:
                        _, loss_value = sess.run([train_step, loss])
                        avg_loss +=  loss_value
                        if j % 50 == 0:
                            print( 'Training Loss in epoch {} and batch {} is {}:'.format(i+1,j+1, avg_loss/(j+1) ))
                    except tf.errors.OutOfRangeError :
                        print('Data used')
                        pass
                print( 'Training Loss in epoch {} is {}:'.format(i+1,avg_loss/(j+1) ))
                training_loss.write('Training Loss for Epoch {} is {}:\n'.format((i+1) , avg_loss/(j+1)))
                var_list = [v for v in tf.trainable_variables()]
                print('variable :',var_list[0],sess.run(var_list[0]).shape)
                print('Variable value:',sess.run(var_list[0]))
    #         Validation pipeline
                validation = pipeline.train_input( validation_filepath, BATCH_SIZE, 1 )
                up_audio_valid, down_audio_valid = validation

                train_flag_val, x_val , up_audio_return_y_valid = models.train( input_data = 
                                                                                                up_audio_valid , 
                                                                        input_shape = up_audio_eval.shape)
                loss_valid = losses.mse("loss_valid",up_audio_valid, up_audio_return_y_valid)
                num_batches_per_epoch_valid = int(SAMPLES_PER_EPOCH_VALID/BATCH_SIZE)

                sess.run(tf.global_variables_initializer())
                for j in range(num_batches_per_epoch_valid):
                    avg_loss = 0
                    try:
                        loss_value = sess.run([loss_valid])
                        avg_loss +=  loss_value[0]

                    except tf.errors.OutOfRangeError :
                        print('Data used')
                        pass
                print('Validation loss for epoch {} is {}:'.format(i+1, avg_loss/(j+1)))
                valid_loss.write('Validation Loss for Epoch {} is {}:\n'.format((i+1) , avg_loss/(j+1)))
                checkpoint_path = saver.save(sess, './aux/checkpoint/checkpoint{}.ckpt'.format(i+1),global_step=global_step)
                print('Checkpoint saved at:', checkpoint_path)
            training_loss.close()
            valid_loss.close()
        except Exception as e:
            print(e)
      

print('Process finished')


