import os
import csv
import json
import numpy as np
import pandas as pd

settings_file = '../settings/data_settings.json'
upsampling_file = '../settings/upsampling_settings.json'


settings = json.load(open(settings_file))
upsampling_settings = json.load(open(upsampling_file))

file_dir_base = os.path.abspath(settings['output_dir_name_base'])
output_dir = os.path.abspath(settings['output_dir_name_base'])
upsampling_csv = os.path.abspath(upsampling_settings['input_file'])
print(upsampling_csv)




validation_fraction = settings['validation_fraction']
test_fraction = settings['test_fraction']


np.random.seed(0)


def write_csv(filename, pairs):
    with open(filename, 'w') as csvfile:
        spamwriter = csv.writer(csvfile)
        for pair in pairs:
            spamwriter.writerow(pair)


if not os.path.exists(output_dir):
    os.makedirs(output_dir)

truth_ds_pairs = []
file_dir_truth = os.path.join(file_dir_base, 'splices')
file_dir_ds = os.path.join(file_dir_base, 'downsampled_splices')

for filename in os.listdir(file_dir_truth):
    truth_input_filename = os.path.join(file_dir_truth, filename)
    ds_input_filename = os.path.join(file_dir_ds, filename)
    if not os.path.isfile(truth_input_filename) or not \
            os.path.isfile(ds_input_filename):
        continue
    truth_ds_pairs.append([truth_input_filename, ds_input_filename])

np.random.shuffle(truth_ds_pairs)

validation_start_index = 0
validation_end_index = validation_start_index +\
    int(len(truth_ds_pairs)*validation_fraction)
test_start_index = validation_end_index
test_end_index = test_start_index +\
    int(len(truth_ds_pairs)*validation_fraction)
train_start_index = test_end_index

validation_truth_ds_pairs =\
    truth_ds_pairs[validation_start_index:validation_end_index]
write_csv(os.path.join(output_dir, 'validation_files.csv'),
          validation_truth_ds_pairs)

test_truth_ds_pairs = truth_ds_pairs[test_start_index:test_end_index]
write_csv(os.path.join(output_dir, 'test_files.csv'), test_truth_ds_pairs)

train_truth_ds_pairs = truth_ds_pairs[train_start_index:]
write_csv(os.path.join(output_dir, 'train_files.csv'), train_truth_ds_pairs)



# Write upsampling filenames to csv

filelist = []

for name in os.listdir(upsampling_csv):
	if name.endswith('.wav'):
		filelist.append(upsampling_csv + '/' + name)

df = pd.DataFrame(filelist, columns = ['name'])
df.to_csv(upsampling_csv+'/upsampling.csv', index = False, header = False)
