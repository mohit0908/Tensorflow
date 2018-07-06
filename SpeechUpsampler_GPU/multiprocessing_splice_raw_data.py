import os
import json
import tqdm
import sox
import time
import numpy as np
import multiprocessing as mp
from multiprocessing import Pool, Process

splice_settings_file = 'settings/data_settings.json'

settings = json.load(open(splice_settings_file))
input_data_suffix = settings['input_data_suffix']
input_dir_name_base = settings['input_dir_name_base']
input_dir_name_dirs = settings['input_dir_name_dirs']
splice_duration = settings['splice_duration']
downsample_rate = settings['downsample_rate']
output_dir_name_base = settings['output_dir_name_base']
upsampling_dir_name_base = settings['upsampling_dir_name_base']

output_dir_name = os.path.join(output_dir_name_base, 'splices')
ds_output_dir_name = os.path.join(output_dir_name_base, 'downsampled_splices')
us_output_dir_name = os.path.join(output_dir_name_base, 'upsampled_splices'


if not os.path.exists(output_dir_name):
    os.makedirs(output_dir_name)
if not os.path.exists(ds_output_dir_name):
    os.makedirs(ds_output_dir_name)
if not os.path.exists(upsampling_dir_name_base):
    os.makedirs(us_output_dir_name)


print('Will send spliced ground truth audio to {}'.format(output_dir_name))
print('Will send spliced input audio to' +' {}'.format(ds_output_dir_name))
print('Will send spliced test audio to' +' {}'.format(us_output_dir_name))

processed_data_info = settings
processed_data_info['original_bitrate'] = None

directory_list = []
file_list = []
for input_dir_name_dir in input_dir_name_dirs:
    input_dir_name = input_dir_name_base.format(input_dir_name_dir)
    directory_list.append(input_dir_name)
    for filename in os.listdir(input_dir_name):
        if filename.endswith('.wav'):
            file_list.append(os.path.join(input_dir_name, filename))    
array_split = np.array_split(file_list, mp.cpu_count())
        
def splitting(directory):
#         Loop over all files within the input directory
    for filename in directory:
#         input_filename = os.path.join(directory_name, filename)
        print(filename)
        filename_base = os.path.basename(filename)
        abspath = os.path.dirname(filename)
        duration = sox.file_info.duration(filename)
        n_iterations = int(duration/splice_duration)
        num_of_digits = len(str(int(duration)))
        num_format = '{{:0{}d}}'.format(num_of_digits)
        file_name_template = '{{}}_{}-{}.wav'.format(num_format, num_format)

        print('On file {}'.format(filename_base))
        for i in tqdm.trange(n_iterations):
            # create trasnformer
            splice = sox.Transformer()
            splice_and_downsample = sox.Transformer()
            begin = int(i*splice_duration)
            end = int(begin + splice_duration)

            output_filename = file_name_template.format(filename_base,begin, end)
            output_filename = os.path.join(output_dir_name, output_filename)

            ds_output_filename = file_name_template.format(filename_base,begin, end)
            ds_output_filename = os.path.join(ds_output_dir_name,ds_output_filename)


            if abspath.endswith('input_16k'):
                splice.trim(begin, end)
                splice.build(filename, output_filename)

            if abspath.endswith('input_8k'):
                splice_and_downsample.trim(begin, end)                    
#                     Use downsample line only if you have 16k wavefiles in input_8k folder and
#                     you want to downsample
#                 splice_and_downsample.convert(samplerate=downsample_rate)
                splice_and_downsample.build(filename, ds_output_filename)



# Process code
# jobs = []

# start = time.time()
# for batch in array_split:
#     p = Process(target = splitting, args = (batch,))
#     jobs.append(p)
    
# [p.start() for p in jobs]
# [p.join() for p in jobs]

# end = time.time()
# print('Time taken:', str(end - start))
# print('Process complete')


if __name__ == '__main__':
    start = time.time()
    array_split = np.array_split(file_list, mp.cpu_count())
    pool = Pool(mp.cpu_count())
    pool.map(splitting, array_split)
    end = time.time()
    print('Time taken:', str(end - start))
    print('Process complete')

