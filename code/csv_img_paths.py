import csv

import os
from os import listdir
from os.path import isfile, join

# dataset = 'hddCARLA'
# dataset = 'CARLA'
dataset = 'nuscenes'

if dataset == 'CARLA':
    data_folder = '../../saivvy/data/carla/'
    rgb_folder = 'rgb'
    town = '_Town01'
elif dataset == 'cityscapes':
    data_folder = 'epe/dataset/cityscapes/'
    rgb_folder = 'data/'
elif dataset== 'nuscenes':
    data_folder = '../../saivvy/data/nuscenes/'
    rgb_folder = False
else:
    exit('dataset does not exist')

if rgb_folder:
    onlyfiles = [f for f in listdir(data_folder+rgb_folder) if isfile(join(data_folder+rgb_folder, f))]

print(f'found {len(onlyfiles)} rgb files!')


# open the file in the write mode
with open(f'{dataset}_files.csv', 'a') as f:
    # create the csv writer
    writer = csv.writer(f)
    rgb_folder += '/'
    if dataset == 'CARLA':
        suffix = 'degrees.png'
        for rgb_file_full in onlyfiles:
            rgb_file = town + rgb_file_full[10:-11]
            writer.writerow([f'{data_folder}rgb/rgb{rgb_file}{suffix}', f'{data_folder}semantic/semantic{rgb_file}{suffix}', f'{data_folder}robust_semantic/gray/robust{rgb_file}{suffix}', f'{data_folder}gbuffers/data{rgb_file}_degrees.npz'])

    elif dataset == 'cityscapes':
        for rgb_file in onlyfiles:
            writer.writerow([f'{data_folder+rgb_folder}{rgb_file}', f'{data_folder}robust_semantic/gray/{rgb_file}'])

    elif dataset == 'nuscenes':
        i = 0
        for root, dirs, files in os.walk(data_folder):
            for file in files:
                if file.endswith('.jpg'):
                    print(file)
                    if i > 10:
                        break