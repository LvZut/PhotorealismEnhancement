import csv

from os import listdir
from os.path import isfile, join

# dataset = 'hddCARLA'
dataset = 'CARLA'
town = '_Town01'

if dataset == 'CARLA':
    data_folder = '/hdd2/LvZut/saivvy/data/carla/'
    rgb_folder = 'rgb'
elif dataset == 'cityscapes':
    data_folder = 'epe/dataset/cityscapes/'
    rgb_folder = 'data/'
elif dataset== 'hddCARLA':
    data_folder = '../../data/carla/'
    rgb_folder = 'rgb'
else:
    exit('dataset does not exist')

# onlyfiles = [f for f in listdir(data_folder+rgb_folder) if isfile(join(data_folder+rgb_folder, f))]
onlyfiles = ['rgb_Town01_3678_8_90_degrees.png']
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

    elif dataset == 'hddCARLA':
        for file in onlyfiles:
            number = file[10:15].lstrip('0')
            writer.writerow([f'{data_folder}rgb/{file}', f'{data_folder}semantic/{file}', f'{data_folder}robust_semantic/gray/{file}', f'{data_folder}gbuffers/gbuffer_{number}.npz'])
