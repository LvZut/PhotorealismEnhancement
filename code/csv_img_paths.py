import csv

from os import listdir
from os.path import isfile, join

# dataset = 'hddCARLA'
dataset = 'cityscapes'

if dataset == 'CARLA':
    data_folder = 'epe/dataset/CARLA/Town02/20220504115602.544453/'
    rgb_folder = 'gbuffers'
elif dataset == 'cityscapes':
    data_folder = 'epe/dataset/cityscapes/leftImg8bit/'
    rgb_folder = 'train'
elif dataset== 'hddCARLA':
    data_folder = '../../data/carla/'
    rgb_folder = 'rgb'
else:
    exit('dataset does not exist')

onlyfiles = [f for f in listdir(data_folder+rgb_folder) if isfile(join(data_folder+rgb_folder, f))]
print(f'found {len(onlyfiles)} rgb files!')


# open the file in the write mode
with open(f'{dataset}_files.csv', 'w') as f:
    # create the csv writer
    writer = csv.writer(f)
    rgb_folder += '/'
    if dataset == 'CARLA':
        # write a row to the csv file
        for rgb_file_full in onlyfiles:
            rgb_file = rgb_file_full[3:-4]
            writer.writerow([f'{data_folder}rgb/rgb{rgb_file}.png', f'{data_folder}semantic/sem{rgb_file}.png', f'{data_folder}robust_semantic/gray/robust{rgb_file}.png', f'{data_folder}gbuffers/{rgb_file_full}'])

    elif dataset == 'cityscapes':
        for rgb_file in onlyfiles:
            writer.writerow([f'{data_folder+rgb_folder}{rgb_file}', f'{data_folder}robust_semantic/gray/{rgb_file}'])

    elif dataset == 'hddCARLA':
        for file in onlyfiles:
            number = file[10:15].lstrip('0')
            writer.writerow([f'{data_folder}rgb/{file}', f'{data_folder}semantic/{file}', f'{data_folder}robust_semantic/gray/{file}', f'{data_folder}gbuffers/gbuffer_{number}.npz'])
