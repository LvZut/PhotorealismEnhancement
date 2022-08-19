from readline import get_line_buffer
import numpy as np
import imageio.v2 as imageio
from PIL import Image
from tqdm import tqdm
import csv

import os
from os import listdir
from os.path import isfile, join


from multiprocessing import Pool
import logging

import re

start_episode = 0

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def save_data(arguments):

    step_path = arguments['step_path']
    suffix = arguments['suffix']
    gbuffer_prefixes = arguments['gbuffer_prefixes']
    gb_path, rgb_path, gt_path = arguments['save_paths']
    oneD = arguments['oneD']

    # resize image?
    try:
        img = imageio.imread(f'{step_path}rgb_{suffix}.jpeg', pilmode='RGB')
        gt_labels = imageio.imread(f'{step_path}sem_seg_{suffix}.png', pilmode='RGB')
    except:
        logging.info(f'img/gt_labels not found for {step_path}')
        return 0

    gb_array = np.zeros((720, 1280, 31))
    arr_index = 0

    for prefix in gbuffer_prefixes:

        try:
            gb = imageio.imread(f'{step_path}{prefix}_{suffix}.png', pilmode='RGB')
        except:
            print(f'gbuffer {step_path}{prefix}_{suffix} not found for image!')
            logging.info(f'gbuffer {step_path}{prefix}_{suffix} not found, skipping step')
            return 0


        # only use first dimension if grayscale
        if prefix in oneD:
            gb_array[:,:,arr_index] = gb[:,:,0]
            arr_index += 1
        else:
            gb_array[:,:,arr_index:arr_index+3] = gb
            arr_index += 3
    logging.info(f'Saving {step_path}')
    np.savez_compressed(gb_path, img=img, gbuffers=gb_array, shader=gt_labels)
    imageio.imsave(rgb_path, img)
    imageio.imsave(gt_path, gt_labels)

def main():
    path = "Town02"
    gpath = "../../../../data/gbuffers/" + path + '/'
    carla_path = "../../../../data/carla/"
    save_path = "/hdd2/LvZut/saivvy/data/carla/"

    # removed the first 2 temporarily as they are white images
    gbuffer_prefixes = [#'ambientocclusion', 'separatetranslucency',
                        'anisotropy', 'basecolor', 'depth', 'diffuse', 'materialao', 'metallic', 
                        'normal', 'opacity', 'roughness', 'sem_seg',  'shadingmodelcolor', 
                        'shadingmodelid', 'specular', 'subsurfacecolor', 'worldtangent']

    oneD = ['anisotropy', 'materialao', 'metallic', 'opacity', 
            'roughness', 'sem_seg', 'specular']


    suffixes = ['0_degrees', '90_degrees', '180_degrees', '270_degrees']

    episode_folders = sorted([ f.path for f in os.scandir(gpath) if f.is_dir() ], key=natural_keys)
    print(f'episodes: {len(episode_folders)}')
   
    logging.basicConfig(filename="/hdd2/LvZut/saivvy/data/carla/process_log.log", level=logging.INFO)
    episode_num = 0

    with open(f'../../CARLA_files.csv', 'a') as f:
        # create the csv writer
        writer = csv.writer(f)
        pool = Pool(8)

        if start_episode != 0:
            episode_folders = episode_folders[start_episode:]

        for index_ep, episode in tqdm(enumerate(episode_folders, start=start_episode)):
            episode_steps = [ f.path for f in os.scandir(episode) if f.is_dir() ]
            
            arguments_list = []
            for index_step, step in enumerate(episode_steps):

                for suffix in suffixes:
                    step_path = f'{step}/'

                    # make sure data/carla/gbuffers exists
                    gb_path = f'{save_path}gbuffers/data_{path}_{index_ep}_{index_step}_{suffix}.npz'
                    rgb_path = f'{save_path}rgb/rgb_{path}_{index_ep}_{index_step}_{suffix}.png'
                    gt_path = f'{save_path}semantic/semantic_{path}_{index_ep}_{index_step}_{suffix}.png'

                    arguments = {'step_path' : step_path,
                                'suffix' : suffix,
                                'gbuffer_prefixes' : gbuffer_prefixes,
                                'save_paths' : (gb_path, rgb_path, gt_path),
                                'oneD' : oneD}

                    arguments_list.append(arguments)

                    # don't write async as that might cause trouble
                    writer.writerow([rgb_path, gt_path, f'{save_path}robust_semantic/gray/robust_{path}_{index_ep}_{index_step}_{suffix}.png', gb_path])
            
            pool.map(save_data, arguments_list)
            episode_num += 1

            if episode_num % 100 == 0:
                try:
                    logging.info(f'processed {episode_num/len(episode_folders)} ({episode_num}/{len(episode_folders)})')
                except:
                    pass
    pool.close()
    pool.join()

if __name__ == "__main__":
    main()

