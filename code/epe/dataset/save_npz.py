import numpy as np
import imageio
from PIL import Image
from tqdm import tqdm
import csv

import os
from os import listdir
from os.path import isfile, join

# # https://stackoverflow.com/questions/39382412/crop-center-portion-of-a-numpy-image
# def crop_center(img,cropx,cropy):
#     y,x = img.shape[:2]
#     startx = x//2-(cropx//2)
#     starty = y//2-(cropy//2)    
#     return img[starty:starty+cropy,startx:startx+cropx,:]


def main():
    path = "Town01"
    gpath = "../../../../data/gbuffers/" + path + '/'
    carla_path = "../../../../data/carla/"

    # removed the first 2 temporarily as they are white images
    gbuffer_prefixes = [#'ambientocclusion', 'separatetranslucency',
                        'anisotropy', 'basecolor', 'depth', 'diffuse', 'materialao', 'metallic', 
                        'normal', 'opacity', 'roughness', 'sem_seg',  'shadingmodelcolor', 
                        'shadingmodelid', 'specular', 'subsurfacecolor', 'worldtangent']

    oneD = ['anisotropy', 'materialao', 'metallic', 'opacity', 
            'roughness', 'sem_seg', 'specular']


    suffixes = ['0_degrees', '90_degrees', '180_degrees', '270_degrees']

    episode_folders = [ f.path for f in os.scandir(gpath) if f.is_dir() ]
    print(f'episodes: {len(episode_folders)}')

# open the file in the write mode
    with open(f'{gpath}{path}_files.csv', 'w') as f:
        # create the csv writer
        writer = csv.writer(f)

        for index_ep, episode in tqdm(enumerate(episode_folders)):
            episode_steps = [ f.path for f in os.scandir(episode) if f.is_dir() ]
            # breakpoint()
            for index_step, step in enumerate(episode_steps):
                # breakpoint()
                # print(step)

                for suffix in suffixes:
                    step_path = f'{step}/'

                    # resize image?
                    img = imageio.imread(f'{step_path}rgb_{suffix}.jpeg', pilmode='RGB')
                    gt_labels = imageio.imread(f'{step_path}sem_seg_{suffix}.png', pilmode='RGB')

                    gb_array = np.zeros((720, 1280, 31))
                    arr_index = 0

                    for prefix in gbuffer_prefixes:

                        try:
                            gb = imageio.imread(f'{step_path}{prefix}_{suffix}.png', pilmode='RGB')
                        except:
                            print(f'gbuffer {step_path}{prefix}_{suffix} not found for image!')
                            continue


                        # only use first dimension if grayscale
                        if prefix in oneD:
                            gb_array[:,:,arr_index] = gb[:,:,0]
                            arr_index += 1
                        else:
                            gb_array[:,:,arr_index:arr_index+3] = gb
                            arr_index += 3

                    # make sure data/carla/gbuffers exists
                    gb_path = f'{carla_path}gbuffers/data_{index_ep}_{index_step}_{suffix}.npz'
                    rgb_path = f'{carla_path}rgb/rgb_{index_ep}_{index_step}_{suffix}.png'
                    gt_path = f'{carla_path}semantic/semantic_{index_ep}_{index_step}_{suffix}.png'



                    np.savez_compressed(gb_path, img=img, gbuffers=gb_array, shader=gt_labels)
                    imageio.imsave(rgb_path, img)
                    imageio.imsave(gt_path, gt_labels)

                    writer.writerow([rgb_path[6:], gt_path[6:], f'{carla_path[6:]}robust_semantic/gray/robust_{index_ep}_{index_step}_{suffix}.png', gb_path[6:]])



if __name__ == "__main__":
    main()

