import numpy as np
import imageio
from PIL import Image

import os
from os import listdir
from os.path import isfile, join

# https://stackoverflow.com/questions/39382412/crop-center-portion-of-a-numpy-image
def crop_center(img,cropx,cropy):
    y,x = img.shape[:2]
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx,:]


def main():
    path = "Town01/"
    gpath = "../../../../data/gbuffers/" + path

    # rgb_files = [f for f in listdir(gpath) if ((isfile(join(gpath, f))) and (len(str(f)) == 19))]
    # gbuffers = [f for f in listdir(gpath) if isfile(join(gpath, f))]
    # print(len(rgb_files), len(gbuffers))
    # gbuffer_suffixes = ['WorldTangent', 'SeparateTranslucencyRGB', 'SceneDepthWorldUnits',
    #                     'BaseColor', 'Opacity', 'SubsurfaceColor', 'Roughness', 'LightingModel',
    #                     'SceneColor', 'WorldNormal', 'SceneDepth', 'SeparateTranslucencyA',
    #                     'PostTonemapHDRColor', 'Specular', 'PreTonemapHDRColor', 'Metallic']


    episode_folders = [x[0] for x in os.walk(gpath)]
    
    for i in episode_folders:
        episode_steps = [y[0] for y in os.walk(i)]
        for j in episode_steps:
            breakpoint()
            print(j)

        # ScreenShot15889.png
        # prefix = f'ScreenShot{str(i).zfill(5)}'

        # # Some screenshots might have missing gbuffers
        # if len([file for file in gbuffers if prefix in file]) > 16:
            
            
        #     # rgb
        #     im = imageio.imread(gpath + prefix + '.png', pilmode='RGB')
        #     im = crop_center(im, 960, 540)

        #     im = Image.fromarray(im)
        #     im.save('rgb/' + prefix + '.png')

        #     # gbuffers
        #     #gbuffer_array = np.zeros((540, 960, 48))

        #     # load all gbuffer images as np arrays
        #     #for j, gbuffer in enumerate(gbuffer_suffixes):
        #     #    file_name = gpath + prefix + '_' + gbuffer + '.png'
        #     #    im = imageio.imread(file_name)
        #     #    im = crop_center(im, 960, 540)
        #     #    gbuffer_array[:,:,j:j+3] = im[:,:,:-1]

            
        #     #np.savez_compressed('gbuffers/gbuffer_' + str(i), data=gbuffer_array)
            
        # else:
        #     print(f'Skipped image {i}, only {len([file for file in gbuffers if prefix in file])}/16 files found!')
        

        # if i % 10 == 0:
        #     print(f'Processed {i} images ({i / round(len(rgb_files))}%)')


if __name__ == "__main__":
    main()

