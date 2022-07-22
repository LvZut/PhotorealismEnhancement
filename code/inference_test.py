from mseg_semantic.tool.inference_task import InferenceTask
from mseg_semantic.utils import config

import numpy as np
import cv2

robust_cfg = config.load_cfg_from_cfg_file('config/robust_config/config_1080.yaml')

assert isinstance(robust_cfg.model_name, str)
assert isinstance(robust_cfg.model_path, str)

task = InferenceTask(robust_cfg, 0, 0, 0, '', 'universal', 'universal', robust_cfg.scales)

img1 = cv2.imread('../../saivvy/data/carla/rgb/rgb_Town01_1000_3_90_degrees.png', cv2.IMREAD_COLOR)
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img1 = np.float32(img1)

task.base_size=min(img1.shape[0], img1.shape[1])
task.crop_h=img1.shape[0]
task.crop_w=img1.shape[1]

img1 = np.transpose(0,1,2)

out = task.execute_on_img(image=img1)