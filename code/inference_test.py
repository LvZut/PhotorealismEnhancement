from mseg_semantic.tool.batched_inference_task import BatchedInferenceTask
from mseg_semantic.utils import config

import numpy as np
import torch
import cv2
import math

# for dataloader
from mseg_semantic.utils import transform
import mseg_semantic.utils.normalization_utils as normalization_utils


def determine_max_possible_base_size(h: int, w: int, crop_sz: int) -> int:
    """Given a crop size and original image dims for aspect ratio, determine
    the max base_size that will fit within the crop.
    """
    longer_size = max(h, w)
    if longer_size == h:
        scale = crop_sz / float(h)
        base_size = math.floor(w * scale)
    else:
        scale = crop_sz / float(w)
        base_size = math.floor(h * scale)

    return base_size


robust_cfg = config.load_cfg_from_cfg_file('config/robust_config/config_1080.yaml')

assert isinstance(robust_cfg.model_name, str)
assert isinstance(robust_cfg.model_path, str)


img1 = cv2.imread('../../saivvy/data/carla/rgb/rgb_Town01_1000_3_90_degrees.png', cv2.IMREAD_COLOR)
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img1 = np.float32(img1)

print('image_shape:', img1.shape)


robust_cfg.base_size = determine_max_possible_base_size(h=img1.shape[0], w=img1.shape[1], crop_sz=min(robust_cfg.test_h, robust_cfg.test_w))

# same transform as mseg dataloader
mean, std = normalization_utils.get_imagenet_mean_std()
crop_transform = transform.Compose([transform.ResizeShort(robust_cfg.base_size), transform.ToTensor(), transform.Normalize(mean=mean, std=std)])

robust_cfg.native_img_h=img1.shape[0]
robust_cfg.native_img_w=img1.shape[1]

img1, _ = crop_transform(img1, img1[:, :, 0])

img1 = torch.from_numpy(np.expand_dims(img1, axis=0))


task = BatchedInferenceTask(robust_cfg, robust_cfg.base_size, robust_cfg.test_h, robust_cfg.test_w, '', 'universal', 'universal', robust_cfg.scales)


# task.base_size=1280
# task.crop_h=img1.shape[0]
# task.crop_w=img1.shape[1]



# labels=np.ones_like(img1)

task.model.eval()
out = task.execute_on_batch(batch=img1)