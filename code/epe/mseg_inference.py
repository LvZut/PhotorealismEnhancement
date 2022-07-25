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

class mseg_task():
    def __init__(self, cfg):
        self.robust_cfg = cfg
        

    def inference(self, image):
        # image = cv2.imread('../../saivvy/data/carla/rgb/rgb_Town01_1000_3_90_degrees.png', cv2.IMREAD_COLOR)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = np.float32(image)
        
        # with torch.no_grad():
        # print('image_shape:', image.shape)
        image = np.transpose(image[0], (1,2,0))
        # need first image to do some initializing
        if not hasattr(self, 'self.robust_cfg.base_size'):
            self.robust_cfg.base_size = determine_max_possible_base_size(h=image.shape[0], w=image.shape[1], crop_sz=min(self.robust_cfg.test_h, self.robust_cfg.test_w))
            # print(f'created base size: {self.robust_cfg.base_size}')

            # same transform as mseg dataloader
            mean, std = normalization_utils.get_imagenet_mean_std()
            crop_transform = transform.Compose([transform.ResizeShort(self.robust_cfg.base_size), transform.ToTensor(), transform.Normalize(mean=mean, std=std)])

            self.robust_cfg.native_img_h=image.shape[0]
            self.robust_cfg.native_img_w=image.shape[1]

            self.task = BatchedInferenceTask(self.robust_cfg, self.robust_cfg.base_size, self.robust_cfg.test_h, self.robust_cfg.test_w, '', 'universal', 'universal', self.robust_cfg.scales)

        # apply transform to image
        image, _ = crop_transform(image, image[:, :, 0])
        image = torch.from_numpy(np.expand_dims(image, axis=0))

        

        self.task.model.eval()
        out = self.task.execute_on_batch(batch=image)

        return out
