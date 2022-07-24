from mseg_semantic.tool.inference_task import InferenceTask
from mseg_semantic.utils import config

import numpy as np
import cv2

# adjusted function that doesnt read/write images but uses mem instead
def render_single_img_pred(self, image: np.ndarray, min_resolution: int = 1080) -> None:
    """Since overlaid class text is difficult to read below 1080p, we upsample predictions."""

    rgb_img = image
    pred_label_img = self.execute_on_img(rgb_img)

    # avoid blurry images by upsampling RGB before overlaying text
    if np.amin(rgb_img.shape[:2]) < min_resolution:
        rgb_img = resize_util.resize_img_by_short_side(rgb_img, min_resolution, "rgb")
        pred_label_img = resize_util.resize_img_by_short_side(pred_label_img, min_resolution, "label")

    metadata = None
    frame_visualizer = Visualizer(rgb_img, metadata)
    overlaid_img = frame_visualizer.overlay_instances(
        label_map=pred_label_img, id_to_class_name_map=self.id_to_class_name_map
    )
    imageio.imwrite(output_demo_fpath, overlaid_img)
    imageio.imwrite(output_gray_fpath, pred_label_img)


robust_cfg = config.load_cfg_from_cfg_file('config/robust_config/config_1080.yaml')

assert isinstance(robust_cfg.model_name, str)
assert isinstance(robust_cfg.model_path, str)

task = InferenceTask(robust_cfg, 0, 0, 0, '', 'universal', 'universal', robust_cfg.scales)
task.render_single_img_pred = render_single_img_pred()

img1 = cv2.imread('../../saivvy/data/carla/rgb/rgb_Town01_1000_3_90_degrees.png', cv2.IMREAD_COLOR)
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img1 = np.float32(img1)

task.base_size=1080
task.crop_h=img1.shape[0]
task.crop_w=img1.shape[1]


out = task.render_single_img_pred(image=img1)