import torch
import matplotlib.pyplot as plt
import numpy as np

from epe.mseg_inference import mseg_task
from mseg_semantic.utils import config

robust_cfg = config.load_cfg_from_cfg_file('config/robust_config/config_1080.yaml')
mseg_inference = mseg_task(robust_cfg)

rgb_path = ' ../../saivvy/data/carla/rgb/rgb_Town01_1111_0_180_degrees.png'
img = np.load(rgb_path)
robust = mseg_inference(img)
breakpoint()

# steps = list(range(1001, 1099, 2))


# for step in steps:
#     rec = torch.load(f'gen_out/rec_{step}.pt', map_location=torch.device('cpu'))
#     robust = torch.load(f'gen_out/robust_{step}.pt', map_location=torch.device('cpu'))
#     inp = torch.load(f'gen_out/input_{step}.pt', map_location=torch.device('cpu')).detach().numpy()
#     outp = torch.load(f'gen_out/output_{step}.pt', map_location=torch.device('cpu')).detach().numpy()

#     inp_robust = torch.from_numpy(mseg_inference.inference(inp.copy())[0])

#     # print(rec.shape, robust.shape)
#     plt.subplot(2, 3, 1)
#     plt.imshow(  rec  )

#     plt.subplot(2, 3, 2)
#     plt.imshow(  robust[0,0,:,:]  )

#     plt.subplot(2, 3, 3)
#     plt.imshow(  np.transpose(inp[0], (1,2,0))  )

#     plt.subplot(2, 3, 4)
#     plt.imshow(  np.transpose(outp[0], (1,2,0))  )

#     plt.subplot(2, 3, 5)
#     plt.imshow(  inp_robust  )
#     plt.savefig(f'gen_out/results_{step}.png')
 

#     print(torch.unique(rec), torch.unique(robust))