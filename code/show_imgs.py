import torch
import matplotlib.pyplot as plt
import numpy as np

from epe.mseg_inference import mseg_task
from mseg_semantic.utils import config

robust_cfg = config.load_cfg_from_cfg_file('config/robust_config/config_480.yaml')
mseg_inference = mseg_task(robust_cfg)

# import imageio
# img = imageio.imread('../../saivvy/data/carla/rgb/rgb_Town01_1111_0_180_degrees.png')
# inp = torch.load(f'gen_out/input_1001.pt', map_location=torch.device('cpu')).detach().numpy()
# inp_robust = torch.from_numpy(mseg_inference.inference(inp.copy())[0])
# plt.imshow(inp_robust)
# plt.savefig('robust_out.png')
steps = list(range(1001, 1099, 2))


for step in steps:
    rec_robust = torch.load(f'gen_out/rec_{step}.pt', map_location=torch.device('cpu'))
    inp_labels = torch.load(f'gen_out/robust_{step}.pt', map_location=torch.device('cpu'))
    img = torch.load(f'gen_out/input_{step}.pt', map_location=torch.device('cpu')).detach().numpy()
    rec = torch.load(f'gen_out/output_{step}.pt', map_location=torch.device('cpu')).detach().numpy()

    inp_robust = torch.from_numpy(mseg_inference.inference(img.copy())[0])

    # input
    plt.subplot(2, 3, 1)
    plt.imshow(  np.transpose(img[0], (1,2,0))  )
    plt.suptitle('input')
    # output
    plt.subplot(2, 3, 2)
    plt.imshow(  np.transpose(rec[0], (1,2,0))  )
    plt.suptitle('output')
    # rec
    plt.subplot(2, 3, 3)
    plt.imshow(  rec_robust  )
    plt.suptitle('output_robust')
    # inp_robust
    plt.subplot(2, 3, 4)
    plt.imshow(  inp_robust  )
    plt.savefig(f'gen_out/results_{step}.png')
    plt.suptitle('input robust')
    # outp_robust
    plt.subplot(2, 3, 5)
    plt.imshow(  inp_labels[0,0,:,:]  )
    plt.suptitle('input labels')
    
    print(step, np.max(inp_labels[0,0,:,:]))