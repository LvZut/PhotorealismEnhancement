import torch
import matplotlib.pyplot as plt
import numpy as np


steps = list(range(1001, 1099, 2))


for step in steps:
    rec = torch.load(f'gen_out/rec_{step}.pt', map_location=torch.device('cpu'))
    robust = torch.load(f'gen_out/robust_{step}.pt', map_location=torch.device('cpu'))
    inp = torch.load(f'gen_out/input_{step}.pt', map_location=torch.device('cpu')).detach().numpy()
    outp = torch.load(f'gen_out/output_{step}.pt', map_location=torch.device('cpu')).detach().numpy()

    # print(rec.shape, robust.shape)
    plt.subplot(2, 2, 1)
    plt.imshow(  rec  )

    plt.subplot(2, 2, 2)
    plt.imshow(  robust[0,0,:,:]  )

    plt.subplot(2, 2, 3)
    plt.imshow(  np.transpose(inp[0], (1,2,0))  )

    plt.subplot(2, 2, 4)
    plt.imshow(  np.transpose(outp[0], (1,2,0))  )
    plt.show()

    print(torch.unique(rec), torch.unique(robust))