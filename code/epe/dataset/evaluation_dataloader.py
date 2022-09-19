import torch
from torch.utils.data import Dataset, DataLoader


class evaluation_dataloader_fake(Dataset):
    """ Wrapper around dataloader that performs inference
        using generator model given in class init """

    def __init__(self, dataset, gen, device, transform=None):
        """
        Args:
            dataset (Torch dataset) : Feeds images to generator
            gen (PassthruGenerator) : Generator that performs inference on images
        """
        self.gen = gen
        self.transform = transform
        self.dataset = dataset
        self.device = device

    def __len__(self):
        #return len(self.dataloader)
        return int(len(self.dataset) / 4)

    def __getitem__(self, idx):
        if idx < self.__len__():
            # get item and run inference with model before returning
            result = False
            for i in range(4):
                batch = self.dataset[4*idx+i]

                # same clamping is used for inference during testing
                model_out = self.gen(batch.to(self.device)).clamp(min=0,max=1)
                del batch
                if not isinstance(results, torch.Tensor):
                    results = batch
                else:
                    torch.cat(results, batch)
            # breakpoint()
            return {'images' : model_out}
        else:
            raise IndexError

class evaluation_dataloader_real(Dataset):
    """ Wrapper around dataloader that performs inference
        using generator model given in class init """

    def __init__(self, dataset, device, transform=None):
        """
        Args:
            dataset (Torch dataset) : Feeds images to generator
            gen (PassthruGenerator) : Generator that performs inference on images
        """
        self.transform = transform
        self.dataset = dataset
        self.device = device
        self.count = 0

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if idx < self.__len__():
            # self.count += 1
            # print('batch ', self.count, idx )
            batch = self.dataset[idx]#.to(self.device)
            return {'images' : batch.img}
        else:
            raise IndexError