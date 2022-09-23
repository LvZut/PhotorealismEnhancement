import torch
from torch.utils.data import Dataset, DataLoader


class evaluation_dataloader_fake(Dataset):
    """ Wrapper around dataloader that performs inference
        using generator model given in class init """

    def __init__(self, dataset, gen, device, batch_size, logger, transform=None):
        """
        Args:
            dataset (Torch dataset) : Feeds images to generator
            gen (PassthruGenerator) : Generator that performs inference on images
        """
        self.gen = gen
        self.transform = transform
        self.dataset = dataset
        self.device = device
        self.batch_size = batch_size
        self.logger = logger
        self.idx_counter = 0

    def __len__(self):
        # return len(self.dataset)
        return 25000

    def __getitem__(self, idx):
        if idx < self.__len__():
            # get item and run inference with model before returning
            if idx > self.idx_counter:
                self.idx_counter = idx
                self.logger.info(idx*self.batch_size)
            batch = self.dataset[[i for i in range(idx*self.batch_size, (idx*self.batch_size)+self.batch_size)]]

            # same clamping is used for inference during testing
            model_out = self.gen(batch.to(self.device)).clamp(min=0,max=1)
            del batch
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