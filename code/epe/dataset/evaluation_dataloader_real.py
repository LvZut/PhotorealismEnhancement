import torch
from torch.utils.data import Dataset, DataLoader


class evaluation_dataloader_real(Dataset):
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
        return len(self.dataset)

    def __getitem__(self, idx):
        # get item and run inference with model before returning
        if idx % 100 == 0:
            print('real: ', idx)
        batch = self.dataset[idx]
        return {'images' : batch.imgs}