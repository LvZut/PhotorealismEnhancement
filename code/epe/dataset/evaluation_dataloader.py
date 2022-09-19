import torch
from torch.utils.data import Dataset, DataLoader


class evaluation_dataloader(Dataset):
    """ Wrapper around dataloader that performs inference
        using generator model given in class init """

    def __init__(self, dataset, gen, transform=None):
        """
        Args:
            dataloader (Torch dataloader) : Feeds images to generator
            gen (PassthruGenerator) : Generator that performs inference on images
        """
        self.gen = gen
        self.transform = transform
        self.dataset = dataset

    def __len__(self):
        #return len(self.dataloader)
        return len(self.dataset)

    def __getitem__(self, idx):
        # get item and run inference with model before returning
        batch = self.dataset[idx]
        model_out = self.gen(batch.fake)
        del batch
        return model_out