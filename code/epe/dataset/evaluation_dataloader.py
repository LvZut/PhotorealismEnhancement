import torch
from torch.utils.data import Dataset, DataLoader


class evaluation_dataloader(Dataset):
    """ Wrapper around dataloader that performs inference
        using generator model given in class init """

    def __init__(self, dataset, gen, collate_fn_val, seed_worker):
        """
        Args:
            dataloader (Torch dataloader) : Feeds images to generator
            gen (PassthruGenerator) : Generator that performs inference on images
        """
        self.gen = gen
        self.dataloader = torch.utils.data.DataLoader(self.dataset_fake_val, batch_size=1, shuffle=True, num_workers=0, pin_memory=True, drop_last=False, collate_fn=collate_fn_val, worker_init_fn=seed_worker)
        

    def __len__(self):
        return len(self.dataloader)

    def __getitem__(self, idx):
        batch = self.dataloader.__getitem__(idx)
        model_out = self.gen(batch.fake)
        del batch
        return model_out