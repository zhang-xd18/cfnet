import os
import scipy.io as sio

import torch
from torch.utils.data import DataLoader, TensorDataset

__all__ = ['Cost2100DataLoader']

class Cost2100DataLoader(object):
    r""" PyTorch DataLoader for COST2100 dataset.
    """

    def __init__(self, root, batch_size, num_workers, scenario, device):
        print(root)
        assert os.path.isdir(root)
        assert scenario in {"A", "C", "D"}
        self.batch_size = batch_size
        self.batch_tiny_size = batch_size // 5
        self.num_workers = num_workers
        self.pin_memory = False #pin_memory

        dir_test = os.path.join(root, f"DATA_Htest{scenario}.mat")
        channel, nt, nc, nc_expand = 2, 32, 32, 125

        # Test data loading, including the sparse data and the raw data
        data_test = sio.loadmat(dir_test)['HT']
        data_test = torch.tensor(data_test, dtype=torch.float32).view(
            data_test.shape[0], channel, nt, nc).to(device)

        self.test_dataset = TensorDataset(data_test)

    def __call__(self):
        test_loader = DataLoader(self.test_dataset,
                                 batch_size=self.batch_size,
                                 num_workers=self.num_workers,
                                 pin_memory=self.pin_memory,
                                 shuffle=False)
        

        return test_loader
