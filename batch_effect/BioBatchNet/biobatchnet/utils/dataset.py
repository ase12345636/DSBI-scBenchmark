import torch
from torch.utils.data import Dataset
import numpy as np


class BBNDataset(Dataset):
    def __init__(self, data, batch_labels, cell_types=None, batch_names=None, cell_type_names=None):
        self.data = torch.tensor(np.array(data), dtype=torch.float32)
        self.batch_labels = torch.tensor(np.array(batch_labels), dtype=torch.long)
        self.cell_types = torch.tensor(np.array(cell_types), dtype=torch.long) if cell_types is not None else None
        # Original string labels for visualization
        self.batch_names = np.array(batch_names) if batch_names is not None else None
        self.cell_type_names = np.array(cell_type_names) if cell_type_names is not None else None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.cell_types is not None:
            return self.data[idx], self.batch_labels[idx], self.cell_types[idx]
        return self.data[idx], self.batch_labels[idx]
