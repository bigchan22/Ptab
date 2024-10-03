import numpy as np
import torch
from torch_geometric.data import Data
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, input_data):
        self.features = input_data.features
        self.labels = input_data.labels
        self.rows = input_data.rows
        self.cols = input_data.columns
        self.edge_types = input_data.edge_types
        self.graph_sizes = input_data.graph_sizes
    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        edge_index = torch.tensor([self.rows[idx], self.cols[idx]], dtype=torch.long)
        return Data(x=torch.from_numpy(self.features[idx]).float(), edge_index=edge_index, 
             edge_types = torch.tensor(self.edge_types[idx][:, np.newaxis], dtype=torch.float),
             y=torch.from_numpy(np.array(self.labels[idx])),graph_sizes = torch.from_numpy(np.array(self.graph_sizes[idx])))