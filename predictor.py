import functools
import enum
import os

from BH.data_loader import *
from BH.generate_data import *
from predictor_info import *
# from Model_e import Model_e,Direction,Reduction
from Train import train,print_accuracies
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv,GCNConv

import pickle

class CustomDataset(Dataset):
    def __init__(self, input_data):
        self.features = input_data.features
        self.labels = input_data.labels
        self.rows = input_data.rows
        self.cols = input_data.columns
        self.edge_types = input_data.edge_types

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        edge_index = torch.tensor([self.rows[idx], self.cols[idx]], dtype=torch.long)
        return Data(x=torch.from_numpy(self.features[idx]).float(), edge_index=edge_index, 
             edge_types = torch.tensor(self.edge_types[idx][:, np.newaxis], dtype=torch.float),
             y=torch.from_numpy(np.array(self.labels[idx])))


node_dim = num_features
edge_dim = 8
graph_deg = graph_deg
depth = num_layers


class GCN_single(torch.nn.Module):
    def __init__(self,num_edge_types,depth):
        super().__init__()
        self.num_edge_types= num_edge_types
        self.depth=depth
        self.node_linear = torch.nn.Linear(1,node_dim)
        self.edge_linear = torch.nn.Linear(1,edge_dim)
        self.conv1 = GCNConv(node_dim, node_dim)
        self.conv2 = GCNConv(node_dim, node_dim)
        self.conv3 = GCNConv(node_dim, node_dim)
        self.conv4 = GCNConv(node_dim, node_dim)
        
        self.out1 = torch.nn.Linear(node_dim,node_dim)
        self.out2 = torch.nn.Linear(node_dim,1)
        self.initialize_parameters()

    def initialize_parameters(self):
        for module in self.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
        

    def forward(self, data):
        x, edge_index, edge_types = data.x, data.edge_index, data.edge_types
        x = self.node_linear(x)        
        for i in range(self.depth):
            mask1 = (edge_types == 1)
            edge_index1 = edge_index[:, mask1.squeeze()]
            x1 = self.conv1(x, edge_index1)
            x1 = F.relu(x1)
            
            mask2 = (edge_types == 2)
            edge_index2 = edge_index[:, mask2.squeeze()]
            x2 = self.conv2(x, edge_index2)
            x2 = F.relu(x2)
            
            mask3 = (edge_types == 3)
            edge_index3 = edge_index[:, mask3.squeeze()]
            x3 = self.conv3(x, edge_index3)
            x3 = F.relu(x3)
            
            mask4 = (edge_types == 4)
            edge_index4 = edge_index[:, mask4.squeeze()]
            x4 = self.conv4(x, edge_index4)
            x4 = F.relu(x4)
            
            x = x1+x2+x3+x4
            x = F.relu(x)
        xx = torch.reshape(x,(-1,graph_deg,node_dim))
        xxx,_ = torch.max(xx,dim=1)
        xxx = self.out1(xxx)
        xxx = self.out2(xxx)
        
        return xxx
#         return F.log_softmax(xxx, dim=1)
    
class GCN_multi(torch.nn.Module):
    def __init__(self, graph_deg, depth):
        super().__init__()
        self.num_edge_types = 4
        self.depth = depth
        self.graph_deg = graph_deg
        self.GCN_single = GCN_single(self.num_edge_types, self.depth)        

    def forward(self, data,T=1):
        batch = data.batch
        batch = batch[::self.graph_deg]
        x = self.GCN_single(data)
        x=torch.sigmoid(x/T)
        unique_batches = torch.unique(batch)
        
#         sums_tensor = torch.zeros(len(unique_batches), requires_grad=True)

#         # Loop over unique batches and add elements of 'x' within each batch directly to sums_tensor
#         for i, ub in enumerate(unique_batches):
#             sums_tensor[i] = x[batch == ub].sum()

            
            # Loop over unique batches and sum elements of 'x' within each batch
        sum_list = [x[batch == ub].sum() for ub in unique_batches]

        # Stack list of sums to create a tensor
        sums_tensor = torch.stack(sum_list)
#         print(sums_tensor)
        
#         unique_batches = torch.unique(batch)
#         sums = []
#         for ub in unique_batches:
#             sums.append(x[batch == ub].sum())

#         # Convert list of sums to tensor
#         sums_tensor = torch.tensor(sums)
        
        return sums_tensor

def predict_tableau(P, word, show=True):
    shape = shape_of_word(P, word)
    if shape == None:
        print("The input tableau is not a P-tableau.")
        return
    T = make_matrix_from_T(P, word)
    graph = nx.from_scipy_sparse_matrix(T, create_using=nx.DiGraph)

    feat_dict = dict()
    for key, value in feature_list.items():
        if value[0] == True:
            feat_dict[key] = value[1](graph)

    feature = np.zeros((len(graph), len(feat_dict)))

    for n, node in enumerate(graph.nodes):
        for i, (key, value) in enumerate(feat_dict.items()):
            feature[n,i] = value[node]

    features = [feature]
    ys = np.array([1])
    adjacencies = convert_networkx_to_adjacency_input(graph)

    rows = [np.array(sp.coo_matrix(a).row, dtype=np.int16) for a in adjacencies]
    cols = [np.array(sp.coo_matrix(a).col, dtype=np.int16) for a in adjacencies]
    edge_types = [np.array(sp.coo_matrix(a).data, dtype=np.int16) for a in adjacencies]

    T_inputdata = InputData(features=features, labels=ys, rows=rows, columns=cols, edge_types=edge_types)
    T_dataset = CustomDataset(T_inputdata)
    T_loader = DataLoader(T_dataset, batch_size=1, shuffle=True)

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device ="cuda:0"

    with open(MODEL_FILE, 'rb') as f:
        model, acc = pickle.load(f)

    for batch in T_loader:
        batch.to(device)
        predicted = model(batch)
        print(predicted)
        print("---------")

