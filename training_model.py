import functools
import enum
import os

from BH.data_loader import *
from BH.generate_data import *
from training_info import *
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





os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device ="cuda:0"

print("Loading input data...")
full_dataset, train_dataset, test_dataset = load_input_data(DIR_PATH, train_fraction)

node_dim = num_features
edge_dim = 8
graph_deg = graph_deg
depth = num_layers

test_dataset = CustomDataset(test_dataset)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

train_dataset = CustomDataset(train_dataset)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = pastGCN().to(device)
if use_pretrained_weights:
    with open(MODEL_FILE, 'r') as f:
      model, max_accuracy = pickle.load(f)
else:
    model = GCN_multi(graph_deg, depth).to(device)
    max_accuracy = 0
# data = batch.to(device)
# torch.nn.init.xavier_normal(model)
loss_function = torch.nn.CrossEntropyLoss()
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=step_size, weight_decay=5e-4)

for epoch in range(num_epochs):
    # Training phase
    model.train()
    for batch in train_loader:
        batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        
        batch.y = batch.y.float()
        loss = loss_function(out, batch.y)
        loss.backward()
        optimizer.step()
    print(loss)
    
    # Evaluation phase
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            batch.to(device)
            outputs = model(batch)
#             _,predicted = torch.max(outputs.data, 1)
            predicted = outputs
            total += batch.y.size(0)
#             correct += (predicted == batch.y).sum().item()
            correct += ((predicted - batch.y)**2<0.1).sum().item()

    # Compute accuracy
    accuracy = correct / total

    print("Epoch [{}/{}], Accuracy: {:.2%}".format(epoch + 1, num_epochs, accuracy))

    if accuracy > max_accuracy:
        max_accuracy = accuracy
        with open(MODEL_FILE, 'w') as f:
            pickle.dump((model, max_accuracy))

