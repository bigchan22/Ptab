import torch
# from torch_geometric.data import Data
# from torch.utils.data import Dataset
import torch.nn.functional as F
# from torch_geometric.nn import GATv2Conv
from torch_geometric.nn import GCNConv

class GCN_single(torch.nn.Module):
    def __init__(self,num_edge_types, graph_deg, depth, node_dim, direction, edge_dim=8):
        super().__init__()
        self.num_edge_types = num_edge_types
        self.graph_deg = graph_deg
        self.depth = depth
        self.node_dim = node_dim
        self.direction = direction
        self.node_linear = torch.nn.Linear(1,self.node_dim)
        self.edge_linear = torch.nn.Linear(1,edge_dim)
        self.conv1 = GCNConv(self.node_dim, self.node_dim)
        self.conv2 = GCNConv(self.node_dim, self.node_dim)
        self.conv3 = GCNConv(self.node_dim, self.node_dim)
        self.conv4 = GCNConv(self.node_dim, self.node_dim)
        
        self.conv1_b = GCNConv(self.node_dim, self.node_dim)
        self.conv2_b = GCNConv(self.node_dim, self.node_dim)
        self.conv3_b = GCNConv(self.node_dim, self.node_dim)
        self.conv4_b = GCNConv(self.node_dim, self.node_dim)

        self.out1 = torch.nn.Linear(self.node_dim,self.node_dim)
        self.out2 = torch.nn.Linear(self.node_dim,1)
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
            index = torch.LongTensor([1,0])
            
            mask1 = (edge_types == 1)
            edge_index1 = edge_index[:, mask1.squeeze()]
            x1 = self.conv1(x, edge_index1)
            
            x_sum = x1

            #######################

            mask2 = (edge_types == 2)
            edge_index2 = edge_index[:, mask2.squeeze()]
            if self.direction[0] == 'F' or self.direction[0] == '2':
                x2 = self.conv2(x, edge_index2)
                x_sum += x2
            if self.direction[0] == 'B' or self.direction[0] == '2':
                edge_index2_b = torch.zeros_like(edge_index2) 
                edge_index2_b[index] = edge_index2
                x2 = self.conv2_b(x, edge_index2_b)
                x_sum += x2

            #######################

            mask3 = (edge_types == 3)
            edge_index3 = edge_index[:, mask3.squeeze()]
            if self.direction[1] == 'F' or self.direction[1] == '2':
                x3 = self.conv3(x, edge_index3)
                x_sum += x3
            if self.direction[1] == 'B' or self.direction[1] == '2':
                edge_index3_b = torch.zeros_like(edge_index3) 
                edge_index3_b[index] = edge_index3
                x3 = self.conv3_b(x, edge_index3_b)
                x_sum += x3

            #######################
            
            mask4 = (edge_types == 4)
            edge_index4 = edge_index[:, mask4.squeeze()]
            if self.direction[2] == 'F' or self.direction[2] == '2':
                x4 = self.conv4(x, edge_index4)
                x_sum += x4
            if self.direction[2] == 'B' or self.direction[2] == '2':
                edge_index4_b = torch.zeros_like(edge_index4) 
                edge_index4_b[index] = edge_index4
                x4 = self.conv4_b(x, edge_index4_b)
                x_sum += x4

            #######################
            
            x = F.relu(x_sum)
        xx = torch.reshape(x, (-1, self.graph_deg, self.node_dim))
        xxx, _ = torch.max(xx, dim=1)
        xxx = self.out1(xxx)
        xxx = F.relu(xxx)
        xxx = self.out2(xxx)
        
        return xxx
#         return F.log_softmax(xxx, dim=1)
    
class GCN_multi(torch.nn.Module):
    def __init__(self, graph_deg, depth, node_dim, direction):
        super().__init__()
        self.num_edge_types = 4
        self.depth = depth
        self.graph_deg = graph_deg
        self.node_dim = node_dim
        self.direction = direction 
        self.GCN_single = GCN_single(self.num_edge_types, self.graph_deg, self.depth, self.node_dim, self.direction)

    def forward(self, data, T=1):
        batch = data.batch
        batch = batch[::self.graph_deg]
        x = self.GCN_single(data)
        x = torch.sigmoid(x/T)
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
