import torch
# from torch_geometric.data import Data
# from torch.utils.data import Dataset
import torch.nn.functional as F
# from torch_geometric.nn import GATv2Conv
from torch_geometric.nn import GCNConv


class GCN_single(torch.nn.Module):
    def __init__(self, num_edge_types, graph_deg, depth, node_dim, direction, edge_dim=8):
        super().__init__()
        self.num_edge_types = num_edge_types
        #         self.graph_deg = graph_deg
        self.depth = depth
        self.node_dim = node_dim
        self.direction = direction
        self.node_linear = torch.nn.Linear(1, self.node_dim)
        self.edge_linear = torch.nn.Linear(1, edge_dim)
        self.conv1 = GCNConv(self.node_dim, self.node_dim)
        self.conv2 = GCNConv(self.node_dim, self.node_dim)
        self.conv3 = GCNConv(self.node_dim, self.node_dim)
        self.conv4 = GCNConv(self.node_dim, self.node_dim)
        self.conv5 = GCNConv(self.node_dim, self.node_dim)

        self.conv1_b = GCNConv(self.node_dim, self.node_dim)
        self.conv2_b = GCNConv(self.node_dim, self.node_dim)
        self.conv3_b = GCNConv(self.node_dim, self.node_dim)
        self.conv4_b = GCNConv(self.node_dim, self.node_dim)
        self.conv5_b = GCNConv(self.node_dim, self.node_dim)

        self.out1 = torch.nn.Linear(self.node_dim, self.node_dim)
        self.out2 = torch.nn.Linear(self.node_dim, 1)
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
            index = torch.LongTensor([1, 0])

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
            if len(self.direction) > 3:
                mask5 = (edge_types == 5)
                edge_index5 = edge_index[:, mask5.squeeze()]
                if self.direction[3] == 'F' or self.direction[3] == '2':
                    x5 = self.conv5(x, edge_index5)
                    x_sum += x5
                if self.direction[3] == 'B' or self.direction[3] == '2':
                    edge_index5_b = torch.zeros_like(edge_index5)
                    edge_index5_b[index] = edge_index5
                    x5 = self.conv5_b(x, edge_index5_b)
                    x_sum += x5

            #######################

            x = F.relu(x_sum)
        batch = data.batch
        device = batch.device
        batch_size = len(data.y)
        row_counts = torch.bincount(batch, minlength=batch_size)
        num_graphs_per_batch = torch.div(row_counts, data.graph_sizes, rounding_mode='floor')
        num_graphs = int(torch.sum(row_counts / data.graph_sizes))
        max_columns = data.graph_sizes.max().item()
        graph_offset = torch.cumsum(num_graphs_per_batch, dim=0) - num_graphs_per_batch

        row_index = torch.cat([torch.arange(n_g, device=device).repeat_interleave(g_s) + offset
                               for n_g, g_s, offset in zip(num_graphs_per_batch, data.graph_sizes, graph_offset)])
        # Compute column indices for each batch
        col_index = torch.cat(
            [torch.arange(g_s, device=device).repeat(n_g) for n_g, g_s in zip(num_graphs_per_batch, data.graph_sizes)])

        xx = torch.full((num_graphs, max_columns, self.node_dim), 0., device=device)

        xx[row_index, col_index, :] = x.view(-1, self.node_dim)

        # xx = torch.reshape(x, (-1, self.graph_deg, self.node_dim))

        xxx, _ = torch.max(xx, dim=1)
        xxx = self.out1(xxx)
        xxx = F.relu(xxx)
        xxx = self.out2(xxx)

        return xxx


#         return F.log_softmax(xxx, dim=1)
class GCN_single_old(torch.nn.Module):
    def __init__(self, num_edge_types, graph_deg, depth, node_dim, direction, edge_dim=8):
        super().__init__()
        self.num_edge_types = num_edge_types
        self.graph_deg = graph_deg
        self.depth = depth
        self.node_dim = node_dim
        self.direction = direction
        self.node_linear = torch.nn.Linear(1, self.node_dim)
        self.edge_linear = torch.nn.Linear(1, edge_dim)
        self.conv1 = GCNConv(self.node_dim, self.node_dim)
        self.conv2 = GCNConv(self.node_dim, self.node_dim)
        self.conv3 = GCNConv(self.node_dim, self.node_dim)
        self.conv4 = GCNConv(self.node_dim, self.node_dim)
        self.conv5 = GCNConv(self.node_dim, self.node_dim)

        self.conv1_b = GCNConv(self.node_dim, self.node_dim)
        self.conv2_b = GCNConv(self.node_dim, self.node_dim)
        self.conv3_b = GCNConv(self.node_dim, self.node_dim)
        self.conv4_b = GCNConv(self.node_dim, self.node_dim)
        self.conv5_b = GCNConv(self.node_dim, self.node_dim)

        self.out1 = torch.nn.Linear(self.node_dim, self.node_dim)
        self.out2 = torch.nn.Linear(self.node_dim, 1)
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
            index = torch.LongTensor([1, 0])

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
            if len(self.direction) > 3:
                mask5 = (edge_types == 5)
                edge_index5 = edge_index[:, mask5.squeeze()]
                if self.direction[3] == 'F' or self.direction[3] == '2':
                    x5 = self.conv5(x, edge_index5)
                    x_sum += x5
                if self.direction[3] == 'B' or self.direction[3] == '2':
                    edge_index5_b = torch.zeros_like(edge_index5)
                    edge_index5_b[index] = edge_index5
                    x5 = self.conv5_b(x, edge_index5_b)
                    x_sum += x5

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
        self.num_edge_types = 5
        self.depth = depth
        self.graph_deg = graph_deg
        self.node_dim = node_dim
        self.direction = direction
        self.GCN_single = GCN_single(self.num_edge_types, self.graph_deg, self.depth, self.node_dim, self.direction)

    def forward(self, data, T=1):
        batch = data.batch
        batch = batch[::self.graph_deg]
        x = self.GCN_single(data)
        x = torch.sigmoid(x / T)
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


class GCN_multi_conv(torch.nn.Module):
    def __init__(self, graph_deg, depth, node_dim, direction):
        super().__init__()
        self.num_edge_types = 5
        self.depth = depth
        #         self.graph_deg = graph_deg
        self.node_dim = node_dim
        self.direction = direction
        self.GCN_single = GCN_single(self.num_edge_types, graph_deg, self.depth, self.node_dim, self.direction)

    def forward(self, data, T=1):
        batch = data.batch
        device = batch.device
        batch_size = len(data.y)
        row_counts = torch.bincount(batch, minlength=batch_size)
        num_graphs_per_batch = torch.div(row_counts, data.graph_sizes, rounding_mode='floor')

        batch = torch.cat(
            [torch.zeros(num_graphs_per_batch[i], dtype=int, device=device) + i for i in range(batch_size)])

        #         batch = batch[::self.graph_deg]
        x = self.GCN_single(data)
        #         x = torch.sigmoid(x/T)
        device = batch.device
        A = x.view(-1)
        ri = batch.view(-1)
        batch_size = batch[-1] + 1

        # Count occurrences of each row index in ri to determine column sizes
        row_counts = torch.bincount(ri, minlength=batch_size)

        # Determine the maximum number of columns needed
        max_columns = row_counts.max().item()

        # Initialize tensor B with -inf to indicate empty values
        B = torch.full((batch_size, max_columns), -float('inf'), device=device)

        # Create an empty tensor to keep track of the next available column index for each row
        col_indices = torch.zeros_like(ri)

        # Use row_counts to generate column indices
        for i in range(batch_size):
            col_indices[ri == i] = torch.arange(row_counts[i], device=device)

        # Place the values from A into B using advanced indexing
        B[ri, col_indices] = A
        B = torch.sigmoid(B / T)
        B_complement = 1 - B

        # Stack B and B_complement along a new dimension
        # This creates a new tensor with shape (64, max_columns, 2)
        stacked_tensor = torch.stack((B_complement, B), dim=-1)
        result = stacked_tensor[:, 0, :]
        for i in range(1, stacked_tensor.size(1)):
            # Extract the next slice to serve as the kernel, which has shape (1, 1, 64, 2)
            kernel_2d = stacked_tensor[:, i, :].unsqueeze(1)
            kernel_2d_flipped = torch.flip(kernel_2d, dims=[-1])

            # Reshape the input for Conv1d: treat each row as a separate channel
            input_for_conv1d = result.unsqueeze(0)  # Shape becomes (1, H, W)

            # Apply Conv1d with groups=H to perform independent convolution on each row
            # in_channels = H (number of rows), out_channels = H (each row produces its own output),
            # groups=H ensures each row is convolved with its own corresponding kernel row
            result = F.conv1d(input_for_conv1d, kernel_2d_flipped, groups=batch_size, padding=1)

            # Remove the extra dimension to get the final result
            result = result.squeeze()

        # The final result will be in `result`

        return result
