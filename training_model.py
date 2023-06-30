import functools
import enum
import os

from BH.data_loader import *
from BH.generate_data import *
from training_info import *
# from Model_e import Model_e,Direction,Reduction
from Train import train,print_accuracies

import pickle

from CustomDataset import *
from GCN_model import *



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
    with open(MODEL_FILE, 'rb') as f:
      model, max_accuracy = pickle.load(f)
else:
    model = GCN_multi(graph_deg, depth, node_dim).to(device)
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

    if accuracy > max_accuracy and save_trained_weights:
        max_accuracy = accuracy
        with open(MODEL_FILE, 'wb') as f:
            pickle.dump((model, max_accuracy), f)

