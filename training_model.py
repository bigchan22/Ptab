# from data.data_loader import *
# from data.generate_data import *
from training_info import *

import pickle

from CustomDataset import *
from models.architectures.GCN_model import *

from torch_geometric.loader import DataLoader

os.environ["CUDA_VISIBLE_DEVICES"] = GPU_NUM
device = "cuda:" + GPU_NUM

print("Loading input data...")
full_dataset, train_dataset, test_dataset = load_input_data(DIR_PATH)

node_dim = num_features
edge_dim = 8
graph_deg = graph_deg
depth = num_layers

test_dataset = CustomDataset(test_dataset)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

train_dataset = CustomDataset(train_dataset)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = pastGCN().to(device)
if use_pretrained_weights == True:
    try:
        with open(MODEL_FILE, 'rb') as f:
            model, max_accuracy, min_loss = pickle.load(f)
            model.to(device)
    except:
        print("There is no trained model")
        use_pretrained_weights = False
if use_pretrained_weights == False:
    if GCN_multi_stack == "conv":
        model = GCN_multi_conv(graph_deg, depth, node_dim, direction).to(device)
    if GCN_multi_stack == 'sum':
        model = GCN_multi(graph_deg, depth, node_dim, direction).to(device)
    max_accuracy = 0
    min_loss = 100
# data = batch.to(device)
# torch.nn.init.xavier_normal(model)
if GCN_multi_stack == "conv":
    loss_function = torch.nn.NLLLoss()
if GCN_multi_stack == 'sum':
    loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=step_size, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                              lr_lambda=lambda epoch: 0.993 ** epoch,
                                              last_epoch=-1,
                                              verbose=False)

for epoch in range(num_epochs):
    # Training phase
    model.train()
    for batch in train_loader:
        batch.to(device)
        optimizer.zero_grad()
        out = model(batch)

        if GCN_multi_stack == "sum":
            batch.y = batch.y.float()
            loss = loss_function(out, batch.y)
        if GCN_multi_stack == "conv":
            log_probs = torch.log(torch.clamp(out, min=1e-9))
            loss = loss_function(log_probs, batch.y)
        loss.backward()
        optimizer.step()
    scheduler.step()
    #    print(loss)

    # Evaluation phase
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            batch.to(device)
            outputs = model(batch)
            #             _,predicted = torch.max(outputs.data, 1)
            if GCN_multi_stack == "sum":
                predicted = outputs
                total += batch.y.size(0)
                #             correct += (predicted == batch.y).sum().item()
                correct += ((predicted - batch.y) ** 2 < 0.1).sum().item()
            if GCN_multi_stack == "conv":
                #                 print(outputs.data.shape)
                _, predicted = torch.max(outputs.data, 1)
                total += batch.y.size(0)
                #             correct += (predicted == batch.y).sum().item()
                correct += (predicted == batch.y).sum().item()
    # Compute accuracy
    accuracy = correct / total
    loss = float(loss.item())

    print("Epoch [{}/{}], Accuracy: {:.2%}, Loss: {:.15f}".format(epoch + 1, num_epochs, accuracy, loss))

    if (accuracy > max_accuracy or (accuracy == max_accuracy and loss < min_loss)) and save_trained_weights:
        max_accuracy = accuracy
        min_loss = loss
        with open(MODEL_FILE, 'wb') as f:
            pickle.dump((model, max_accuracy, min_loss), f)
