from training_info import *

import pickle

from CustomDataset import *
from models.architectures.GCN_model import *

from torch_geometric.loader import DataLoader


def is_1row_graph(row, col, edge_type):
    if EDGE_TYPE.SINGLE_ARROW in edge_type:
        return False
    return True


def is_1row_graph(row, col, edge_type):
    if EDGE_TYPE.SINGLE_ARROW in edge_type:
        return False
    return True


def is_hook_graph(row, col, edge_type):
    parent = dict()
    for i in range(len(row)):
        if row[i] == 0: parent[col[i]] = -1
        if col[i] == 0: parent[row[i]] = -1
    for i in range(len(row)):
        if not row[i] in parent.keys(): continue
        if edge_type[i] == EDGE_TYPE.SINGLE_ARROW:
            parent[row[i]] = col[i]
    cnt = dict()
    for node in parent.keys():
        if parent[node] == -1:
            cnt[node] = 0
    for node in parent.keys():
        p = node
        while parent[p] != -1:
            p = parent[p]
        cnt[p] += 1
    cnt_list = []
    for node in cnt.keys():
        cnt_list.append(cnt[node])
    cnt_list.sort(reverse=True)
    if len(cnt_list) == 1 or cnt_list[1] == 1:
        return True
    return False


def is_2col_graph(row, col, edge_type):
    parent = dict()
    for i in range(len(row)):
        if row[i] == 0: parent[col[i]] = -1
        if col[i] == 0: parent[row[i]] = -1
    for i in range(len(row)):
        if not row[i] in parent.keys(): continue
        if edge_type[i] == EDGE_TYPE.SINGLE_ARROW:
            parent[row[i]] = col[i]
    cnt = 0
    for node in parent.keys():
        if parent[node] == -1:
            cnt += 1
    if cnt <= 2:
        return True
    return False


def test_acc(MODELS, DATA_PATH=None):
    features = {'norm_column': (True, normalized_column_indicator), }
    for feat in feature_list.keys():
        if feature_list[feat][0] == True:
            features[feat] = feature_list[feat][1]

    if DATA_PATH:
        DIR_PATH = DATA_PATH
    else:

        MODEL = MODELS[0]
        MODEL_DIR = MODEL[:MODEL[4:].index('/') + 4]
        graph_deg = MODEL[28]
        underbar_list = list(filter(lambda i: MODEL[i] == '_', range(len(MODEL))))
        a = underbar_list[4]
        b = underbar_list[-4]
        DIR_PATH = f'/Data/Ptab/n={graph_deg}' + MODEL[a:b]
    print("Loading test data...")

    graph_data = generate_graph_data(DIR_PATH, features)
    features = graph_data.features
    adjacencies = graph_data.adjacencies
    ys = graph_data.labels

    num_training = int(len(ys) * train_fraction)

    features_test = []
    rows_test = []
    cols_test = []
    edge_types_test = []
    ys_test = []

    for i in range(num_training, len(ys)):
        a = adjacencies[i]
        row = np.array(sp.coo_matrix(a).row, dtype=np.int16)
        col = np.array(sp.coo_matrix(a).col, dtype=np.int16)
        edge_type = np.array(sp.coo_matrix(a).data, dtype=np.int16)

        if is_1row_graph(row, col, edge_type): continue
        if is_hook_graph(row, col, edge_type): continue
        if is_2col_graph(row, col, edge_type): continue

        features_test.append(features[i])
        rows_test.append(row)
        cols_test.append(col)
        edge_types_test.append(edge_type)
        ys_test.append(ys[i])
    ys_test = np.array(ys_test)

    test_dataset = InputData(features=features_test, labels=ys_test, rows=rows_test, columns=cols_test,
                             edge_types=edge_types_test)
    test_dataset = CustomDataset(test_dataset)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    for i, MODEL in enumerate(MODELS):
        with open(MODEL, 'rb') as f:
            model_input = pickle.load(f)
            if len(model_input) == 3:
                model, GPU_NUM = model_input[0], "0"
            else:
                model, GPU_NUM = model_input[0], model_input[3]
            os.environ["CUDA_VISIBLE_DEVICES"] = GPU_NUM
            device = "cuda:" + GPU_NUM
            model.to(device)
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
                correct += ((predicted - batch.y) ** 2 < 0.1).sum().item()
        accuracy = correct / total
        # loss = float(loss.item())

        print("{}, Accuracy: {:.2%}".format(i, accuracy))
