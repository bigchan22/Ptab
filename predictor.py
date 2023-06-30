# import functools
# import enum
import os

from BH.data_loader import *
from BH.generate_data import *
from predictor_info import *
# from Model_e import Model_e,Direction,Reduction
# from Train import train,print_accuracies

import pickle

from CustomDataset import *
from GCN_model import *

from torch_geometric.loader import DataLoader


# node_dim = num_features
# edge_dim = 8
# graph_deg = graph_deg
# depth = num_layers

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



def predict_orbit(P, word, shape_checkers, show=True):
    words = words_from_orbit(P, word)
    graphs = sp.coo_matrix(([], ([], [])), shape=(0,0), dtype=np.int16)
    
    for word in words:
        shape = shape_of_word(P, word)
        if shape == None: continue
        if all(shape_checker(shape)==False for shape_checker in shape_checkers):
            continue
        T = make_matrix_from_T(P, word)
        graphs = sp.block_diag((graphs, T))
    graphs = nx.from_scipy_sparse_matrix(graphs, create_using=nx.DiGraph)
    feat_dict = dict()
    for key, value in feature_list.items():
        if value[0] == True:
            feat_dict[key] = value[1](graphs)

    feature = np.zeros((len(graphs), len(feat_dict)))

    for n, node in enumerate(graphs.nodes):
        for i, (key, value) in enumerate(feat_dict.items()):
            feature[n,i] = value[node]
    features = [feature]
    ys = np.array([1])
    adjacencies = convert_networkx_to_adjacency_input(graphs)

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