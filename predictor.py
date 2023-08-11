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

from itertools import permutations as Perm



# node_dim = num_features
# edge_dim = 8
# graph_deg = graph_deg
# depth = num_layers

def load_models(show=True, MODEL_DIR='./trained_models', keywords=[], cutoff=0.1):
    MODELS = [os.path.join(MODEL_DIR, f) for f in os.listdir(MODEL_DIR)
              if f.endswith('.pickle') and all(keyword in f for keyword in keywords)]
    MODELS = []
    
    for f in os.listdir(MODEL_DIR):
        if not f.endswith('.pickle'): continue
        if not all(keyword in f for keyword in keywords): continue
        MODEL = os.path.join(MODEL_DIR, f)
        with open(MODEL, 'rb') as file:
            _, acc, _ = pickle.load(file)
            if acc > cutoff:
                MODELS.append(MODEL)
                if show == True:
                    MODEL = MODEL.split('parameters_')[-1].split('.')[0]
                    print(f'{len(MODELS)-1:3d} {acc:.4f} {MODEL}')
    return MODELS

def find_direction(MODEL):
    direction_dict = {"F": Direction.FORWARD,
                      "B": Direction.BACKWARD,
                      "2": Direction.BOTH}
    for direction1 in direction_dict.keys():
        for direction2 in direction_dict.keys():
            for direction3 in direction_dict.keys():
                if direction1+direction2+direction3 in MODEL:
                    return (direction_dict[direction1], direction_dict[direction2], direction_dict[direction3])
    return False

def find_feature(MODEL):
    features = dict()
    for key in feature_dict.keys():
        if 'ted_'+key in MODEL:
            features[key] = feature_dict[key]
    return features

def compare_models(P, word, MODELS, cutoff = 0.7):
    shape = shape_of_word(P, word)
    if shape == None:
        print("The input tableau is not a P-tableau.")
        return
    for MODEL in MODELS:
        pred_prob = predict_tableau(P, word, MODEL)
        if pred_prob > cutoff: pred = "GOOD"
        elif pred_prob < 1-cutoff: pred = " BAD"
        else: pred = "    "
        print(f"{pred_prob:.5f} {pred} {MODEL.split('/')[-1][11:-7]}")

def check_inclusion_criterion(MODEL, shape_checker=any_shape, cutoff = 0.7):
    cnt_pair = 0
    cnt_correct = 0
    N = int(MODEL.split('parameters_')[-1][0])
    for perm in Perm([i+1 for i in range(N)]):
        word = list(perm)
        Ps = dict()
        for P in generate_UIO(N):
            shape = shape_of_word(P, word)
            if shape == None: continue
            if shape_checker(shape) == False: continue
            pred_prob = predict_tableau(P, word, MODEL)
            if pred_prob > cutoff: pred = 'GOOD'
            elif pred_prob < 1 - cutoff: pred = 'BAD'
            else: pred = 'None'
            Ps[str(P)] = (pred_prob, pred, shape)
        for P1 in Ps.keys():
            for P2 in Ps.keys():
                if P1 == P2: continue
                if Ps[P1][2] != Ps[P2][2]: continue
                if is_included(P1, P2) == False: continue
                cnt_pair += 1
                if Ps[P1][0] < Ps[P2][0]: cnt_correct += 1
    return (cnt_correct / cnt_pair, cnt_pair, cnt_correct)

def is_included(P1, P2):
    for i in range(len(P1)):
        if P1[i] > P2[i]:
            return False
    return True

def predict_tableaux_around_tableau(P, word, MODEL, diameter = 1, cutoff = 0.7):
    shape = shape_of_word(P, word)
    if shape == None:
        print("The input tableau is not a P-tableau.")
        return
    
    print(f"T = {word}\nSHAPE = {shape}\nMODEL = {MODEL.split('/')[-1][11:-7]}")

    n = len(P)
    for Q in generate_UIO(n):
        diff = 0
        for i in range(n):
            if P[i] >= Q[i]: diff += P[i] - Q[i]
            else: diff += Q[i] - P[i]
        if diff > diameter: continue
        if shape_of_word(Q, word) != shape: continue
        pred_prob = predict_tableau(Q, word, MODEL)
        if pred_prob > cutoff: pred = "GOOD"
        elif pred_prob < 1-cutoff: pred = " BAD"
        else: pred = "    "
        print(f"{pred_prob:.5f} {pred} {Q} {diff}")

def predict_tableau(P, word, MODEL):
    shape = shape_of_word(P, word)
    if shape == None:
        print("The input tableau is not a P-tableau.")
        return
    direction1, direction2, direction3 = find_direction(MODEL)
    features = find_feature(MODEL)

    T = make_matrix_from_T(P, word, direction=(direction1, direction2, direction3))
    graph = nx.from_scipy_sparse_matrix(T, create_using=nx.DiGraph)

    feat_dict = dict()
    for key, value in features.items():
        feat_dict[key] = value(graph)

    feature = np.zeros((len(graph), len(feat_dict)))

    for n, node in enumerate(graph.nodes):
        for i, (key, value) in enumerate(feat_dict.items()):
            feature[n,i] = value[node]

    features = [feature]
    ys = np.array([1])
    adjacencies = [convert_networkx_to_adjacency_input(graph)]

    rows = [np.array(sp.coo_matrix(a).row, dtype=np.int16) for a in adjacencies]
    cols = [np.array(sp.coo_matrix(a).col, dtype=np.int16) for a in adjacencies]
    edge_types = [np.array(sp.coo_matrix(a).data, dtype=np.int16) for a in adjacencies]

    T_inputdata = InputData(features=features, labels=ys, rows=rows, columns=cols, edge_types=edge_types)
    T_dataset = CustomDataset(T_inputdata)
    T_loader = DataLoader(T_dataset, batch_size=1, shuffle=True)

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device ="cuda:0"

    with open(MODEL, 'rb') as f:
        model, acc, loss = pickle.load(f)
        model.to(device)

    for batch in T_loader:
        batch.to(device)
        predicted = model(batch)
        # print(predicted)
        # print("---------")
    return float(predicted[0])

def predict_orbit(P, word, shape_checkers, MODEL):
    words = words_from_orbit(P, word)
    graphs = sp.coo_matrix(([], ([], [])), shape=(0,0), dtype=np.int16)
    direction1, direction2, direction3 = find_direction(MODEL)
    features = find_feature(MODEL)

    for word in words:
        shape = shape_of_word(P, word)
        if shape == None: continue
        if all(shape_checker(shape)==False for shape_checker in shape_checkers):
            continue
        T = make_matrix_from_T(P, word, direction=(direction1, direction2, direction3))
        graphs = sp.block_diag((graphs, T))
    graphs = nx.from_scipy_sparse_matrix(graphs, create_using=nx.DiGraph)
    feat_dict = dict()
    for key, value in features.items():
        feat_dict[key] = value(graphs)

    feature = np.zeros((len(graphs), len(feat_dict)))

    for n, node in enumerate(graphs.nodes):
        for i, (key, value) in enumerate(feat_dict.items()):
            feature[n,i] = value[node]
    features = [feature]
    ys = np.array([1])
    adjacencies = [convert_networkx_to_adjacency_input(graphs)]

    rows = [np.array(sp.coo_matrix(a).row, dtype=np.int16) for a in adjacencies]
    cols = [np.array(sp.coo_matrix(a).col, dtype=np.int16) for a in adjacencies]
    edge_types = [np.array(sp.coo_matrix(a).data, dtype=np.int16) for a in adjacencies]

    T_inputdata = InputData(features=features, labels=ys, rows=rows, columns=cols, edge_types=edge_types)
    T_dataset = CustomDataset(T_inputdata)
    T_loader = DataLoader(T_dataset, batch_size=1, shuffle=True)

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device ="cuda:0"

    with open(MODEL, 'rb') as f:
        model, acc, loss = pickle.load(f)
        model.to(device)

    for batch in T_loader:
        batch.to(device)
        predicted = model(batch)
        # print(predicted)
        # print("---------")
    return float(predicted[0])