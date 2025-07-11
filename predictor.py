# import functools
# import enum
import pickle
from itertools import permutations as Perm

import scipy.sparse as sp
from torch_geometric.loader import DataLoader

from predictor_info import *
from src.CustomDataset import CustomDataset
from src.data.Data_gen_utils import shape_of_word, generate_UIO, words_from_orbit, Direction
from src.data.data_loader import convert_networkx_to_adjacency_input, InputData
from src.data.generate_data import make_matrix_from_T
from src.data.shapes import any_shape


# from Model_e import Model_e,Direction,Reduction
# from Train import train,print_accuracies


# node_dim = num_features
# edge_dim = 8
# graph_deg = graph_deg
# depth = num_layers

def load_models(show=True, MODEL_DIR='./models/trained_models', keywords=[], cutoff=0.1):
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
                    print(f'{len(MODELS) - 1:3d} {acc:.4f} {MODEL}')
    return MODELS


def find_direction(MODEL):
    direction_dict = {"F": Direction.FORWARD,
                      "B": Direction.BACKWARD,
                      "2": Direction.BOTH}
    for direction1 in direction_dict.keys():
        for direction2 in direction_dict.keys():
            for direction3 in direction_dict.keys():
                if direction1 + direction2 + direction3 in MODEL:
                    return (direction_dict[direction1], direction_dict[direction2], direction_dict[direction3])
    return (Direction.FORWARD, Direction.FORWARD, Direction.FORWARD)


def find_feature(MODEL):
    feature_dict = {
        'constant': (True, constant_feature),
        'column': (False, column_indicator),
        'norm_column': (False, normalized_column_indicator),  # Feauture
        'norm_column_rev': (False, normalized_column_rev_indicator),
    }

    features = dict()

    for key in feature_dict.keys():
        if key in MODEL:
            features[key] = feature_dict[key]
    return features


def compare_models(P, word, MODELS, cutoff=0.7):
    shape = shape_of_word(P, word)
    if shape == None:
        print("The input tableau is not a P-tableau.")
        return
    for MODEL in MODELS:
        pred_prob = predict_tableau(P, word, MODEL)
        if pred_prob > cutoff:
            pred = "GOOD"
        elif pred_prob < 1 - cutoff:
            pred = " BAD"
        else:
            pred = "    "
        print(f"{pred_prob:.5f} {pred} {MODEL.split('/')[-1][11:-7]}")


def check_inclusion_criterion(MODEL, shape_checker=any_shape, gap=0.5, cutoff=0.7):
    cnt_pair = 0
    cnt_incorrect = 0
    N = int(MODEL.split('parameters_')[-1][0])
    for perm in Perm([i + 1 for i in range(N)]):
        word = list(perm)
        Ps = dict()
        for P in generate_UIO(N):
            shape = shape_of_word(P, word)
            if shape == None: continue
            if shape_checker(shape) == False: continue
            pred_prob = predict_tableau(P, word, MODEL)
            if pred_prob > cutoff:
                pred = 'GOOD'
            elif pred_prob < 1 - cutoff:
                pred = 'BAD'
            else:
                pred = 'None'
            Ps[str(P)] = (pred_prob, pred, shape)
        for P1 in Ps.keys():
            for P2 in Ps.keys():
                if P1 == P2: continue
                if Ps[P1][2] != Ps[P2][2]: continue
                if is_included(P1, P2) == False: continue
                cnt_pair += 1
                if Ps[P1][1] == 'GOOD' and Ps[P2][1] == 'BAD': cnt_incorrect += 1
    return (1 - (cnt_incorrect / cnt_pair), cnt_pair, cnt_pair - cnt_incorrect)


def is_included(P1, P2):
    for i in range(len(P1)):
        if P1[i] > P2[i]:
            return False
    return True


def predict_tableaux_around_tableau(P, word, MODEL, diameter=1, cutoff=0.7):
    shape = shape_of_word(P, word)
    if shape == None:
        print("The input tableau is not a P-tableau.")
        return

    print(f"T = {word}\nSHAPE = {shape}\nMODEL = {MODEL.split('/')[-1][11:-7]}")

    n = len(P)
    for Q in generate_UIO(n):
        diff = 0
        for i in range(n):
            if P[i] >= Q[i]:
                diff += P[i] - Q[i]
            else:
                diff += Q[i] - P[i]
        if diff > diameter: continue
        if shape_of_word(Q, word) != shape: continue
        pred_prob = predict_tableau(Q, word, MODEL)
        if pred_prob > cutoff:
            pred = "GOOD"
        elif pred_prob < 1 - cutoff:
            pred = " BAD"
        else:
            pred = "    "
        print(f"{pred_prob:.5f} {pred} {Q} {diff}")


def predict_tableau(P, word, MODEL):
    shape = shape_of_word(P, word)
    if shape == None:
        print("The input tableau is not a P-tableau.")
        return

    direction1, direction2, direction3 = find_direction(MODEL)
    # direction1, direction2, direction3 = Direction.FORWARD, Direction.FORWARD, Direction.FORWARD
    features = find_feature(MODEL)

    T = make_matrix_from_T(P, word, direction=(direction1, direction2, direction3))
    graph = nx.from_scipy_sparse_array(T, create_using=nx.DiGraph)

    feat_dict = dict()
    for key, value in features.items():
        feat_dict[key] = value[1](graph)

    feature = np.zeros((len(graph), len(feat_dict)))

    for n, node in enumerate(graph.nodes):
        for i, (key, value) in enumerate(feat_dict.items()):
            feature[n, i] = value[node]

    features = [feature]
    ys = np.array([1])
    adjacencies = [convert_networkx_to_adjacency_input(graph)]

    rows = [np.array(sp.coo_matrix(a).row, dtype=np.int16) for a in adjacencies]
    cols = [np.array(sp.coo_matrix(a).col, dtype=np.int16) for a in adjacencies]
    edge_types = [np.array(sp.coo_matrix(a).data, dtype=np.int16) for a in adjacencies]
    graphs_sizes = [len(graph.nodes)]
    print(len(graph.nodes))
    # T_inputdata = InputData(features=features, labels=ys, rows=rows, columns=cols, edge_types=edge_types)
    T_inputdata = InputData(features=features, labels=ys, rows=rows, columns=cols, edge_types=edge_types, graph_sizes=graphs_sizes)
    T_dataset = CustomDataset(T_inputdata)
    T_loader = DataLoader(T_dataset, batch_size=1, shuffle=True)

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = "cuda:0"

    with open(MODEL, 'rb') as f:
        model, acc, loss = pickle.load(f)
        model.to(device)

    for batch in T_loader:
        batch.to(device)
        print(batch.batch)
        print(batch.x)
        print(batch.edge_index)
        print(batch.edge_types)
        predicted = model(batch)
        print(predicted)
        print("---------")
        return batch, predicted


def predict_orbit(P, word, shape_checkers, MODEL):
    words = words_from_orbit(P, word)
    graphs = sp.coo_matrix(([], ([], [])), shape=(0, 0), dtype=np.int16)

    # direction1, direction2, direction3 = find_direction(MODEL)
    direction1, direction2, direction3 = Direction.FORWARD, Direction.FORWARD, Direction.FORWARD

    features = find_feature(MODEL)

    for word in words:
        shape = shape_of_word(P, word)
        if shape == None: continue
        if all(shape_checker(shape) == False for shape_checker in shape_checkers):
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
            feature[n, i] = value[node]
    features = [feature]
    ys = np.array([1])
    adjacencies = [convert_networkx_to_adjacency_input(graphs)]

    rows = [np.array(sp.coo_matrix(a).row, dtype=np.int16) for a in adjacencies]
    cols = [np.array(sp.coo_matrix(a).col, dtype=np.int16) for a in adjacencies]
    edge_types = [np.array(sp.coo_matrix(a).data, dtype=np.int16) for a in adjacencies]
    graph_sizes = []

    T_inputdata = InputData(features=features, labels=ys, rows=rows, columns=cols, edge_types=edge_types, )
    T_dataset = CustomDataset(T_inputdata)
    T_loader = DataLoader(T_dataset, batch_size=1, shuffle=True)

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = "cuda:0"

    with open(MODEL, 'rb') as f:
        model, acc, loss = pickle.load(f)
        model.to(device)

    for batch in T_loader:
        batch.to(device)
        predicted = model(batch)
        # print(predicted)
        # print("---------")
    return float(predicted[0])
