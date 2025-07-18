# import json
import itertools
import json
import os

import numpy as np
import scipy.sparse as sp

from src.data.Data_gen_utils import generate_UIO, is_P_compatible, is_P_less, P_Des, words_from_orbit, shape_of_word, \
    iter_shuffles, cluster_vertices, EDGE_TYPE, Direction, orbits_from_P, PTab_from_word
from src.data.criterion import *
from src.data.shapes import *
# check_all_row_connected, is_good_P_1row_B
class GraphSaver:
    def __init__(self, dir_path):
        self.dir_path = dir_path
        self.counter = 0
        self.graphs = []
        self.labels = []
        self.graph_sizes = []

    def save_streaming(self, g, label, N):
        file_path = os.path.join(self.dir_path, f"graph_{self.counter:05d}.npz")
        sp.save_npz(file_path, g)
        with open(os.path.join(self.dir_path, "labels.jsonl"), 'a') as f_label:
            f_label.write(f"{label}\n")
        with open(os.path.join(self.dir_path, "graph_sizes.jsonl"), 'a') as f_size:
            f_size.write(f"{N}\n")
        self.counter += 1

    def save_batch(self, g, label, N):
        self.graphs.append(g)
        self.labels.append(label)
        self.graph_sizes.append(N)


# === Step 1: Load Metadata ===
def load_metadata(json_path):
    with open(os.path.join(json_path, "Partitions.json")) as f:
        Partitions = json.load(f)
    with open(os.path.join(json_path, "PartitionIndex.json")) as f:
        PartitionIndex = json.load(f)
    with open(os.path.join(json_path, "TransitionMatrix.json")) as f:
        TM = json.load(f)
    return Partitions, PartitionIndex, TM

# === Step 2: Generate Words from Poset ===
def generate_words(P, N, primitive):
    if primitive:
        return iter_shuffles(cluster_vertices(P))
    else:
        return itertools.permutations(range(1, N + 1))

# === Step 3: Process Single Word ===
def process_word(P, word, shape_checkers, column_info):
    shape = shape_of_word(P, word)
    if shape is None:
        return None, None
    if not any(checker(shape) for checker in shape_checkers):
        return None, None

    if column_info == "original":
        g = make_matrix_from_T(P, word)
    elif column_info == "column_direction":
        g = make_matrix_from_T_col_info(P, word)
    elif column_info == "column_direc_column_same":
        g = make_matrix_from_T_col_info(P, word,
            direction=(Direction.FORWARD, Direction.BOTH, Direction.FORWARD))
    else:
        raise ValueError("Invalid column_info option")

    return shape, g

# === Step 4: Label Decision ===
def decide_label(P, word):
    chk = check_all_row_connected(P, word)
    if chk == "GOOD":
        return 1
    elif chk == "BAD":
        return 0
    elif chk == "UNKNOWN":
        return None
    else:
        raise ValueError("Unexpected chk result")

# === Step 5: Compute Schur Label ===
def compute_transition_label(TM_n, Fs, k):
    return sum(TM_n[i][k] * Fs[i] for i in range(len(Fs)))

# === Step 6: Save Graph and Metadata ===
def write_graph_and_metadata(g, label, size, DIR_PATH, idx):
    path = os.path.join(DIR_PATH, f"graph_{idx:05d}.npz")
    sp.save_npz(path, g)
    return {"index": idx, "label": label, "size": size}

# === Utility Functions ===
def load_metadata(json_path):
    with open(os.path.join(json_path, "Partitions.json")) as f:
        Partitions = json.load(f)
    with open(os.path.join(json_path, "PartitionIndex.json")) as f:
        PartitionIndex = json.load(f)
    with open(os.path.join(json_path, "TransitionMatrix_btw_s_h.json")) as f:
        TM = json.load(f)
    return Partitions, PartitionIndex, TM

def init_shape_counters(N, Partitions):
    gs, s_vec, pre_calc = {}, {}, {}
    for k in range(1, N+1):
        gs[k], s_vec[k], pre_calc[k] = {}, {}, {}
        for lamb in Partitions:
            lamb_str = str(lamb)
            gs[k][lamb_str] = sp.coo_matrix(([], ([], [])), shape=(0, 0), dtype=np.int16)
            s_vec[k][lamb_str] = 0
            pre_calc[k][lamb_str] = 0
    return gs, s_vec, pre_calc

def generate_graph_from_word(P, word, shape_checkers, column_info):
    shape = shape_of_word(P, word)
    if shape is None or not any(checker(shape) for checker in shape_checkers):
        return None, None
    if column_info == "original":
        g = make_matrix_from_T(P, word)
    elif column_info == "column_direction":
        g = make_matrix_from_T_col_info(P, word)
    elif column_info == "column_direc_column_same":
        g = make_matrix_from_T_col_info(P, word, direction=(Direction.FORWARD, Direction.BOTH, Direction.FORWARD))
    else:
        raise ValueError("Invalid column_info")
    return shape, g

def filter_word_and_calculate_scoeff(P, word, shape_checkers, column_info, filter_fn,
                                     gs, pre_calc, s_vec, saver, N, mode):
    shape, g = generate_graph_from_word(P, word, shape_checkers, column_info)
    if shape is None:
        return
    T = PTab_from_word(P, word)
    if mode == "decomp":
        k = T[0][0]
    elif mode == "position_one":
        k = T[0].index(1) + 1
    elif mode == "vanilla" or mode == "ppath" or mode == "test":
        k = 1
    else: 
        raise ValueError
    shape_str = str(shape)
    s_vec[k][shape_str] += 1
    result = filter_fn(P, word)
    if result == 'UNKNOWN':
        gs[k][shape_str] = sp.block_diag((gs[k][shape_str], g))
    elif result == 'BAD':
        saver.save_streaming(g, label=0, N=N)
    elif result == 'GOOD':
        saver.save_streaming(g, label=1, N=N)
        pre_calc[k][shape_str] += 1
    else:
        raise Exception(f"{P}, {word}, {k}, {shape_str}: filter error")
        
def calculate_hcoeff(TM_n, N, Partitions, gs, pre_calc, s_vec,
                     shape_checkers, P, word, saver):
    for k in s_vec.keys():
        for lamb in Partitions:
            lamb_str = str(lamb)
            if gs[k][lamb_str].size == 0 and pre_calc[k][lamb_str] == 0:
                continue
            if not any(chk(lamb) for chk in shape_checkers):
                continue
            h_coeff = 0
            for mu in Partitions:
                mu_str = str(mu)
                h_coeff += TM_n[mu_str][lamb_str] * s_vec[k][mu_str]
            if h_coeff < 0:
                raise Exception(f"{P}, {word}, {k}, {lamb}, {h_coeff}: negative coefficient!")
#             est_count = int(gs[k][lamb_str].size / Nchoose2)
            est_count = gs[k][lamb_str].shape[0] // N
#             if est_count != int(gs[k][lamb_str].size / Nchoose2):
#                 print(gs[k][lamb_str])
#                 print(est_count, int(gs[k][lamb_str].size / Nchoose2))
#                 raise ValueError
            usable = h_coeff - pre_calc[k][lamb_str]
            if est_count < usable:
                raise Exception(f"{P}, {word}, {k}, {lamb}, {h_coeff}: insufficient filtered data!")
            saver.save_streaming(gs[k][lamb_str], usable, N)


def process_orbits(P, orbit, N, Partitions, TM_n, shape_checkers,
                   column_info, filter_fn, saver, mode):
    gs, s_vec, pre_calc = init_shape_counters(N, Partitions)
    for word in orbit:
        filter_word_and_calculate_scoeff(P, word, shape_checkers, column_info, filter_fn,
                                         gs, pre_calc, s_vec, saver, N, mode)
    calculate_hcoeff(TM_n, N, Partitions,
                     gs, pre_calc, s_vec, shape_checkers, P, word, saver)
#     print(f"orbit with {orbit[0]} completed")

def post_shuffle_graph_data(dir_path, output_path=None):
    if output_path is None:
        output_path = os.path.join(dir_path, "shuffled")

    os.makedirs(output_path, exist_ok=True)

    # Load labels and sizes
    with open(os.path.join(dir_path, "labels.jsonl")) as f:
        labels = [int(line.strip()) for line in f]

    with open(os.path.join(dir_path, "graph_sizes.jsonl")) as f:
        graph_sizes = [int(line.strip()) for line in f]

    # Get list of .npz files
    graph_files = sorted([
        f for f in os.listdir(dir_path) if f.endswith(".npz")
    ])
    graph_files = [os.path.join(dir_path, f) for f in graph_files]

    assert len(labels) == len(graph_files) == len(graph_sizes), "Mismatch in data lengths!"

    # Shuffle indices
    indices = np.random.permutation(len(graph_files))

    # Save shuffled files
    shuffled_labels = []
    shuffled_sizes = []

    for new_idx, old_idx in enumerate(indices):
        # Copy and rename .npz file
        g = sp.load_npz(graph_files[old_idx])
        out_graph_file = os.path.join(output_path, f"graph_{new_idx:05d}.npz")
        sp.save_npz(out_graph_file, g)

        # Save label and size
        shuffled_labels.append(labels[old_idx])
        shuffled_sizes.append(graph_sizes[old_idx])

    # Save labels and sizes
    with open(os.path.join(output_path, "labels.json"), 'w') as f:
        json.dump(shuffled_labels, f)

    with open(os.path.join(output_path, "graph_sizes.json"), 'w') as f:
        json.dump(shuffled_sizes, f)

    print(f"Shuffled dataset saved to: {output_path}")
    
    
def generate_data_PTabs(DIR_PATH,
                        input_N,
                        shape_checkers = [any_shape],
                        filter_fn = trivial_criterion,
                        primitive = True,
                        connected = False,
                        UPTO_N = False,
                        json_path = "src/data/json/",
                        column_info = 'original',
                        mode = "decomp"):

    Partitions, PartitionIndex, TM = load_metadata(json_path)

    # Create saver
    saver = GraphSaver(DIR_PATH)  # your defined class with streaming save

    if UPTO_N:
        N = 1
    else:
        N = input_N

    while N <= input_N:
        n_str = str(N)
        TM_n = TM[n_str]
        Partitions_n = Partitions[n_str]
        PosetList = generate_UIO(N, connected=connected)
        print("PosetList generated!")
        for P in PosetList:
            orbitList = orbits_from_P(P, primitive)
#             print("Orbit list generated!")
            for orbit in orbitList:
                process_orbits(P, orbit, N, Partitions_n, TM_n,
                               shape_checkers, column_info, filter_fn,
                               saver, mode)
            print(f"finished for orbit {P}")

        print(f"data Gen finished for N = {N}")
        N += 1

    # Shuffle all .npz/label/size files after full streaming generation
    post_shuffle_graph_data(DIR_PATH)
    
    
## Given a poset P and a word (which is a column word of some P-tableau), return the graph model of the input tableau as a scipy matrix.
## The additional parameter 'direction' determines directions of edges, but at this moment, we do not use this parameter.
## This version of make_matrix_from_T is old, which means that the matrix has 3 types of edges (DASHED_ARROW, DASHED_ARROW, DOUBLE_ARROW)
def make_matrix_from_T(P, word, direction=(Direction.BOTH, Direction.BOTH, Direction.BOTH)):
    n = len(word)
    row = []
    col = []
    edge_type = []

    col_index = [1]
    for i in range(1, n):
        if is_P_less(P, word[i], word[i - 1]):
            col_index.append(col_index[-1])
        else:
            col_index.append(col_index[-1] + 1)
    for i in range(n):
        for j in range(i + 1, n):
            if not is_P_compatible(P, word[i], word[j]):
                if direction[0] == Direction.FORWARD or direction[0] == Direction.BOTH:
                    row.append(word[i] - 1)
                    col.append(word[j] - 1)
                    edge_type.append(EDGE_TYPE.DASHED_ARROW)
                if direction[0] == Direction.BACKWARD or direction[0] == Direction.BOTH:
                    row.append(word[j] - 1)
                    col.append(word[i] - 1)
                    edge_type.append(EDGE_TYPE.DASHED_ARROW)
            elif col_index[i] == col_index[j]:
                if direction[1] == Direction.FORWARD or direction[1] == Direction.BOTH:
                    row.append(word[j] - 1)
                    col.append(word[i] - 1)
                    edge_type.append(EDGE_TYPE.SINGLE_ARROW)
                if direction[1] == Direction.BACKWARD or direction[1] == Direction.BOTH:
                    row.append(word[i] - 1)
                    col.append(word[j] - 1)
                    edge_type.append(EDGE_TYPE.SINGLE_ARROW)
            else:
                if direction[2] == Direction.FORWARD or direction[2] == Direction.BOTH:
                    row.append(min(word[i], word[j]) - 1)
                    col.append(max(word[i], word[j]) - 1)
                    edge_type.append(EDGE_TYPE.DOUBLE_ARROW)
                if direction[2] == Direction.BACKWARD or direction[2] == Direction.BOTH:
                    row.append(max(word[i], word[j]) - 1)
                    col.append(min(word[i], word[j]) - 1)
                    edge_type.append(EDGE_TYPE.DOUBLE_ARROW)
    return sp.coo_matrix((edge_type, (row, col)), shape=(n, n))


def make_matrix_from_T_col_info(P, word, direction=(Direction.BOTH, Direction.BOTH, Direction.BOTH)):
    n = len(word)
    row = []
    col = []
    edge_type = []

    col_index = [1]
    for i in range(1, n):
        if is_P_less(P, word[i], word[i - 1]):
            col_index.append(col_index[-1])
        else:
            col_index.append(col_index[-1] + 1)
    for i in range(n):
        for j in range(i + 1, n):
            if not is_P_compatible(P, word[i], word[j]):
                if direction[0] == Direction.FORWARD or direction[0] == Direction.BOTH:
                    row.append(word[i] - 1)
                    col.append(word[j] - 1)
                    edge_type.append(EDGE_TYPE.DASHED_ARROW)
                if direction[0] == Direction.BACKWARD or direction[0] == Direction.BOTH:
                    row.append(word[j] - 1)
                    col.append(word[i] - 1)
                    edge_type.append(EDGE_TYPE.DASHED_ARROW)
            elif col_index[i] == col_index[j]:
                if direction[1] == Direction.FORWARD or direction[1] == Direction.BOTH:
                    row.append(word[j] - 1)
                    col.append(word[i] - 1)
                    edge_type.append(EDGE_TYPE.SINGLE_ARROW)
                if direction[1] == Direction.BACKWARD or direction[1] == Direction.BOTH:
                    row.append(word[i] - 1)
                    col.append(word[j] - 1)
                    edge_type.append(EDGE_TYPE.SINGLE_ARROW)
            else:
                if direction[2] == Direction.FORWARD or direction[2] == Direction.BOTH:
                    row.append(word[i] - 1)
                    col.append(word[j] - 1)
                    edge_type.append(EDGE_TYPE.DOUBLE_ARROW)
                if direction[2] == Direction.BACKWARD or direction[2] == Direction.BOTH:
                    row.append(word[j] - 1)
                    col.append(word[i] - 1)
                    edge_type.append(EDGE_TYPE.DOUBLE_ARROW)
    return sp.coo_matrix((edge_type, (row, col)), shape=(n, n))


## Given a poset P and a word (which is a column word of some P-tableau), return the graph model of the input tableau as a scipy matrix.
## The additional parameter 'direction' determines directions of edges, but at this moment, we do not use this parameter.
## This version of make_matrix_from_T is new, which means that the matrix has 4 types of edges
## (DASHED_ARROW, DASHED_ARROW, DOUBLE_ARROW, and TRIPLE_ARROW).
## TRIPLE_ARROW is for indicating maximal P-paths.
def make_matrix_from_T_v2(P, word, direction=(Direction.FORWARD, Direction.FORWARD, Direction.FORWARD)):
    n = len(word)
    row = []
    col = []
    edge_type = []

    col_index = [1]
    for i in range(1, n):
        if is_P_less(P, word[i], word[i - 1]):
            col_index.append(col_index[-1])
        else:
            col_index.append(col_index[-1] + 1)
    for i in range(n):
        for j in range(i + 1, n):
            if not is_P_compatible(P, word[i], word[j]):
                if direction[0] == Direction.FORWARD or direction[0] == Direction.BOTH:
                    row.append(word[i] - 1)
                    col.append(word[j] - 1)
                    edge_type.append(EDGE_TYPE.DASHED_ARROW)
                if direction[0] == Direction.BACKWARD or direction[0] == Direction.BOTH:
                    row.append(word[j] - 1)
                    col.append(word[i] - 1)
                    edge_type.append(EDGE_TYPE.DASHED_ARROW)
            elif col_index[i] == col_index[j]:
                if direction[1] == Direction.FORWARD or direction[1] == Direction.BOTH:
                    row.append(word[j] - 1)
                    col.append(word[i] - 1)
                    edge_type.append(EDGE_TYPE.SINGLE_ARROW)
                if direction[1] == Direction.BACKWARD or direction[1] == Direction.BOTH:
                    row.append(word[i] - 1)
                    col.append(word[j] - 1)
                    edge_type.append(EDGE_TYPE.SINGLE_ARROW)
            else:
                if direction[2] == Direction.FORWARD or direction[2] == Direction.BOTH:
                    row.append(min(word[i], word[j]) - 1)
                    col.append(max(word[i], word[j]) - 1)
                    edge_type.append(EDGE_TYPE.DOUBLE_ARROW)
                if direction[2] == Direction.BACKWARD or direction[2] == Direction.BOTH:
                    row.append(max(word[i], word[j]) - 1)
                    col.append(min(word[i], word[j]) - 1)
                    edge_type.append(EDGE_TYPE.DOUBLE_ARROW)
    for a in range(1, n + 1):
        for c in range(a + 1, n + 1):
            if not is_P_less(P, a, c): continue
            chk = False
            for b in range(a + 1, c):
                if not is_P_less(P, a, b) and not is_P_less(P, b, c):
                    chk = True
                    break
            if chk == True:
                row.append(a - 1)
                col.append(c - 1)
                edge_type.append(EDGE_TYPE.TRIPLE_ARROW)
    return sp.coo_matrix((edge_type, (row, col)), shape=(n, n))


#                             labels.append(mult-pre_calculated[str(lamb)])

## Generate training data
## shape_checkers determines shapes of tableaux in the generated data.
##      If we want to make data consisting of only tableaux of 2 row shape or hook shape, then set shape_checkers=[is_2row, is_hook]
## good_checker determine which tableau is 'GOOD' or 'BAD'
##      Many checkers we found determine only whether the tableau is 'BAD' or not.
##      So, good_checker has to have three types of return values ('GOOD', 'BAD', 'UNKNOWN')
##      I implemented some good_checkers below this function.
## Three parameters (primitive, connected, UPTO_N) may be not important, and I recommend to set (True, False, False), respectively.


# def generate_data_PTabs(DIR_PATH,
#                         input_N,
#                         shape_checkers,
#                         good_1row_checker=is_good_P_1row_B,
#                         primitive=True,
#                         connected=False,
#                         UPTO_N=False,
#                         json_path="src/data/json/",
#                         column_info='original'):
#     with open(os.path.join(json_path, "Partitions.json")) as f:
#         Partitions = json.load(f)
#     with open(os.path.join(json_path, "PartitionIndex.json")) as f:
#         PartitionIndex = json.load(f)
#     with open(os.path.join(json_path, "TransitionMatrix.json")) as f:
#         TM = json.load(f)

#     if UPTO_N:
#         N = 1
#     else:
#         N = input_N
#     graphs = []
#     labels = []
#     graph_sizes = []
#     while N <= input_N:
#         n_str = str(N)
#         TM_n = np.matrix(TM[n_str])
#         for P in generate_UIO(N, connected=connected):
#             word_list = []
#             if primitive:
#                 iter_words = iter_shuffles(cluster_vertices(P))
#             else:
#                 iter_words = itertools.permutations(range(1, N + 1))

#             for word in iter_words:
#                 word = list(word)
#                 if word in word_list: continue
#                 words = words_from_orbit(P, word)
#                 word_list.extend(words)

#                 gs = dict()
#                 pre_calculated = dict()
#                 Fs = []
#                 for lamb in Partitions[n_str]:
#                     gs[str(lamb)] = sp.coo_matrix(([], ([], [])), shape=(0, 0), dtype=np.int16)
#                     pre_calculated[str(lamb)] = 0
#                     Fs.append(0)
#                 for word in words:
#                     shape = shape_of_word(P, word)
#                     D = P_Des(P, word)
#                     if D in Partitions[n_str]: Fs[Partitions[n_str].index(D)] += 1
#                     if shape == None: continue
#                     if all(shape_checker(shape) == False for shape_checker in shape_checkers): continue
#                     if column_info == "original":
#                         g = make_matrix_from_T(P, word)
#                     elif column_info == "column_direction":
#                         g = make_matrix_from_T_col_info(P, word)
#                     elif column_info == "column_direc_column_same":
#                         g = make_matrix_from_T_col_info(P, word,
#                                                         direction=(
#                                                             Direction.FORWARD, Direction.BOTH, Direction.FORWARD))
#                     chk = check_all_row_connected(P, word)
#                     if chk == 'UNKNOWN':
#                         gs[str(shape)] = sp.block_diag((gs[str(shape)], g))
#                     else:
#                         graphs.append(g)
#                         if chk == 'BAD':
#                             labels.append(0)
#                             graph_sizes.append(N)
#                         elif chk == 'GOOD':
#                             labels.append(1)
#                             graph_sizes.append(N)
#                             pre_calculated[str(shape)] += 1
#                         else:
#                             print("SOMETHING GOES WRONG!")
#                             return
#                 for k, lamb in enumerate(Partitions[n_str]):
#                     if gs[str(lamb)].size == 0: continue
#                     for shape_checker in shape_checkers:
#                         if shape_checker(lamb) == True:
#                             mult = 0
#                             for i in range(len(Partitions[n_str])):
#                                 mult += TM[n_str][i][k] * Fs[i]
#                             graphs.append(gs[str(lamb)])
#                             labels.append(mult - pre_calculated[str(lamb)])
#                             graph_sizes.append(N)
#                             if mult < pre_calculated[str(lamb)]:
#                                 print("mult < pre_calculated!!")
#                                 print(P, word, lamb, mult, pre_calculated[str(lamb)])
#                                 return
#                             break
#         N += 1
#     indices = np.arange(len(graphs))
# #     np.random.shuffle(indices)
#     shuffled_labels = [int(labels[indices[i]]) for i in range(len(graphs))]
#     shuffled_graph_sizes = [int(graph_sizes[indices[i]]) for i in range(len(graphs))]

#     for i in range(len(indices)):
#         file_path = os.path.join(DIR_PATH, f"graph_{i:05d}.npz")
#         sp.save_npz(file_path, graphs[indices[i]])
#     with open(os.path.join(DIR_PATH, f"labels.json"), 'w') as f:
#         json.dump(shuffled_labels, f)
#     with open(os.path.join(DIR_PATH, f"graph_sizes.json"), 'w') as f:
#         json.dump(shuffled_graph_sizes, f)
        
# def generate_data_PTabs_position_of_one(DIR_PATH,
#                         input_N,
#                         shape_checkers,
#                         primitive = True,
#                         connected = False,
#                         UPTO_N = False,
#                         json_path = "src/data/json/",
#                         column_info = 'original'):
#     with open(os.path.join(json_path, "Partitions.json")) as f:
#         Partitions = json.load(f)
#     with open(os.path.join(json_path, "PartitionIndex.json")) as f:
#         PartitionIndex = json.load(f)
#     with open(os.path.join(json_path, "TransitionMatrix_btw_s_h.json")) as f:
#         TM = json.load(f)
    
#     if UPTO_N:
#         N = 1
#     else:
#         N = input_N
#     graphs = []
#     labels = []
#     graph_sizes = []
#     while N <= input_N:
#         n_str = str(N)
#         for P in generate_UIO(N, connected=connected):
#             for words in orbits_from_P(P, primitive):
#                 gs = dict()
#                 s_vec = dict()
#                 for k in range(1, N+1):
#                     s_vec[k] = dict()
#                     gs[k] = dict()
#                     for lamb in Partitions[n_str]:
#                         lamb_str = str(lamb)
#                         gs[k][lamb_str] = sp.coo_matrix(([], ([], [])), shape=(0,0), dtype=np.int16)
#                         s_vec[k][lamb_str] = 0
#                 for word in words:
#                     shape = shape_of_word(P, word)
#                     if shape == None: continue
#                     T = PTab_from_word(P, word)
#                     k = T[0].index(1) + 1
#                     s_vec[k][str(shape)] += 1
#                     if all(shape_checker(shape) == False for shape_checker in shape_checkers): continue
#                     if column_info == "original":
#                         g = make_matrix_from_T(P, word)
#                     elif column_info == "column_direction":
#                         g = make_matrix_from_T_col_info(P, word)
#                     elif column_info == "column_direc_column_same":
#                         g = make_matrix_from_T_col_info(P, word,
#                                                         direction=(Direction.FORWARD, Direction.BOTH, Direction.FORWARD))
                    
#                     gs[k][str(shape)] = sp.block_diag((gs[k][str(shape)], g))
                
#                 for k in s_vec.keys():
#                     for lamb in Partitions[n_str]:
#                         lamb_str = str(lamb)
#                         if gs[k][lamb_str].size == 0: continue
#                         for shape_checker in shape_checkers:
#                             if shape_checker(lamb) == True:
#                                 h_coeff = 0
#                                 for mu in Partitions[n_str]:
#                                     mu_str = str(mu)
#                                     h_coeff += TM[n_str][mu_str][lamb_str] * s_vec[k][mu_str]
#                                 if h_coeff < 0:
#                                     print("It is not positive!!!!")
#                                     print(P, word, k, lamb, h_coeff)
#                                     raise Exception(f"{P}, {word}, {k}, {lamb}, {h_coeff}: not positivie")
#                                 graphs.append(gs[k][lamb_str])
#                                 labels.append(h_coeff)
#                                 graph_sizes.append(N)
#                                 break
#         N += 1
#     indices = np.arange(len(graphs))
#     np.random.shuffle(indices)
#     shuffled_labels = [int(labels[indices[i]]) for i in range(len(graphs))]
#     shuffled_graph_sizes = [int(graph_sizes[indices[i]]) for i in range(len(graphs))]
    
#     for i in range(len(indices)):
#         file_path = os.path.join(DIR_PATH, f"graph_{i:05d}.npz")
#         sp.save_npz(file_path, graphs[indices[i]])
#     with open(os.path.join(DIR_PATH, f"labels.json"), 'w') as f:
#         json.dump(shuffled_labels, f)
#     with open(os.path.join(DIR_PATH, f"graph_sizes.json"), 'w') as f:
#         json.dump(shuffled_graph_sizes, f)


# def generate_data_PTabs_decomposed_via_first_entry(DIR_PATH,
#                         input_N,
#                         shape_checkers = [any_shape],
#                         filter = trivial_criterion,
#                         primitive = True,
#                         connected = False,
#                         UPTO_N = False,
#                         json_path = "src/data/json/",
#                         column_info = 'original'):
#     with open(os.path.join(json_path, "Partitions.json")) as f:
#         Partitions = json.load(f)
#     with open(os.path.join(json_path, "PartitionIndex.json")) as f:
#         PartitionIndex = json.load(f)
#     with open(os.path.join(json_path, "TransitionMatrix_btw_s_h.json")) as f:
#         TM = json.load(f)
    
#     if UPTO_N:
#         N = 1
#     else:
#         N = input_N
#     graphs = []
#     labels = []
#     graph_sizes = []
#     while N <= input_N:
#         n_str = str(N)
#         Nchoose2 = N * (N-1) / 2
#         for P in generate_UIO(N, connected=connected):
#             for words in orbits_from_P(P, primitive):
#                 gs = dict()
#                 pre_calculated = dict()
#                 s_vec = dict()
#                 for k in range(1, N+1):
#                     s_vec[k] = dict()
#                     gs[k] = dict()
#                     pre_calculated[k] = dict()
#                     for lamb in Partitions[n_str]:
#                         lamb_str = str(lamb)
#                         gs[k][lamb_str] = sp.coo_matrix(([], ([], [])), shape=(0,0), dtype=np.int16)
#                         s_vec[k][lamb_str] = 0
#                         pre_calculated[k][lamb_str] = 0
#                 for word in words:
#                     shape = shape_of_word(P, word)
#                     if shape == None: continue
#                     T = PTab_from_word(P, word)
#                     k = T[0][0]
#                     s_vec[k][str(shape)] += 1
#                     if all(shape_checker(shape) == False for shape_checker in shape_checkers): continue
#                     if column_info == "original":
#                         g = make_matrix_from_T(P, word)
#                     elif column_info == "column_direction":
#                         g = make_matrix_from_T_col_info(P, word)
#                     elif column_info == "column_direc_column_same":
#                         g = make_matrix_from_T_col_info(P, word,
#                                                         direction=(Direction.FORWARD, Direction.BOTH, Direction.FORWARD))
#                     filtered = filter(P, word)
#                     if filtered == 'UNKNOWN':
#                         gs[k][str(shape)] = sp.block_diag((gs[k][str(shape)], g))
#                     elif filtered == 'BAD':
#                         graphs.append(g)
#                         labels.append(0)
#                         graph_sizes.append(N)
#                     elif filtered == 'GOOD':
#                         graphs.append(g)
#                         labels.append(1)
#                         graph_sizes.append(N)
#                         pre_calculated[k][str(shape)] += 1
#                     else:
#                         print("SOMETHING GOES WRONG!")
#                         print(P, word, k, lamb, h_coeff)
#                         raise Exception(f"{P}, {word}, {k}, {lamb}, {h_coeff}: the filter makes an error")
                
#                 for k in s_vec.keys():
#                     for lamb in Partitions[n_str]:
#                         lamb_str = str(lamb)
#                         if gs[k][lamb_str].size == 0 and pre_calculated[k][lamb_str] == 0: continue
#                         for shape_checker in shape_checkers:
#                             if shape_checker(lamb) == True:
#                                 h_coeff = 0
#                                 for mu in Partitions[n_str]:
#                                     mu_str = str(mu)
#                                     h_coeff += TM[n_str][mu_str][lamb_str] * s_vec[k][mu_str]
#                                 if h_coeff < 0:
#                                     print("It is not positive!!!!")
#                                     print(P, word, k, lamb, h_coeff)
#                                     raise Exception(f"{P}, {word}, {k}, {lamb}, {h_coeff}: not positive")
#                                 if int(gs[k][lamb_str].size / Nchoose2) < h_coeff - pre_calculated[k][lamb_str]:
#                                     print("The filter is not a valid filter!!!!")
#                                     print(P, word, k, lamb, h_coeff)
#                                     raise Exception(f"{P}, {word}, {k}, {lamb}, {h_coeff}: not valid filter")
#                                 graphs.append(gs[k][lamb_str])
#                                 labels.append(h_coeff-pre_calculated[k][lamb_str])
#                                 graph_sizes.append(N)
#                                 break
#         N += 1
#     indices = np.arange(len(graphs))
# #     np.random.shuffle(indices)
#     shuffled_labels = [int(labels[indices[i]]) for i in range(len(graphs))]
#     shuffled_graph_sizes = [int(graph_sizes[indices[i]]) for i in range(len(graphs))]
    
#     for i in range(len(indices)):
#         file_path = os.path.join(DIR_PATH, f"graph_{i:05d}.npz")
#         sp.save_npz(file_path, graphs[indices[i]])
#     with open(os.path.join(DIR_PATH, f"labels.json"), 'w') as f:
#         json.dump(shuffled_labels, f)
#     with open(os.path.join(DIR_PATH, f"graph_sizes.json"), 'w') as f:
#         json.dump(shuffled_graph_sizes, f)

# def generate_data_PTabs_ppath(DIR_PATH,
#                               input_N,
#                               shape_checkers,
#                               good_checker,
#                               primitive=True,
#                               connected=False,
#                               UPTO_N=False,
#                               json_path="src/data/json/", ):
#     with open(os.path.join(json_path, "Partitions.json")) as f:
#         Partitions = json.load(f)
#     with open(os.path.join(json_path, "PartitionIndex.json")) as f:
#         PartitionIndex = json.load(f)
#     with open(os.path.join(json_path, "TransitionMatrix.json")) as f:
#         TM = json.load(f)

#     if UPTO_N:
#         N = 1
#     else:
#         N = input_N
#     graphs = []
#     labels = []
#     while N <= input_N:
#         n_str = str(N)
#         TM_n = np.matrix(TM[n_str])
#         for P in generate_UIO(N, connected=connected):
#             word_list = []
#             if primitive:
#                 iter_words = iter_shuffles(cluster_vertices(P))
#             else:
#                 iter_words = itertools.permutations(range(1, N + 1))

#             for word in iter_words:
#                 word = list(word)
#                 if word in word_list: continue
#                 words = words_from_orbit(P, word)
#                 word_list.extend(words)

#                 gs = dict()
#                 pre_calculated = dict()
#                 Fs = []
#                 for lamb in Partitions[n_str]:
#                     gs[str(lamb)] = sp.coo_matrix(([], ([], [])), shape=(0, 0), dtype=np.int16)
#                     pre_calculated[str(lamb)] = 0
#                     Fs.append(0)
#                 for word in words:
#                     shape = shape_of_word(P, word)
#                     D = P_Des(P, word)
#                     if D in Partitions[n_str]: Fs[Partitions[n_str].index(D)] += 1
#                     if shape == None: continue
#                     if all(shape_checker(shape) == False for shape_checker in shape_checkers): continue
#                     g = make_matrix_from_T_v2(P, word)
#                     chk = good_checker(P, word)
#                     if chk == 'UNKNOWN':
#                         gs[str(shape)] = sp.block_diag((gs[str(shape)], g))
#                     else:
#                         graphs.append(g)
#                         if chk == 'BAD':
#                             labels.append(0)
#                         elif chk == 'GOOD':
#                             labels.append(1)
#                             pre_calculated[str(shape)] += 1
#                         else:
#                             print("SOMETHING GOES WRONG!")
#                             return
#                 for k, lamb in enumerate(Partitions[n_str]):
#                     if gs[str(lamb)].size == 0: continue
#                     for shape_checker in shape_checkers:
#                         if shape_checker(lamb) == True:
#                             mult = 0
#                             for i in range(len(Partitions[n_str])):
#                                 mult += TM[n_str][i][k] * Fs[i]
#                             graphs.append(gs[str(lamb)])
#                             labels.append(mult - pre_calculated[str(lamb)])
#                             if mult < pre_calculated[str(lamb)]:
#                                 print("mult < pre_calculated!!")
#                                 print(P, word, lamb, mult, pre_calculated[str(lamb)])
#                                 return
#                             break
#         N += 1
#     indices = np.arange(len(graphs))
#     np.random.shuffle(indices)
#     shuffled_labels = [int(labels[indices[i]]) for i in range(len(graphs))]

#     for i in range(len(indices)):
#         file_path = os.path.join(DIR_PATH, f"graph_{i:05d}.npz")
#         sp.save_npz(file_path, graphs[indices[i]])
#     with open(os.path.join(DIR_PATH, f"labels.json"), 'w') as f:
#         json.dump(shuffled_labels, f)


########################################
############## criterions ##############
########################################


