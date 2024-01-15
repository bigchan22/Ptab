# import json
# import numpy as np
# import networkx as nx
# import scipy.sparse as sp
# import os
# import imports

from .imports import *
from copy import deepcopy

def iter_UIO(n, connected=False):
    if n == 1:
        yield [1]
        return
    if connected == False:
        k = 1
    else:
        k = 2
    seq = [i+k for i in range(n)]
    seq[n-1] = n
    seq[0] -= 1
    while seq[0] < n:
        for i in range(n-1):
            if seq[i] < seq[i+1]:
                seq[i] += 1
                for j in range(i):
                    seq[j] = j+k
                break
        yield seq

def generate_UIO(n, connected=False):
    if connected == False: mode = 1
    else: mode = 2
    seq = [i+mode for i in range(n)]          
    seq[n-1] = n
    list_UIO = [list(seq)]
    while seq[0] < n:
        for i in range(n-1):
            if seq[i] < seq[i+1]:
                seq[i] += 1
                for j in range(i):
                    seq[j] = j+mode
                break
        list_UIO += [list(seq)]
    return list_UIO

def P_inv(P, word):
    inv = 0
    for i in range(1,len(word)):
        for j in range(i):
            if word[i] < word[j] and is_P_less(P, word[i], word[j]) == 0 and is_P_less(P, word[j], word[i]) == 0:
                inv += 1
    return inv

def is_P_compatible(P, a, b):
    if P[a-1] < b or P[b-1] < a:
        return True
    return False

def is_P_less(P, a, b):
    if P[a-1] < b:
        return True
    return False

def P_Des(P, word):     ## this function returns a composition
    prev = 0
    comp = []
    for i in range(1, len(word)):
        if is_P_less(P, word[i], word[i-1]):
            comp.append(i-prev)
            prev = i
    comp.append(len(word)-prev)
    return comp

def has_rl_P_min(P, word):
    for i in reversed(range(len(word)-1)):
        chk = 0
        for j in range(i+1, len(word)):
            if is_P_less(P, word[i], word[j]) == False:
                chk = 1
                break
        if chk == 0:
            return True
    return False

def has_rl_P_max(P, word):
    for i in reversed(range(len(word)-1)):
        chk = 0
        for j in range(i+1, len(word)):
            if is_P_less(P, word[j], word[i]) == False:
                chk = 1
                break
        if chk == 0:
            return True
    return False

def has_lr_P_max(P, word):
    for i in range(1,len(word)):
        chk = 0
        for j in range(i):
            if is_P_less(P, word[j], word[i]) == False:
                chk = 1
                break
        if chk == 0:
            return True
    return False

def words_no_des(P):
    words = []
    n = len(P)
    for word in itertools.permutations(range(1,n+1)):
        if P_Des(P, word) == [n]:
            words.append(list(word))
    return words

def words_from_heap(P, word):
    words = [list(word)]
    for word in words:
        for i in range(len(word)-1):
            if is_P_compatible(P, word[i], word[i+1]):
                temp = word[:i] + [word[i+1],word[i]] + word[i+2:]
                if not temp in words:
                    words.append(temp)
    return words

def words_from_orbit(P, word):
    words = [list(word)]
    for word in words:
        for i in range(len(word)-1):
            if is_P_compatible(P, word[i], word[i+1]):
                temp = word[:i] + [word[i+1],word[i]] + word[i+2:]
                if not temp in words:
                    words.append(temp)
        for i in range(1,len(word)-1):
            if word[i] < word[i-1] < word[i+1] and is_P_less(P, word[i], word[i+1]) and not is_P_less(P, word[i], word[i-1]) and not is_P_less(P, word[i-1], word[i+1]):
                temp = word[:i-1] + [word[i],word[i+1],word[i-1]] + word[i+2:]
                if not temp in words:
                    words.append(temp)
            if word[i] > word[i+1] > word[i-1] and is_P_less(P, word[i-1], word[i]) and not is_P_less(P, word[i-1], word[i+1]) and not is_P_less(P, word[i+1], word[i]):
                temp = word[:i-1] + [word[i+1],word[i-1],word[i]] + word[i+2:]
                if not temp in words:
                    words.append(temp)
    return words

def shape_of_word(P, word):
    shape = []
    n = len(word)
    k = 1
    while k < n and is_P_less(P, word[k], word[k-1]):
        k += 1
    shape.append(k)
    a = k
    while a < n:
        k += 1
        while k < n and is_P_less(P, word[k], word[k-1]):
            k += 1
        if shape[-1] < k - a: return None
        for i in range(1, k-a+1):
            if is_P_less(P, word[k-i], word[a-i]):
                return None
        shape.append(k-a)
        a = k
    conj_shape = []
    for i in range(shape[0]):
        cnt = 0
        for j in range(len(shape)):
            if shape[j] > i:
                cnt += 1
        conj_shape.append(cnt)
    return conj_shape

def is_1row(shape):
    if shape == None: return False
    if len(shape) == 1: return True
    return False

def is_2row(shape):
    if shape == None: return False
    if len(shape) == 2: return True
    return False

def is_2row_less(shape):
    if shape == None: return False
    if len(shape) <= 2: return True
    return False

def is_3row(shape):
    if shape == None: return False
    if len(shape) == 3: return True
    return False

def is_3row_less(shape):
    if shape == None: return False
    if len(shape) <= 3: return True
    return False

def is_hook(shape):
    if shape == None: return False
    if len(shape) == 1 or shape[1] == 1: return True
    return False

def is_2col(shape):
    if shape == None: return False
    if shape[0] == 2: return True
    return False

def is_2col_less(shape):
    if shape == None: return False
    if shape[0] <= 2: return True
    return False

def is_3col(shape):
    if shape == None: return False
    if shape[0] == 3: return True
    return False

def is_3col_less(shape):
    if shape == None: return False
    if shape[0] <= 3: return True
    return False

def is_4col(shape):
    if shape == None: return False
    if shape[0] == 4: return True
    return False

def is_4col_less(shape):
    if shape == None: return False
    if shape[0] <= 4: return True
    return False

def is_43(shape):
    if shape == None: return False
    if shape == [4,3]: return True
    return False

def is_52(shape):
    if shape == None: return False
    if shape == [5,2]: return True
    return False

def is_61(shape):
    if shape == None: return False
    if shape == [6,1]: return True
    return False

def is_511(shape):
    if shape == None: return False
    if shape == [5,1,1]: return True
    return False

def is_4111(shape):
    if shape == None: return False
    if shape == [4,1,1,1]: return True
    return False

def is_31111(shape):
    if shape == None: return False
    if shape == [3,1,1,1,1]: return True
    return False

def is_211111(shape):
    if shape == None: return False
    if shape == [2,1,1,1,1,1]: return True
    return False

def any_shape(shape):
    if shape == None: return False
    return True

def is_good_P_1row_F(P, word):
    return not has_lr_P_max(P, word) and P_Des(P, word) == [len(word)]

def is_good_P_1row_B(P, word):
    return not has_rl_P_min(P, word) and P_Des(P, word) == [len(word)]

def is_good_P_hook(P, word):
    sh = shape_of_word(P, word)
    arm = sh[0] - 1
    n = len(word)
    for i in range(n-arm):
        if is_good_P_1row_B(P, [word[i]]+word[n-arm:]):
            return True
    return False

def is_good_P_2col(P, word):
    sh = shape_of_word(P, word)
    ell = len(sh)
    r = 0
    for k in range(ell, len(word)):
        while r < ell and is_P_compatible(P, word[r], word[k]):
            r += 1
        if r == ell: return False
        r += 1
    return True

def comb_to_shuffle(comb, A, B):
    iterA = iter(A)
    iterB = iter(B)
    return [next(iterA) if i in comb else next(iterB) for i in range(len(A) + len(B))]

def iter_shuffles(lists):
    if len(lists) == 1:
        yield lists[0]
    elif len(lists) == 2:
        for comb in itertools.combinations(range(len(lists[0]) + len(lists[1])), len(lists[0])):
            yield comb_to_shuffle(comb, lists[0], lists[1])
    else:
        length_sum = sum(len(word) for word in lists)
        for comb in itertools.combinations(range(length_sum), len(lists[0])):
            for shuffled in iter_shuffles(lists[1:]):
                yield comb_to_shuffle(comb, lists[0], shuffled)

def cluster_vertices(P):
    n = len(P)
    arr = [0 for i in range(n)]
    k = 0
    for i in range(1,len(P)):
        if P[i-1] != P[i]:
            for j in range(P[i-1], P[i]):
                arr[j] += i
            k += 1
        arr[i] += k
    vertices = [[1]]
    for i in range(1, len(P)):
        if arr[i-1] == arr[i]:
            vertices[-1].append(i+1)
        else:
            vertices.append([i+1])
    return vertices

def make_matrix_from_T(P, word, direction=(Direction.FORWARD, Direction.FORWARD, Direction.FORWARD)):
    n = len(word)
    row = []
    col = []
    edge_type = []
    
    col_index = [1]
    for i in range(1, n):
        if is_P_less(P, word[i], word[i-1]):
            col_index.append(col_index[-1])
        else:
            col_index.append(col_index[-1]+1)
    for i in range(n):
        for j in range(i+1, n):
            if not is_P_compatible(P, word[i], word[j]):
                if direction[0] == Direction.FORWARD or direction[0] == Direction.BOTH:
                    row.append(word[i]-1)
                    col.append(word[j]-1)
                    edge_type.append(EDGE_TYPE.DASHED_ARROW)
                if direction[0] == Direction.BACKWARD or direction[0] == Direction.BOTH:
                    row.append(word[j]-1)
                    col.append(word[i]-1)
                    edge_type.append(EDGE_TYPE.DASHED_ARROW)
            elif col_index[i] == col_index[j]:
                if direction[1] == Direction.FORWARD or direction[1] == Direction.BOTH:
                    row.append(word[j]-1)
                    col.append(word[i]-1)
                    edge_type.append(EDGE_TYPE.SINGLE_ARROW)
                if direction[1] == Direction.BACKWARD or direction[1] == Direction.BOTH:
                    row.append(word[i]-1)
                    col.append(word[j]-1)
                    edge_type.append(EDGE_TYPE.SINGLE_ARROW)
            else:
                if direction[2] == Direction.FORWARD or direction[2] == Direction.BOTH:
                    row.append(min(word[i], word[j])-1)
                    col.append(max(word[i], word[j])-1)
                    edge_type.append(EDGE_TYPE.DOUBLE_ARROW)
                if direction[2] == Direction.BACKWARD or direction[2] == Direction.BOTH:
                    row.append(max(word[i], word[j])-1)
                    col.append(min(word[i], word[j])-1)
                    edge_type.append(EDGE_TYPE.DOUBLE_ARROW)
    return sp.coo_matrix((edge_type, (row,col)), shape=(n,n))

def generate_data_PTabs(DIR_PATH,
                        input_N,
                        checkers,
                        connected=False,
                        UPTO_N=False):
    if UPTO_N:
        N = 1
    else:
        N = input_N
    graphs = []
    labels = []
    while N <= input_N:
        for P in generate_UIO(N, connected=connected):
            for word in itertools.permutations(range(1,N+1)):
                word = list(word)
                shape = shape_of_word(P, word)
                if shape == None: continue
                for (shape_checker, good_checker) in checkers:
                    if shape_checker(shape):
                        graphs.append(make_matrix_from_T(P, word))
                        if good_checker(P, word): labels.append(1)
                        else: labels.append(0)
                        break
        N += 1
    indices = np.arange(len(graphs))
    np.random.shuffle(indices)
    shuffled_labels = [int(labels[indices[i]]) for i in range(len(graphs))]

    for i in range(len(indices)):
        file_path = os.path.join(DIR_PATH, f"graph_{i:05d}.npz")
        sp.save_npz(file_path, graphs[indices[i]])
    with open(os.path.join(DIR_PATH, f"labels.json"), 'w') as f:
        json.dump(shuffled_labels, f)

def generate_data_PTabs_v2(DIR_PATH,
                        input_N,
                        shape_checkers,
                        primitive = True,
                        connected = False,
                        UPTO_N = False,
                        json_path = "./json/",
                        direction = (Direction.FORWARD, Direction.FORWARD, Direction.FORWARD)):
    with open(os.path.join(json_path, "Partitions.json")) as f:
        Partitions = json.load(f)
    with open(os.path.join(json_path, "PartitionIndex.json")) as f:
        PartitionIndex = json.load(f)
    with open(os.path.join(json_path, "TransitionMatrix.json")) as f:
        TM = json.load(f)
        
    if UPTO_N:
        N = 1
    else:
        N = input_N
    graphs = []
    labels = []
    while N <= input_N:
        n_str = str(N)
        TM_n = np.matrix(TM[n_str])
        for P in generate_UIO(N, connected=connected):
            word_list = []
            if primitive:
                iter_words = iter_shuffles(cluster_vertices(P))
            else:
                iter_words = itertools.permutations(range(1,N+1))

            for word in iter_words:
                word = list(word)
                if word in word_list: continue
                words = words_from_orbit(P, word)
                word_list.extend(words)
                
                gs = dict()
                Fs = []
                for lamb in Partitions[n_str]:
                    gs[str(lamb)] = sp.coo_matrix(([], ([], [])), shape=(0,0), dtype=np.int16)
                    Fs.append(0)
                for word in words:
                    shape = shape_of_word(P, word)
                    D = P_Des(P, word)
                    if D in Partitions[n_str]: Fs[Partitions[n_str].index(D)] += 1
                    if shape == None: continue
                    shape = str(shape)
                    g = make_matrix_from_T(P, word, direction)
                    gs[shape] = sp.block_diag((gs[shape], g))
                for k, lamb in enumerate(Partitions[n_str]):
                    if gs[str(lamb)].size == 0: continue
                    for shape_checker in shape_checkers:
                        if shape_checker(lamb) == True:
                            mult = 0
                            for i in range(len(Partitions[n_str])):
                                mult += TM[n_str][i][k] * Fs[i]
                            graphs.append(gs[str(lamb)])
                            labels.append(mult)
                            break
        N += 1
    indices = np.arange(len(graphs))
    np.random.shuffle(indices)
    shuffled_labels = [int(labels[indices[i]]) for i in range(len(graphs))]

    for i in range(len(indices)):
        file_path = os.path.join(DIR_PATH, f"graph_{i:05d}.npz")
        sp.save_npz(file_path, graphs[indices[i]])
    with open(os.path.join(DIR_PATH, f"labels.json"), 'w') as f:
        json.dump(shuffled_labels, f)

def generate_data_PTabs_v3(DIR_PATH,
                        input_N,
                        good_shape_checkers,
                        other_shape_checkers,
                        primitive = True,
                        connected = False,
                        UPTO_N = False,
                        json_path = "./json/",):
    with open(os.path.join(json_path, "Partitions.json")) as f:
        Partitions = json.load(f)
    with open(os.path.join(json_path, "PartitionIndex.json")) as f:
        PartitionIndex = json.load(f)
    with open(os.path.join(json_path, "TransitionMatrix.json")) as f:
        TM = json.load(f)
        
    if UPTO_N:
        N = 1
    else:
        N = input_N
    graphs = []
    labels = []
    while N <= input_N:
        n_str = str(N)
        TM_n = np.matrix(TM[n_str])
        for P in generate_UIO(N, connected=connected):
            word_list = []
            if primitive:
                iter_words = iter_shuffles(cluster_vertices(P))
            else:
                iter_words = itertools.permutations(range(1,N+1))

            for word in iter_words:
                word = list(word)
                if word in word_list: continue
                words = words_from_orbit(P, word)
                word_list.extend(words)
                
                gs = dict()
                Fs = []
                for lamb in Partitions[n_str]:
                    gs[str(lamb)] = sp.coo_matrix(([], ([], [])), shape=(0,0), dtype=np.int16)
                    Fs.append(0)
                for word in words:
                    shape = shape_of_word(P, word)
                    D = P_Des(P, word)
                    if D in Partitions[n_str]: Fs[Partitions[n_str].index(D)] += 1
                    if shape == None: continue
                    g = make_matrix_from_T(P, word)
                    chk = False
                    for (shape_checker, good_checker) in good_shape_checkers:
                        if shape_checker(shape) == True:
                            graphs.append(g)
                            if good_checker(P, word) == True: labels.append(1)
                            else: labels.append(0)
                            chk = True
                            break
                    if chk == False: gs[str(shape)] = sp.block_diag((gs[str(shape)], g))
                for k, lamb in enumerate(Partitions[n_str]):
                    if gs[str(lamb)].size == 0: continue
                    for shape_checker in other_shape_checkers:
                        if shape_checker(lamb) == True:
                            mult = 0
                            for i in range(len(Partitions[n_str])):
                                mult += TM[n_str][i][k] * Fs[i]
                            graphs.append(gs[str(lamb)])
                            labels.append(mult)
                            break
        N += 1
    indices = np.arange(len(graphs))
    np.random.shuffle(indices)
    shuffled_labels = [int(labels[indices[i]]) for i in range(len(graphs))]

    for i in range(len(indices)):
        file_path = os.path.join(DIR_PATH, f"graph_{i:05d}.npz")
        sp.save_npz(file_path, graphs[indices[i]])
    with open(os.path.join(DIR_PATH, f"labels.json"), 'w') as f:
        json.dump(shuffled_labels, f)

def generate_data_PTabs_v4(DIR_PATH,
                        input_N,
                        shape_checkers,
                        good_1row_checker=is_good_P_1row_B,
                        primitive = True,
                        connected = False,
                        UPTO_N = False,
                        json_path = "./json/",):
    with open(os.path.join(json_path, "Partitions.json")) as f:
        Partitions = json.load(f)
    with open(os.path.join(json_path, "PartitionIndex.json")) as f:
        PartitionIndex = json.load(f)
    with open(os.path.join(json_path, "TransitionMatrix.json")) as f:
        TM = json.load(f)
        
    if UPTO_N:
        N = 1
    else:
        N = input_N
    graphs = []
    labels = []
    while N <= input_N:
        n_str = str(N)
        TM_n = np.matrix(TM[n_str])
        for P in generate_UIO(N, connected=connected):
            components = split_into_connected_components(P)
            index = index_set_from_connected_components(components)
            word_list = []
            if primitive:
                iter_words = iter_shuffles(cluster_vertices(P))
            else:
                iter_words = itertools.permutations(range(1,N+1))

            for word in iter_words:
                word = list(word)
                if word in word_list: continue
                words = words_from_orbit(P, word)
                word_list.extend(words)
                
                gs = dict()
                pre_calculated = dict()
                Fs = []
                for lamb in Partitions[n_str]:
                    gs[str(lamb)] = sp.coo_matrix(([], ([], [])), shape=(0,0), dtype=np.int16)
                    pre_calculated[str(lamb)] = 0
                    Fs.append(0)
                for word in words:
                    shape = shape_of_word(P, word)
                    D = P_Des(P, word)
                    if D in Partitions[n_str]: Fs[Partitions[n_str].index(D)] += 1
                    if shape == None: continue
                    if all(shape_checker(shape) == False for shape_checker in shape_checkers): continue
                    g = make_matrix_from_T(P, word)
                    chk = check_disconnectedness_criterion(P, word, components, index, good_1row_checker)
                    if chk == 'UNKNOWN': gs[str(shape)] = sp.block_diag((gs[str(shape)], g))
                    else:
                        graphs.append(g)
                        if chk == 'BAD': labels.append(0)
                        elif chk == 'GOOD':
                            labels.append(1)
                            pre_calculated[str(shape)] += 1
                        else:
                            print("SOMETHING GOES WRONG!")
                            return
                for k, lamb in enumerate(Partitions[n_str]):
                    if gs[str(lamb)].size == 0: continue
                    for shape_checker in shape_checkers:
                        if shape_checker(lamb) == True:
                            mult = 0
                            for i in range(len(Partitions[n_str])):
                                mult += TM[n_str][i][k] * Fs[i]
                            graphs.append(gs[str(lamb)])
                            labels.append(mult-pre_calculated[str(lamb)])
                            if mult < pre_calculated[str(lamb)]:
                                print("mult < pre_calculated!!")
                                print(P, word, lamb, mult, pre_calculated[str(lamb)])
                                return
                            break
        N += 1
    indices = np.arange(len(graphs))
    np.random.shuffle(indices)
    shuffled_labels = [int(labels[indices[i]]) for i in range(len(graphs))]

    for i in range(len(indices)):
        file_path = os.path.join(DIR_PATH, f"graph_{i:05d}.npz")
        sp.save_npz(file_path, graphs[indices[i]])
    with open(os.path.join(DIR_PATH, f"labels.json"), 'w') as f:
        json.dump(shuffled_labels, f)

def generate_data_PTabs_v5(DIR_PATH,
                        input_N,
                        shape_checkers,
                        good_1row_checker=is_good_P_1row_B,
                        primitive = True,
                        connected = False,
                        UPTO_N = False,
                        json_path = "./json/",):
    with open(os.path.join(json_path, "Partitions.json")) as f:
        Partitions = json.load(f)
    with open(os.path.join(json_path, "PartitionIndex.json")) as f:
        PartitionIndex = json.load(f)
    with open(os.path.join(json_path, "TransitionMatrix.json")) as f:
        TM = json.load(f)
        
    if UPTO_N:
        N = 1
    else:
        N = input_N
    graphs = []
    labels = []
    while N <= input_N:
        n_str = str(N)
        TM_n = np.matrix(TM[n_str])
        for P in generate_UIO(N, connected=connected):
            components = split_into_connected_components(P)
            index = index_set_from_connected_components(components)
            word_list = []
            if primitive:
                iter_words = iter_shuffles(cluster_vertices(P))
            else:
                iter_words = itertools.permutations(range(1,N+1))

            for word in iter_words:
                word = list(word)
                if word in word_list: continue
                words = words_from_orbit(P, word)
                word_list.extend(words)
                
                gs = dict()
                pre_calculated = dict()
                Fs = []
                for lamb in Partitions[n_str]:
                    gs[str(lamb)] = sp.coo_matrix(([], ([], [])), shape=(0,0), dtype=np.int16)
                    pre_calculated[str(lamb)] = 0
                    Fs.append(0)
                for word in words:
                    shape = shape_of_word(P, word)
                    D = P_Des(P, word)
                    if D in Partitions[n_str]: Fs[Partitions[n_str].index(D)] += 1
                    if shape == None: continue
                    if all(shape_checker(shape) == False for shape_checker in shape_checkers): continue
                    g = make_matrix_from_T(P, word)
                    chk = check_disconnectedness_criterion(P, word, components, index, good_1row_checker)
                    if chk == 'UNKNOWN':
                        if len(components) == 1 and is_2row(shape):
                            chk = check_bad_2row_criterion(P, word, good_1row_checker)
                    if chk == 'UNKNOWN':
                        gs[str(shape)] = sp.block_diag((gs[str(shape)], g))
                    else:
                        graphs.append(g)
                        if chk == 'BAD': labels.append(0)
                        elif chk == 'GOOD':
                            labels.append(1)
                            pre_calculated[str(shape)] += 1
                        else:
                            print("SOMETHING GOES WRONG!")
                            return
                for k, lamb in enumerate(Partitions[n_str]):
                    if gs[str(lamb)].size == 0: continue
                    for shape_checker in shape_checkers:
                        if shape_checker(lamb) == True:
                            mult = 0
                            for i in range(len(Partitions[n_str])):
                                mult += TM[n_str][i][k] * Fs[i]
                            graphs.append(gs[str(lamb)])
                            labels.append(mult-pre_calculated[str(lamb)])
                            if mult < pre_calculated[str(lamb)]:
                                print("mult < pre_calculated!!")
                                print(P, word, lamb, mult, pre_calculated[str(lamb)])
                                return
                            break
        N += 1
    indices = np.arange(len(graphs))
    np.random.shuffle(indices)
    shuffled_labels = [int(labels[indices[i]]) for i in range(len(graphs))]

    for i in range(len(indices)):
        file_path = os.path.join(DIR_PATH, f"graph_{i:05d}.npz")
        sp.save_npz(file_path, graphs[indices[i]])
    with open(os.path.join(DIR_PATH, f"labels.json"), 'w') as f:
        json.dump(shuffled_labels, f)

def generate_data_PTabs_v6(DIR_PATH,
                        input_N,
                        shape_checkers,
                        good_1row_checker=is_good_P_1row_B,
                        primitive = True,
                        connected = False,
                        UPTO_N = False,
                        json_path = "./json/",):
    with open(os.path.join(json_path, "Partitions.json")) as f:
        Partitions = json.load(f)
    with open(os.path.join(json_path, "PartitionIndex.json")) as f:
        PartitionIndex = json.load(f)
    with open(os.path.join(json_path, "TransitionMatrix.json")) as f:
        TM = json.load(f)

    if UPTO_N:
        N = 1
    else:
        N = input_N
    graphs = []
    labels = []
    while N <= input_N:
        n_str = str(N)
        TM_n = np.matrix(TM[n_str])
        for P in generate_UIO(N, connected=connected):
            word_list = []
            if primitive:
                iter_words = iter_shuffles(cluster_vertices(P))
            else:
                iter_words = itertools.permutations(range(1,N+1))

            for word in iter_words:
                word = list(word)
                if word in word_list: continue
                words = words_from_orbit(P, word)
                word_list.extend(words)
                
                gs = dict()
                pre_calculated = dict()
                Fs = []
                for lamb in Partitions[n_str]:
                    gs[str(lamb)] = sp.coo_matrix(([], ([], [])), shape=(0,0), dtype=np.int16)
                    pre_calculated[str(lamb)] = 0
                    Fs.append(0)
                for word in words:
                    shape = shape_of_word(P, word)
                    D = P_Des(P, word)
                    if D in Partitions[n_str]: Fs[Partitions[n_str].index(D)] += 1
                    if shape == None: continue
                    if all(shape_checker(shape) == False for shape_checker in shape_checkers): continue
                    g = make_matrix_from_T(P, word)
                    chk = check_inductive_disconnectedness_criterion(P, word)
                    if chk == 'UNKNOWN':
                        gs[str(shape)] = sp.block_diag((gs[str(shape)], g))
                    else:
                        graphs.append(g)
                        if chk == 'BAD': labels.append(0)
                        elif chk == 'GOOD':
                            labels.append(1)
                            pre_calculated[str(shape)] += 1
                        else:
                            print("SOMETHING GOES WRONG!")
                            return
                for k, lamb in enumerate(Partitions[n_str]):
                    if gs[str(lamb)].size == 0: continue
                    for shape_checker in shape_checkers:
                        if shape_checker(lamb) == True:
                            mult = 0
                            for i in range(len(Partitions[n_str])):
                                mult += TM[n_str][i][k] * Fs[i]
                            graphs.append(gs[str(lamb)])
                            labels.append(mult-pre_calculated[str(lamb)])
                            if mult < pre_calculated[str(lamb)]:
                                print("mult < pre_calculated!!")
                                print(P, word, lamb, mult, pre_calculated[str(lamb)])
                                return
                            break
        N += 1
    indices = np.arange(len(graphs))
    np.random.shuffle(indices)
    shuffled_labels = [int(labels[indices[i]]) for i in range(len(graphs))]

    for i in range(len(indices)):
        file_path = os.path.join(DIR_PATH, f"graph_{i:05d}.npz")
        sp.save_npz(file_path, graphs[indices[i]])
    with open(os.path.join(DIR_PATH, f"labels.json"), 'w') as f:
        json.dump(shuffled_labels, f)

def generate_data_PTabs_v7(DIR_PATH,
                        input_N,
                        shape_checkers,
                        good_1row_checker=is_good_P_1row_B,
                        primitive = True,
                        connected = False,
                        UPTO_N = False,
                        json_path = "./json/",):
    with open(os.path.join(json_path, "Partitions.json")) as f:
        Partitions = json.load(f)
    with open(os.path.join(json_path, "PartitionIndex.json")) as f:
        PartitionIndex = json.load(f)
    with open(os.path.join(json_path, "TransitionMatrix.json")) as f:
        TM = json.load(f)

    if UPTO_N:
        N = 1
    else:
        N = input_N
    graphs = []
    labels = []
    while N <= input_N:
        n_str = str(N)
        TM_n = np.matrix(TM[n_str])
        for P in generate_UIO(N, connected=connected):
            word_list = []
            if primitive:
                iter_words = iter_shuffles(cluster_vertices(P))
            else:
                iter_words = itertools.permutations(range(1,N+1))

            for word in iter_words:
                word = list(word)
                if word in word_list: continue
                words = words_from_orbit(P, word)
                word_list.extend(words)
                
                gs = dict()
                pre_calculated = dict()
                Fs = []
                for lamb in Partitions[n_str]:
                    gs[str(lamb)] = sp.coo_matrix(([], ([], [])), shape=(0,0), dtype=np.int16)
                    pre_calculated[str(lamb)] = 0
                    Fs.append(0)
                for word in words:
                    shape = shape_of_word(P, word)
                    D = P_Des(P, word)
                    if D in Partitions[n_str]: Fs[Partitions[n_str].index(D)] += 1
                    if shape == None: continue
                    if all(shape_checker(shape) == False for shape_checker in shape_checkers): continue
                    g = make_matrix_from_T(P, word)
                    chk = check_all_row_connected(P, word)
                    if chk == 'UNKNOWN':
                        gs[str(shape)] = sp.block_diag((gs[str(shape)], g))
                    else:
                        graphs.append(g)
                        if chk == 'BAD': labels.append(0)
                        elif chk == 'GOOD':
                            labels.append(1)
                            pre_calculated[str(shape)] += 1
                        else:
                            print("SOMETHING GOES WRONG!")
                            return
                for k, lamb in enumerate(Partitions[n_str]):
                    if gs[str(lamb)].size == 0: continue
                    for shape_checker in shape_checkers:
                        if shape_checker(lamb) == True:
                            mult = 0
                            for i in range(len(Partitions[n_str])):
                                mult += TM[n_str][i][k] * Fs[i]
                            graphs.append(gs[str(lamb)])
                            labels.append(mult-pre_calculated[str(lamb)])
                            if mult < pre_calculated[str(lamb)]:
                                print("mult < pre_calculated!!")
                                print(P, word, lamb, mult, pre_calculated[str(lamb)])
                                return
                            break
        N += 1
    indices = np.arange(len(graphs))
    np.random.shuffle(indices)
    shuffled_labels = [int(labels[indices[i]]) for i in range(len(graphs))]

    for i in range(len(indices)):
        file_path = os.path.join(DIR_PATH, f"graph_{i:05d}.npz")
        sp.save_npz(file_path, graphs[indices[i]])
    with open(os.path.join(DIR_PATH, f"labels.json"), 'w') as f:
        json.dump(shuffled_labels, f)



########################################
############## criterions ##############
########################################

def check_disconnectedness_criterion(P, word, components, index, good_1row_checker=is_good_P_1row_B):
    shape = shape_of_word(P, word)
    conj = conjugate(shape)
    cnts = [[] for comp in components]
    k = 0
    for i in range(len(conj)):
        for cnt in cnts: cnt.append(0)
        for j in range(conj[i]):
            cnts[index[word[k]]][-1] += 1
            k += 1
    chk = True
    for cnt in cnts:
        if is_non_increasing(cnt) == False:
            chk = False
            break
    if chk == False:
        return 'BAD'
    for cnt in cnts:
        if cnt[0] != 1:
            return 'UNKNOWN'
    splitted_words = [[] for comp in components]
    for w in word:
        splitted_words[index[w]].append(w)
    if all(good_1row_checker(P, w) for w in splitted_words):
        return 'GOOD'
    return 'BAD'

def is_connected(P):
    for i in range(len(P)-1):
        if P[i] == i+1:
            return False
    return True

def split_into_connected_components(P):
    components = [[]]
    for i in range(len(P)-1):
        components[-1].append(i+1)
        if P[i] == i+1:
            components.append([])
    components[-1].append(len(P))
    return components

def index_set_from_connected_components(components):
    N = max(max(component) for component in components)
    index = [-1 for i in range(N+1)]
    for i, component in enumerate(components):
        for k in component:
            index[k] = i
    return index

def conjugate(lamb):
    conj = []
    for i in range(1, lamb[0]+1):
        cnt = 0
        for part in lamb:
            if part >= i:
                cnt += 1
        conj.append(cnt)
    return conj

def is_non_increasing(seq):
    for i in range(1, len(seq)):
        if seq[i-1] < seq[i]:
            return False
    return True

def check_bad_2row_criterion(P, word, good_1row_checker=is_good_P_1row_B):
    shape = shape_of_word(P, word)
    word1 = []
    word2 = []
    word3 = []
    for i in range(shape[1]):
        word2.append(word[i*2])
        word1.append(word[i*2+1])
    for i in range(shape[1]*2, len(word)):
        word3.append(word[i])
    if good_1row_checker(P, word1+word3) and good_1row_checker(P, word2):
        return 'UNKNOWN'
    if good_1row_checker(P, word1) and good_1row_checker(P, word2+word3):
        return 'UNKNOWN'
    return 'BAD'

def check_inductive_disconnectedness_criterion(P, word):
    shape = shape_of_word(P, word)
    conj = conjugate(shape)
    k = len(word)
    for c in reversed(range(shape[0])):
        k -= conj[c]
        res_P, res_word = restricted_P_word(P, word[k:])
        if check_disconnectedness_criterion_for_inductive_argument(res_P, res_word) == False:
            return "BAD"
    return "UNKNOWN"

def check_disconnectedness_criterion_for_inductive_argument(P, word):
    shape = shape_of_word(P, word)
    conj = conjugate(shape)

    components = split_into_connected_components(P)
    index = index_set_from_connected_components(components)

    cnts = [[] for comp in components]
    res_words = [[] for comp in components]

    k = 0
    for i in range(len(conj)):
        for cnt in cnts: cnt.append(0)
        for j in range(conj[i]):
            cnts[index[word[k]]][-1] += 1
            res_words[index[word[k]]].append(word[k])
            k += 1
    chk = True
    for cnt in cnts:
        if is_non_increasing(cnt) == False:
            return False
    for i in range(len(components)):
        res_P, res_word = restricted_P_word(P, res_words[i])
        res_shape = shape_of_word(res_P, res_word)
        if res_shape == None or conjugate(cnts[i]) != res_shape:
            return False
    return True

def check_2row_each_row_connected(P, word):
    shape = shape_of_word(P, word)
    T = PTab_from_word(P, word)
    word1 = list(T[0][:shape[1]])
    word2 = list(T[1])
    word3 = list(T[0][shape[1]:])
    if is_good_P_1row_B(P, word1+word3) and is_good_P_1row_B(P, word2): return 'UNKNOWN'
    if is_good_P_1row_B(P, word1) and is_good_P_1row_B(P, word2+word3): return 'UNKNOWN'
    return 'BAD'

def restricted_P_word(P, word):
    res_P = []
    res_word = []
    N = len(P)
    n = len(word)
    sorted_word = sorted(word)
    for i in range(n):
        j = i + 1
        while j < n:
            if P[sorted_word[i]-1] < sorted_word[j]:
                break
            j += 1
        res_P.append(j)
    for i in range(n):
        res_word.append(sorted_word.index(word[i])+1)
    return res_P, res_word

def PTab_from_word(P, word):
    shape = shape_of_word(P, word)
    T = [[] for row in shape]
    conj = conjugate(shape)
    k = 0
    for i in range(len(conj)):
        for j in reversed(range(conj[i])):
            T[j].append(word[k])
            k += 1
    return T

def check_all_row_connected(P, word, direction='B'):
    if direction == 'B': row_checker = is_good_P_1row_B
    elif direction == 'F': row_checker = is_good_P_1row_F
    else:
        print("Check the parameter for 'direction'")
        return

    T = PTab_from_word(P, word)
    shape = shape_of_word(P, word)
    shape_of_pieces = []
    pieces = [[] for i in range(len(shape))]
    prev = 0
    for k in reversed(range(len(shape))):
        if shape[k] > prev:
            shape_of_pieces.append(k+1)
            for i in range(k+1):
                pieces[i].append(T[i][prev:shape[k]])
            prev = shape[k]
    base_words = []
    for i in range(len(pieces)):
        base_words.append(list(pieces[i][0]))
    if concatenating(P, shape_of_pieces, list(range(shape_of_pieces[0])), 1, pieces, base_words, row_checker) == True:
        return 'UNKNOWN'
    return 'BAD'

def concatenating(P, shape_of_pieces, prev_block, k, pieces, prev_concatenated_words, good_1row_checker):
    if k == len(shape_of_pieces):
        for word in prev_concatenated_words:
            if good_1row_checker(P, word) == False:
                return False
        return True
    for block in itertools.combinations(prev_block, shape_of_pieces[k]):
        concatenated_words = deepcopy(prev_concatenated_words)
        for i, p in enumerate(block):
            concatenated_words[p].extend(pieces[i][k])
        if concatenating(P, shape_of_pieces, block, k+1, pieces, concatenated_words, good_1row_checker):
            return True