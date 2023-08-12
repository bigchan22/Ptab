# import json
# import numpy as np
# import networkx as nx
# import scipy.sparse as sp
# import os
# import imports

from .imports import *

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
        if is_good_P_1row(P, [word[i]]+word[n-arm:]):
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

def make_matrix_from_T(P, word_of_T, direction=(Direction.FORWARD, Direction.FORWARD, Direction.FORWARD)):
    n = len(P)
    row = []
    col = []
    edge_type = []
    
    col_index = [1]
    for i in range(1, n):
        if is_P_less(P, word_of_T[i], word_of_T[i-1]):
            col_index.append(col_index[-1])
        else:
            col_index.append(col_index[-1]+1)
    for i in range(n):
        for j in range(i+1, n):
            if not is_P_compatible(P, word_of_T[i], word_of_T[j]):
                if direction[0] == Direction.FORWARD or direction[0] == Direction.BOTH:
                    row.append(word_of_T[i]-1)
                    col.append(word_of_T[j]-1)
                    edge_type.append(EDGE_TYPE.DASHED_ARROW)
                if direction[0] == Direction.BACKWARD or direction[0] == Direction.BOTH:
                    row.append(word_of_T[j]-1)
                    col.append(word_of_T[i]-1)
                    edge_type.append(EDGE_TYPE.DASHED_ARROW)
            elif col_index[i] == col_index[j]:
                if direction[1] == Direction.FORWARD or direction[1] == Direction.BOTH:
                    row.append(word_of_T[j]-1)
                    col.append(word_of_T[i]-1)
                    edge_type.append(EDGE_TYPE.SINGLE_ARROW)
                if direction[1] == Direction.BACKWARD or direction[1] == Direction.BOTH:
                    row.append(word_of_T[i]-1)
                    col.append(word_of_T[j]-1)
                    edge_type.append(EDGE_TYPE.SINGLE_ARROW)
            else:
                if direction[2] == Direction.FORWARD or direction[2] == Direction.BOTH:
                    row.append(min(word_of_T[i], word_of_T[j])-1)
                    col.append(max(word_of_T[i], word_of_T[j])-1)
                    edge_type.append(EDGE_TYPE.DOUBLE_ARROW)
                if direction[2] == Direction.BACKWARD or direction[2] == Direction.BOTH:
                    row.append(max(word_of_T[i], word_of_T[j])-1)
                    col.append(min(word_of_T[i], word_of_T[j])-1)
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
