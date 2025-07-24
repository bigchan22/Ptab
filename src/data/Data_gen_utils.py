import enum
import itertools
import json
import os
from copy import deepcopy
def conjugate(lamb):
    conj = []
    for i in range(1, lamb[0]+1):
        cnt = 0
        for part in lamb:
            if part >= i:
                cnt += 1
        conj.append(cnt)
    return conj

def iter_UIO(n, connected=False):
    if n == 1:
        yield [1]
        return
    if connected == False:
        k = 1
    else:
        k = 2
    seq = [i + k for i in range(n)]
    seq[n - 1] = n
    seq[0] -= 1
    while seq[0] < n:
        for i in range(n - 1):
            if seq[i] < seq[i + 1]:
                seq[i] += 1
                for j in range(i):
                    seq[j] = j + k
                break
        yield seq


def generate_UIO(n, connected=False):
    ## Generate Dyck paths (in our notation, we denote it by P)
    if connected == False:
        mode = 1
    else:
        mode = 2
    seq = [i + mode for i in range(n)]
    seq[n - 1] = n
    list_UIO = [list(seq)]
    while seq[0] < n:
        for i in range(n - 1):
            if seq[i] < seq[i + 1]:
                seq[i] += 1
                for j in range(i):
                    seq[j] = j + mode
                break
        list_UIO += [list(seq)]
    return list_UIO


def P_inv(P, word):
    inv = 0
    for i in range(1, len(word)):
        for j in range(i):
            if word[i] < word[j] and is_P_less(P, word[i], word[j]) == 0 and is_P_less(P, word[j], word[i]) == 0:
                inv += 1
    return inv


def is_P_compatible(P, a, b):
    ## Given a poset P, if either a <_P b or b <_P a, then return True
    if P[a - 1] < b or P[b - 1] < a:
        return True
    return False


def is_P_less(P, a, b):
    ## Given a poset P, if a <_P b, then return True
    if P[a - 1] < b:
        return True
    return False


def P_Des(P, word):
    ## Given a poset P, return the P-descent set of word as a composition
    ## e.g. P = [2,4,4,5,5], and word = 3142522, then the Descent set is {1,5}, hence return (1,4,2).
    prev = 0
    comp = []
    for i in range(1, len(word)):
        if is_P_less(P, word[i], word[i - 1]):
            comp.append(i - prev)
            prev = i
    comp.append(len(word) - prev)
    return comp


def has_rl_P_min(P, word):
    ## Given a poset P, if the input word has a nontrivial P minimum when we read it from right to left
    for i in reversed(range(len(word) - 1)):
        chk = 0
        for j in range(i + 1, len(word)):
            if is_P_less(P, word[i], word[j]) == False:
                chk = 1
                break
        if chk == 0:
            return True
    return False


def has_rl_P_max(P, word):
    ## Given a poset P, if the input word has a nontrivial P maximum when we read it from right to left
    for i in reversed(range(len(word) - 1)):
        chk = 0
        for j in range(i + 1, len(word)):
            if is_P_less(P, word[j], word[i]) == False:
                chk = 1
                break
        if chk == 0:
            return True
    return False


def has_lr_P_max(P, word):
    ## Given a poset P, if the input word has a nontrivial P maximum when we read it from left to right
    for i in range(1, len(word)):
        chk = 0
        for j in range(i):
            if is_P_less(P, word[j], word[i]) == False:
                chk = 1
                break
        if chk == 0:
            return True
    return False


def words_no_des(P):
    ## Given a poset P, return permutations with no P-Descent
    words = []
    n = len(P)
    for word in itertools.permutations(range(1, n + 1)):
        if P_Des(P, word) == [n]:
            words.append(list(word))
    return words


def words_from_heap(P, word):
    ## Given a poset P and a word, return words which are equivalent to the input word only using the relation ac = ca (if a <_P c)
    words = [list(word)]
    for word in words:
        for i in range(len(word) - 1):
            if is_P_compatible(P, word[i], word[i + 1]):
                temp = word[:i] + [word[i + 1], word[i]] + word[i + 2:]
                if not temp in words:
                    words.append(temp)
    return words

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

# def orbits_from_P(P, primitive=True):
#     orbs = []
#     word_list = []
#     if primitive == True:
#         words_domain = iter_shuffles(cluster_vertices(P))
#     else:
#         words_domain = Permutations(len(P))
#     for word in words_domain:
#         if word in word_list: continue
#         words = words_from_orbit(P, word)
#         orbs.append(words_from_orbit(P, word))
#         word_list.extend(words)
#         print(len(word_list))
#     return orbs
def orbits_from_P(P, PtabOnly = True, primitive=True):
    seen_words = set()
    orbs = []

    if primitive:
        words_domain = iter_shuffles(cluster_vertices(P))
    else:
        words_domain = Permutations(len(P))

    for word in words_domain:
        if PtabOnly and shape_of_word(P, word) is None:
            continue
        word_tuple = tuple(word)
        if word_tuple in seen_words:
            continue
        orbit = words_from_orbit(P, word, PtabOnly)
        for w in orbit:
            seen_words.add(tuple(w))
        orbs.append(orbit)
#         print(len(seen_words))
    return orbs
# def words_from_orbit(P, word):
#     ## Given a poset P and a word, return words which are equivalent to the input word w.r.t. Hwang's relations
#     words = [list(word)]
#     for word in words:
#         for i in range(len(word) - 1):
#             if is_P_compatible(P, word[i], word[i + 1]):
#                 temp = word[:i] + [word[i + 1], word[i]] + word[i + 2:]
#                 if not temp in words:
#                     words.append(temp)
#         for i in range(1, len(word) - 1):
#             if word[i] < word[i - 1] < word[i + 1] and is_P_less(P, word[i], word[i + 1]) and not is_P_less(P, word[i],
#                                                                                                             word[
#                                                                                                                 i - 1]) and not is_P_less(
#                 P, word[i - 1], word[i + 1]):
#                 temp = word[:i - 1] + [word[i], word[i + 1], word[i - 1]] + word[i + 2:]
#                 if not temp in words:
#                     words.append(temp)
#             if word[i] > word[i + 1] > word[i - 1] and is_P_less(P, word[i - 1], word[i]) and not is_P_less(P,
#                                                                                                             word[i - 1],
#                                                                                                             word[
#                                                                                                                 i + 1]) and not is_P_less(
#                 P, word[i + 1], word[i]):
#                 temp = word[:i - 1] + [word[i + 1], word[i - 1], word[i]] + word[i + 2:]
#                 if not temp in words:
#                     words.append(temp)
#     return words
#######################################################################################New version
def words_from_orbit(P, word,PtabOnly = True):
    """
    Given a poset P and a word, return the orbit under Hwang's relations.
    """
    from collections import deque

    seen = set()
    queue = deque()
    
    word_tuple = tuple(word)
    queue.append(word_tuple)
    seen.add(word_tuple)

    while queue:
        current = list(queue.popleft())
        
        # Local swaps
        for i in range(len(current) - 1):
            if is_P_compatible(P, current[i], current[i + 1]):
                temp = current[:i] + [current[i + 1], current[i]] + current[i + 2:]
                temp_tuple = tuple(temp)
                if temp_tuple not in seen:
                    seen.add(temp_tuple)
                    queue.append(temp_tuple)

        # Hwang's ternary rules
        for i in range(1, len(current) - 1):
            a, b, c = current[i - 1], current[i], current[i + 1]

            # Rule 1
            if b < a < c and is_P_less(P, b, c) and not is_P_less(P, b, a) and not is_P_less(P, a, c):
                temp = current[:i - 1] + [b, c, a] + current[i + 2:]
                temp_tuple = tuple(temp)
                if temp_tuple not in seen:
                    seen.add(temp_tuple)
                    queue.append(temp_tuple)

            # Rule 2
            if b > c > a and is_P_less(P, a, b) and not is_P_less(P, a, c) and not is_P_less(P, c, b):
                temp = current[:i - 1] + [c, a, b] + current[i + 2:]
                temp_tuple = tuple(temp)
                if temp_tuple not in seen:
                    seen.add(temp_tuple)
                    queue.append(temp_tuple)

#     return [list(w) for w in seen]
    if PtabOnly:
        return [list(w) for w in seen if shape_of_word(P, w) is not None]
    else:
        return [list(w) for w in seen]


def words_from_Blasiak_orbit(P, word):
    ## Given a poset P and a word, return words which are equivalent to the input word w.r.t. Blasiak's relations
    words = [list(word)]
    for word in words:
        for i in range(1, len(word) - 1):
            if is_P_less(P, word[i], word[i - 1]) and (not is_P_less(P, word[i + 1], word[i - 1])) and is_P_less(P,
                                                                                                                 word[
                                                                                                                     i],
                                                                                                                 word[
                                                                                                                     i + 1]):
                temp = word[:i - 1] + [word[i - 1], word[i + 1], word[i]] + word[i + 2:]
                if not temp in words:
                    words.append(temp)
            if is_P_less(P, word[i + 1], word[i - 1]) and (not is_P_less(P, word[i], word[i - 1])) and is_P_less(P,
                                                                                                                 word[
                                                                                                                     i + 1],
                                                                                                                 word[
                                                                                                                     i]):
                temp = word[:i - 1] + [word[i - 1], word[i + 1], word[i]] + word[i + 2:]
                if not temp in words:
                    words.append(temp)
            if (not is_P_less(P, word[i + 1], word[i - 1])) and is_P_less(P, word[i + 1], word[i]) and is_P_less(P,
                                                                                                                 word[
                                                                                                                     i - 1],
                                                                                                                 word[
                                                                                                                     i]):
                temp = word[:i - 1] + [word[i], word[i - 1], word[i + 1]] + word[i + 2:]
                if not temp in words:
                    words.append(temp)
            if (not is_P_less(P, word[i + 1], word[i])) and is_P_less(P, word[i + 1], word[i - 1]) and is_P_less(P,
                                                                                                                 word[
                                                                                                                     i],
                                                                                                                 word[
                                                                                                                     i - 1]):
                temp = word[:i - 1] + [word[i], word[i - 1], word[i + 1]] + word[i + 2:]
                if not temp in words:
                    words.append(temp)
            if not is_P_compatible(P, word[i], word[i + 1]) and not is_P_compatible(P, word[i + 1],
                                                                                    word[i - 1]) and is_P_less(P,
                                                                                                               word[i],
                                                                                                               word[
                                                                                                                   i - 1]):
                temp = word[:i - 1] + [word[i + 1], word[i - 1], word[i]] + word[i + 2:]
                if not temp in words:
                    words.append(temp)
            if not is_P_compatible(P, word[i + 1], word[i - 1]) and not is_P_compatible(P, word[i - 1],
                                                                                        word[i]) and is_P_less(P, word[
                i + 1], word[i]):
                temp = word[:i - 1] + [word[i], word[i + 1], word[i - 1]] + word[i + 2:]
                if not temp in words:
                    words.append(temp)
    return words


def words_from_Blasiak_variant1_orbit(P, word):
    words = [list(word)]
    for word in words:
        for i in range(1, len(word) - 1):
            if is_P_less(P, word[i], word[i - 1]) and (not is_P_less(P, word[i + 1], word[i - 1])) and is_P_less(P,
                                                                                                                 word[
                                                                                                                     i],
                                                                                                                 word[
                                                                                                                     i + 1]):
                temp = word[:i - 1] + [word[i - 1], word[i + 1], word[i]] + word[i + 2:]
                if not temp in words:
                    words.append(temp)
            if is_P_less(P, word[i + 1], word[i - 1]) and (not is_P_less(P, word[i], word[i - 1])) and is_P_less(P,
                                                                                                                 word[
                                                                                                                     i + 1],
                                                                                                                 word[
                                                                                                                     i]):
                temp = word[:i - 1] + [word[i - 1], word[i + 1], word[i]] + word[i + 2:]
                if not temp in words:
                    words.append(temp)
            if (not is_P_less(P, word[i + 1], word[i - 1])) and is_P_less(P, word[i + 1], word[i]) and is_P_less(P,
                                                                                                                 word[
                                                                                                                     i - 1],
                                                                                                                 word[
                                                                                                                     i]):
                temp = word[:i - 1] + [word[i], word[i - 1], word[i + 1]] + word[i + 2:]
                if not temp in words:
                    words.append(temp)
            if (not is_P_less(P, word[i + 1], word[i])) and is_P_less(P, word[i + 1], word[i - 1]) and is_P_less(P,
                                                                                                                 word[
                                                                                                                     i],
                                                                                                                 word[
                                                                                                                     i - 1]):
                temp = word[:i - 1] + [word[i], word[i - 1], word[i + 1]] + word[i + 2:]
                if not temp in words:
                    words.append(temp)
            if not is_P_compatible(P, word[i], word[i + 1]) and not is_P_compatible(P, word[i + 1],
                                                                                    word[i - 1]) and is_P_compatible(P,
                                                                                                                     word[
                                                                                                                         i - 1],
                                                                                                                     word[
                                                                                                                         i]):
                temp = word[:i - 1] + [word[i + 1], word[i - 1], word[i]] + word[i + 2:]
                if not temp in words:
                    words.append(temp)
            if not is_P_compatible(P, word[i + 1], word[i - 1]) and not is_P_compatible(P, word[i - 1],
                                                                                        word[i]) and is_P_compatible(P,
                                                                                                                     word[
                                                                                                                         i],
                                                                                                                     word[
                                                                                                                         i + 1]):
                temp = word[:i - 1] + [word[i], word[i + 1], word[i - 1]] + word[i + 2:]
                if not temp in words:
                    words.append(temp)
    return words


def shape_of_word(P, word):
    ## Given a poset P and a word, if the given word is a column word of some P-tableau, then return the shape of the tableau.
    ##                            Otherwise, return None
    shape = []
    n = len(word)
    k = 1
    while k < n and is_P_less(P, word[k], word[k - 1]):
        k += 1
    shape.append(k)
    a = k
    while a < n:
        k += 1
        while k < n and is_P_less(P, word[k], word[k - 1]):
            k += 1
        if shape[-1] < k - a: return None
        for i in range(1, k - a + 1):
            if is_P_less(P, word[k - i], word[a - i]):
                return None
        shape.append(k - a)
        a = k
    conj_shape = []
    for i in range(shape[0]):
        cnt = 0
        for j in range(len(shape)):
            if shape[j] > i:
                cnt += 1
        conj_shape.append(cnt)
    return conj_shape


def s_expansion(P, words):
    ## Given a poset P and a list of words, return the s-expansion of the quasisymmetric function \sum F_{Des_P(w)}
    ## The return value is of type dictionary, and its keys are of form str(partition), e.g., '[5, 2, 1]'
    ##  Warning: We assume that the quasisymmetric function from the input words is symmetric function.
    result = dict()
    for word in words:
        shape = shape_of_word(P, word)
        if shape == None: continue
        if str(shape) in result.items():
            result[str(shape)] += 1
        else:
            result[str(shape)] = 1
    return result


def h_expansion(P, words):
    ## Given a poset P and a list of words, return the h-expansion of the quasisymmetric function \sum F_{Des_P(w)}
    ## The return value is of type dictionary, and its keys are of form str(partition), e.g., '[5, 2, 1]'
    ##  Warning: We assume that the quasisymmetric function from the input words is symmetric function.
    ##           Under this assumption, we use the transition matrix between F-basis and h-basis. (TransitionMatrix.json)
    n_str = str(len(P))
    result = dict()
    json_path = "json/"
    with open(os.path.join(json_path, "Partitions.json")) as f:
        Partitions = json.load(f)
    with open(os.path.join(json_path, "PartitionIndex.json")) as f:
        PartitionIndex = json.load(f)
    with open(os.path.join(json_path, "TransitionMatrix.json")) as f:
        TM = json.load(f)

    Fs = [0 for lam in Partitions[n_str]]
    for word in words:
        D = P_Des(P, word)
        if D in Partitions[n_str]: Fs[Partitions[n_str].index(D)] += 1
    for k, lamb in enumerate(Partitions[n_str]):
        mult = 0
        for i in range(len(Partitions[n_str])):
            mult += TM[n_str][i][k] * Fs[i]
        if mult != 0: result[str(lamb)] = mult
    return result


def comb_to_shuffle(comb, A, B):
    ## The three functions below (comb_to_shuffle, iter_shuffles, cluster_vertices) reduce several equivalent data sets to a single one.
    # For example, if P=[2,4,4,4], the roles of the two letters 3 and 4 are exactly the same,
    # which means that, for instance, we do not distinguish between the words 1324 and 1423.
    # Hence, when we generate training data, we want to consider only words where 3 precedes 4.
    # Therefore, if we call cluster_vertices([2,4,4,4]), then it returns permutations where 3 precedes 4.
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
    for i in range(1, len(P)):
        if P[i - 1] != P[i]:
            for j in range(P[i - 1], P[i]):
                arr[j] += i
            k += 1
        arr[i] += k
    vertices = [[1]]
    for i in range(1, len(P)):
        if arr[i - 1] == arr[i]:
            vertices[-1].append(i + 1)
        else:
            vertices.append([i + 1])
    return vertices


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
        if concatenating(P, shape_of_pieces, block, k + 1, pieces, concatenated_words, good_1row_checker):
            return True


class EDGE_TYPE():
    SELF_LOOP = 1
    SINGLE_ARROW = 2
    DOUBLE_ARROW = 3
    DASHED_ARROW = 4
    TRIPLE_ARROW = 5


class Direction(enum.Enum):
    FORWARD = enum.auto()
    BACKWARD = enum.auto()
    BOTH = enum.auto()
    NONE = enum.auto()
