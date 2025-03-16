from src.data import shape_of_word, concatenating, has_lr_P_max, P_Des, has_rl_P_min, is_P_compatible


def is_good_P_1row_B(P, word):
    return not has_rl_P_min(P, word) and P_Des(P, word) == [len(word)]


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
    for i in range(len(P) - 1):
        if P[i] == i + 1:
            return False
    return True


def split_into_connected_components(P):
    components = [[]]
    for i in range(len(P) - 1):
        components[-1].append(i + 1)
        if P[i] == i + 1:
            components.append([])
    components[-1].append(len(P))
    return components


def index_set_from_connected_components(components):
    N = max(max(component) for component in components)
    index = [-1 for i in range(N + 1)]
    for i, component in enumerate(components):
        for k in component:
            index[k] = i
    return index


def conjugate(lamb):
    conj = []
    for i in range(1, lamb[0] + 1):
        cnt = 0
        for part in lamb:
            if part >= i:
                cnt += 1
        conj.append(cnt)
    return conj


def is_non_increasing(seq):
    for i in range(1, len(seq)):
        if seq[i - 1] < seq[i]:
            return False
    return True


def check_bad_2row_criterion(P, word, good_1row_checker=is_good_P_1row_B):
    shape = shape_of_word(P, word)
    word1 = []
    word2 = []
    word3 = []
    for i in range(shape[1]):
        word2.append(word[i * 2])
        word1.append(word[i * 2 + 1])
    for i in range(shape[1] * 2, len(word)):
        word3.append(word[i])
    if good_1row_checker(P, word1 + word3) and good_1row_checker(P, word2):
        return 'UNKNOWN'
    if good_1row_checker(P, word1) and good_1row_checker(P, word2 + word3):
        return 'UNKNOWN'
    return 'BAD'


def check_inductive_disconnectedness_criterion(P, word):  ####'with_inductive_connectedness_criterion':(False,),
    shape = shape_of_word(P, word)
    conj = conjugate(shape)
    k = len(word)
    for c in reversed(range(shape[0])):
        k -= conj[c]
        res_P, res_word = restricted_P_word(P, word[k:])
        if check_disconnectedness_criterion_for_inductive_argument(res_P, res_word) == False:
            return 'BAD'
    return 'UNKNOWN'


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
    if is_good_P_1row_B(P, word1 + word3) and is_good_P_1row_B(P, word2): return 'UNKNOWN'
    if is_good_P_1row_B(P, word1) and is_good_P_1row_B(P, word2 + word3): return 'UNKNOWN'
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
            if P[sorted_word[i] - 1] < sorted_word[j]:
                break
            j += 1
        res_P.append(j)
    for i in range(n):
        res_word.append(sorted_word.index(word[i]) + 1)
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


def check_all_row_connected(P, word, direction='B'):  ###'with_all_row_connectedness_criterion':(False,),
    if direction == 'B':
        row_checker = is_good_P_1row_B
    elif direction == 'F':
        row_checker = is_good_P_1row_F
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
            shape_of_pieces.append(k + 1)
            for i in range(k + 1):
                pieces[i].append(T[i][prev:shape[k]])
            prev = shape[k]
    base_words = []
    for i in range(len(pieces)):
        base_words.append(list(pieces[i][0]))
    if concatenating(P, shape_of_pieces, list(range(shape_of_pieces[0])), 1, pieces, base_words, row_checker) == True:
        return 'UNKNOWN'
    return 'BAD'


def is_good_P_1row_F(P, word):
    return not has_lr_P_max(P, word) and P_Des(P, word) == [len(word)]


def is_good_P_hook(P, word):
    sh = shape_of_word(P, word)
    arm = sh[0] - 1
    n = len(word)
    for i in range(n - arm):
        if is_good_P_1row_B(P, [word[i]] + word[n - arm:]):
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
