from predictor import *

from itertools import permutations as Perm

def verify_1row(MODEL, cutoff=0.7, good_checker=is_good_P_1row_B):
  total_cnt = 0
  incorrect_cnt = 0
  N = int(MODEL.split('parameters_')[-1][0])
  for P in generate_UIO(N, connected=False):
    for perm in iter_shuffles(cluster_vertices(P)):
      word = list(perm)
      if shape_of_word(P, word) != [N]: continue
      total_cnt += 1
      pred = predict_tableau(P, word, MODEL)
      if (pred > cutoff and good_checker(P, word)) or (pred < 1-cutoff and not good_checker(P, word)):
        continue
      incorrect_cnt += 1
      # print(f"{P} {word} {pred:.6f} {good_checker(P, word)}")
  return (total_cnt-incorrect_cnt, total_cnt)

def verify_disconnected_2row(MODEL, cutoff=0.7, good_checker=is_good_P_1row_B):
  total_cnt = 0
  incorrect_cnt = 0
  N = int(MODEL.split('parameters_')[-1][0])
  for i in range(1, N):
    for P1 in generate_UIO(i, connected=True):
      for P2 in generate_UIO(N-i, connected=True):
        P = [a for a in P1] + [a+i for a in P2]
        for perm1 in iter_shuffles(cluster_vertices(P1)):
          perm1 = list(perm1)
          for perm2 in iter_shuffles(cluster_vertices(P2)):
            temp = list(perm2)
            perm2 = [a+i for a in temp]
            word = []
            if i <= N - i:
              for j in range(i):
                word.append(perm2[j])
                word.append(perm1[j])
              for j in range(i, N-i):
                word.append(perm2[j])
            else:
              for j in range(N-i):
                word.append(perm2[j])
                word.append(perm1[j])
              for j in range(N-i, i):
                word.append(perm1[j])
            shape = shape_of_word(P, word)
            if shape == None or len(shape) != 2: continue
            total_cnt += 1
            pred = predict_tableau(P, word, MODEL)
            if (pred > cutoff and good_checker(P, perm1) and good_checker(P, perm2)) or (pred < 1-cutoff and not (good_checker(P, perm1) and good_checker(P, perm2))):
              continue
            incorrect_cnt += 1
            # print(f"{P} {word} {pred:.6f} {good_checker(P, perm1) and good_checker(P, perm2)}")
  return (total_cnt-incorrect_cnt, total_cnt)


def verify_inclusion_criterion(MODEL, shape_checker=any_shape, gap=0.5, cutoff=0.7):
    total_cnt = 0
    incorrect_cnt = 0
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
                total_cnt += 1
                if Ps[P1][1] == 'GOOD' and Ps[P2][1] == 'BAD': incorrect_cnt += 1
    return (total_cnt-incorrect_cnt, total_cnt)

def is_included(P1, P2):
    for i in range(len(P1)):
        if P1[i] > P2[i]:
            return False
    return True

def verify_disconnected_bad_criterion(MODEL, shape_checker=any_shape, cutoff = 0.7):
  total_cnt = 0
  incorrect_cnt = 0
  N = int(MODEL.split('parameters_')[-1][0])
  for P in generate_UIO(N, connected=False):
    if is_connected(P): continue
    components = split_into_connected_components(P)
    index = index_set_from_connected_components(components)
    for perm in iter_shuffles(cluster_vertices(P)):
      word = list(perm)
      shape = shape_of_word(P, word)
      if shape == None: continue
      if shape_checker(shape) == False: continue
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
        total_cnt += 1
        pred = predict_tableau(P, word, MODEL)
        if pred >= 1-cutoff:
          incorrect_cnt += 1
  return (total_cnt-incorrect_cnt, total_cnt)

def verify_disconnectedness_criterion(MODEL, shape_checker=any_shape, cutoff=0.7):
  total_cnt = 0
  incorrect_cnt = 0
  N = int(MODEL.split('parameters_')[-1][0])
  for P in generate_UIO(N, connected=False):
    components = split_into_connected_components(P)
    index = index_set_from_connected_components(components)
    for perm in iter_shuffles(cluster_vertices(P)):
      word = list(perm)
      shape = shape_of_word(P, word)
      if shape == None: continue
      if shape_checker(shape) == False: continue
      result = check_disconnectedness_criterion(P, word, components, index)
      if result == 'UNKNOWN': continue
      total_cnt += 1
      pred = predict_tableau(P, word, MODEL)
      if (result == 'GOOD' and pred <= cutoff) or (result == 'BAD' and pred >= 1-cutoff):
        incorrect_cnt += 1
  return (total_cnt-incorrect_cnt, total_cnt)

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

def verify_bad_2row_criterion(MODEL, cutoff=0.7):
  total_cnt = 0
  incorrect_cnt = 0
  N = int(MODEL.split('parameters_')[-1][0])
  for P in generate_UIO(N, connected=False):
    for perm in iter_shuffles(cluster_vertices(P)):
      word = list(perm)
      shape = shape_of_word(P, word)
      if shape == None: continue
      if is_2row(shape) == False: continue
      result = is_bad_P_2row(P, word)
      if result == 'UNKNOWN': continue
      total_cnt += 1
      pred = predict_tableau(P, word, MODEL)
      if (result == 'GOOD' and pred <= cutoff) or (result == 'BAD' and pred >= 1-cutoff):
        incorrect_cnt += 1
  return (total_cnt-incorrect_cnt, total_cnt)

def is_bad_P_2row(P, word):
    shape = shape_of_word(P, word)
    word1 = []
    word2 = []
    word3 = []
    for i in range(shape[1]):
        word2.append(word[i*2])
        word1.append(word[i*2+1])
    for i in range(shape[1]*2, len(word)):
        word3.append(word[i])
    if is_good_P_1row_B(P, word1+word3) and is_good_P_1row_B(P, word2):
      return 'UNKNOWN'
    if is_good_P_1row_B(P, word1) and is_good_P_1row_B(P, word2+word3):
      return 'UNKNOWN'
    return 'BAD'