from BH.data_loader import *
from BH.generate_data import *
from predictor_info import *
from predictor import *

from itertools import permutations as Perm

def verify_1row(N, MODEL, cutoff=0.7, good_checker=is_good_P_1row_B):
  for P in generate_UIO(N, connected=False):
    for perm in Perm([i+1 for i in range(N)]):
      word = list(perm)
      if shape_of_word(P, word) != [N]: continue
      pred = predict_tableau(P, word, MODEL)
      if (pred > cutoff and good_checker(P, word)) or (pred < 1-cutoff and not good_checker(P, word)):
        continue
      print(f"{P} {word} {pred:.6lf} {good_checker(P, word)}")

def verify_disconnected_2row(N, MODEL, cutoff=0.7):
  for i in range(1, N):
    print(f"{i}...")
    for P1 in generate_UIO(i, connected=True):
      for P2 in generate_UIO(N-i, connected=True):
        P = [a for a in P1] + [a+i for a in P2]
        for perm1 in Perm([a+1 for a in range(i)]):
          for perm2 in Perm([a+1 for a in range(i, N)]):
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
            pred = predict_tableau(P, word, MODEL)
            if (pred > cutoff and is_good_P_1row_B(P, perm1) and is_good_P_1row_B(P, perm2)) or (pred < 1-cutoff and not (is_good_P_1row_B(P, perm1) and is_good_P_1row_B(P, perm2))):
              continue
            print(f"{P} {word} {pred:.6lf} {is_good_P_1row_B(P, perm1) and is_good_P_1row_B(P, perm2)}")
