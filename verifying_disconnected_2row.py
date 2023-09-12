from BH.data_loader import *
from BH.generate_data import *
from predictor_info import *
from predictor import *

from itertools import permutations as Perm

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
            pred = predict_tableau(P, word, MODEL)
            if (pred > cutoff and is_good_P_1row_B(P, perm1) and is_good_P_1row_B(P, perm2)) or (pred < 1-cutoff and not (is_good_P_1row_B(P, perm1) and is_good_P_1row_B(P, perm2))):
              continue
            print(f"{P} {word} {pred:.6lf} {is_good_P_1row_B(P, perm1) and is_good_P_1row_B(P, perm2)}")
