import itertools
import json
import os
import copy

import numpy as np
import scipy.sparse as sp

from src.data.Data_gen_utils import generate_UIO, is_P_compatible, is_P_less, P_Des, words_from_orbit, shape_of_word, \
    iter_shuffles, cluster_vertices, EDGE_TYPE, Direction, orbits_from_P, PTab_from_word
from src.data.criterion import *
from src.data.shapes import *

def generate_counter_examples(Ns=[2,3,4,5,6,7,8],
                              good_filter="backward-weak-connectivity",
                              shape_filter="all",
                              necess_suff="n",
                              json_path="./src/data/json/"):
    shape_filters = {"all" : any_shape,
                    "1row" : is_1row,
                    "2row" : is_2row,
                    "3row" : is_3row,
                    "hook" : is_hook,
                    "2col" : is_2col,
                    "3col" : is_3col,
                    # "4col" : is_4col,
                    # "2row_le" : is_2row_less,
                    # "3row_le" : is_3row_less,
                    # "2col_le" : is_2col_less,
                    # "3col_le" : is_3col_less,
                    # "4col_le" : is_4col_less,
                    }
    good_filters = {"backward-weak-connectivity" : check_inductive_disconnectedness_criterion,
                    "forward-weak-connectivity" : check_inductive_disconnectedness_criterion_forward,
                    "backward-strong-connectivity" : check_all_row_connected,
                    "forward-strong-connectivity" : check_all_row_connected_forward,
                }

    if necess_suff in ["n", "necess", "necessary", "necessity", "necessary-condition", "necessary_condition", "necessary condition"]:
        necess_suff = "necessary-condition"
    elif necess_suff in ["s", "suff", "sufficiency", "sufficient", "sufficient-condition", "sufficient_condition", "sufficient condition"]:
        necess_suff = "sufficient-condition"
    else:
        necess_suff = "necessary-sufficient-condition"

    with open(os.path.join(json_path, "Partitions.json")) as f:
        Partitions = json.load(f)
    with open(os.path.join(json_path, "TransitionMatrix_btw_s_h.json")) as f:
        TM = json.load(f)

    for n in Ns:
        FILE_NAME = f"./counter-examples/counter-examples-{good_filter}-{n}.json"
        FILE_NAME = f"./counter-examples/{necess_suff}/counter-examples_shape={shape_filter}_good={good_filter}_n={n}.json"
        counter_examples = []
        
        n_str = str(n)
        part_dict = {}
        for lamb in Partitions[n_str]:
            part_dict[str(lamb)] = 0
        for P in generate_UIO(n, connected=True):
            for orbit in orbits_from_P(P, PtabOnly=True, primitive=True):
                coeffs1 = copy.copy(part_dict)
                coeffs2 = copy.copy(part_dict)
                for word in orbit:
                    shape = shape_of_word(P, word)
                    shape_str = str(shape)
                    coeffs1[shape_str] += 1
                    if good_filters[good_filter](P, word) not in ["BAD", False]:
                        coeffs2[shape_str] += 1
                for lamb in Partitions[n_str]:
                    if shape_filters[shape_filter](lamb) == False: continue
                    lamb_str = str(lamb)

                    coeff1 = 0
                    for mu in Partitions[n_str]:
                        mu_str = str(mu)
                        coeff1 += TM[n_str][mu_str][lamb_str] * coeffs1[mu_str]
                    coeff2 = coeffs2[lamb_str]
                    if coeff1 > coeff2 and necess_suff == "necessary-condition":
                        counter_examples.append(dict())
                        counter_examples[-1]["P"] = P
                        counter_examples[-1]["word"] = word
                        counter_examples[-1]["shape"] = lamb
                        counter_examples[-1]["coeff_of_lamb"] = coeff1
                        counter_examples[-1]["number_of_good"] = coeff2
                    elif coeff1 < coeff2 and necess_suff == "sufficient-condition":
                        counter_examples.append(dict())
                        counter_examples[-1]["P"] = P
                        counter_examples[-1]["word"] = word
                        counter_examples[-1]["shape"] = lamb
                        counter_examples[-1]["coeff_of_lamb"] = coeff1
                        counter_examples[-1]["number_of_good"] = coeff2
                    elif coeff1 != coeff2 and necess_suff == "necessary-sufficient-condition":
                        counter_examples.append(dict())
                        counter_examples[-1]["P"] = P
                        counter_examples[-1]["word"] = word
                        counter_examples[-1]["shape"] = lamb
                        counter_examples[-1]["coeff_of_lamb"] = coeff1
                        counter_examples[-1]["number_of_good"] = coeff2

        with open(FILE_NAME, "w") as f:
            json.dump(counter_examples, f)
        print("{n} Done")