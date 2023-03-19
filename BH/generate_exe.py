from .generate_data import *
from data_loader import *

DIR_PATH = "/Data/Ptab/n=5"
generate_data_PTabs(DIR_PATH, 5, ((is_1row, is_good_P_1row),
                                  (is_hook, is_good_P_hook),
                                  (is_2col, is_good_P_2col)))
# inputs = load_input_data(DIR_PATH)

# print(inputs[0][0])



