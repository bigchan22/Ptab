import os
from src.data.feature_functions import constant_feature,column_indicator, normalized_column_indicator, normalized_column_rev_indicator

GPU_NUM = "1"
num_epochs = 10
batch_size = 8192

# GCN_multi_stack = "sum"
GCN_multi_stack = "conv"

use_ppath = False
# column_info = "column_direction"
# column_info = "column_direc_column_same"
column_info = "original"
graph_deg = 8
# num_layers = graph_deg
num_layers = 4
num_features = 256
feature_list = {
    'constant': (True, constant_feature),
    'column': (False, column_indicator),
    'norm_column': (False, normalized_column_indicator),  # Feauture
    'norm_column_rev': (False, normalized_column_rev_indicator),
}
connected = True
UPTO = True

direction = "222"

shape_indicator = {
    'all_with_all_row_connectedness_criterion': (True,),
    'all_with_inductive_connectedness_criterion': (False,),
    '2row_less': (False,),
    '2row_less_with_all_row_connectedness_criterion': (False,),
    '2row_less_with_inductive_connectedness_criterion': (False,),
    '3col_less_with_all_row_connectedness_criterion': (False,),
    '3col_less_with_inductive_connectedness_criterion': (False,),
}

shape = {
    'all': (False,),  # 3개다
    '2row_less': (True,),
    '3row_less': (False,),
}
filter_indicator = {
    'with_all_row_connectedness_criterion': (False,),
    'with_inductive_connectedness_criterion': (False,),
}

use_pretrained_weights = False
save_trained_weights = True

step_size = 0.001
train_fraction = .8

DIR_PATH = f'Data/n={graph_deg}'
MODEL_DIR = 'models/trained_models'
MODEL_FILE = os.path.join(MODEL_DIR, f'parameters_{graph_deg}_{num_layers}_{num_features}')

for key in shape_indicator:
    if shape_indicator[key][0]:
        DIR_PATH += f'_{key}'
        MODEL_FILE += f'_{key}'

if connected:
    DIR_PATH += "_connected"
    MODEL_FILE += "_connected"
elif connected:
    DIR_PATH += "_disconnected"
    MODEL_FILE += "_disconnected"
if UPTO:
    DIR_PATH += "_UPTO"
    MODEL_FILE += "_UPTO"
if not column_info == "original":
    DIR_PATH += column_info
    MODEL_FILE += column_info
if use_ppath:
    DIR_PATH += "_ppath"
    MODEL_FILE += "_ppath"

for key in feature_list.keys():
    if feature_list[key][0]:
        MODEL_FILE += f'_{key}'
MODEL_FILE += f'_{direction}'
if GCN_multi_stack == "conv":
    MODEL_FILE += f'_{GCN_multi_stack}'
MODEL_FILE += '.pickle'
