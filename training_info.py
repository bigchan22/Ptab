import os
from BH.feature_functions import *

GPU_NUM = "1"
num_epochs = 1000
batch_size = 64

use_ppath = False

graph_deg = 8
num_layers = graph_deg
num_features = 64
feature_list = {
    'constant':    (False, constant_feature),
    'column':      (False, column_indicator),
    'norm_column': (True, normalized_column_indicator), #   Feauture
    'norm_column_rev': (False, normalized_column_rev_indicator),
}
connected = False
UPTO = False

direction = "222"

shape_indicator = {
    'all_with_all_row_connectedness_criterion': (False, ),
    'all_with_inductive_connectedness_criterion': (True, ),
    '2row_less': (False, ),
    '2row_less_with_all_row_connectedness_criterion': (False, ),
    '2row_less_with_inductive_connectedness_criterion': (False, ),
    '3row_less_with_all_row_connectedness_criterion': (False, ),
    '3row_less_with_inductive_connectedness_criterion': (False, ),
}

shape= {
    'all': (False, ),# 3개다 
    '2row_less': (True, ),
    '3row_less': (False, ),
}
filter_indicator = {
    'with_all_row_connectedness_criterion':(False,), 
    'with_inductive_connectedness_criterion':(False,),
}


use_pretrained_weights = False
save_trained_weights = True

step_size = 0.001
train_fraction = .8

DIR_PATH = f'/Data/Ptab/n={graph_deg}'
MODEL_DIR = './trained_models'
MODEL_FILE = os.path.join(MODEL_DIR, f'parameters_{graph_deg}_{num_layers}_{num_features}')

for key in shape_indicator:
    if shape_indicator[key][0] == True:
        DIR_PATH += f'_{key}'
        MODEL_FILE += f'_{key}'


if connected == True:
    DIR_PATH += "_connected"
    MODEL_FILE += "_connected"
elif connected == False:
    DIR_PATH += "_disconnected"
    MODEL_FILE += "_disconnected"
if UPTO == True:
    DIR_PATH += "_UPTO"
    MODEL_FILE += "_UPTO"
if use_ppath:
    DIR_PATH += "_ppath"    
    MODEL_FILE += "_ppath"
    
for key in feature_list.keys():
    if feature_list[key][0] == True:
        MODEL_FILE += f'_{key}'
MODEL_FILE += f'_{direction}'
MODEL_FILE += '.pickle'

