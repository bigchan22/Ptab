import os
# from feature_functions import *

graph_deg = 7
num_layers = 7
num_features = 64
feature_list = {
    'constant': (True, ),
    'column':   (False, ),
}

MODEL_DIR = './trained_models'
MODEL_FILE = os.path.join(MODEL_DIR, f'parameters_{graph_deg}_{num_layers}_{num_features}')
for key in feature_list.keys():
    if feature_list[key][0] == True:
        MODEL_FILE += f'_{key}'
MODEL_FILE += '.pickle'

