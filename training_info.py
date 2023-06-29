import os
# from feature_functions import *

graph_deg = 7
# partition_parts = [i for i in range(1,N+1)]
num_layers = 7
num_features = 64
num_epochs = 20
batch_size = 32
feature_list = {
    'constant': (True, ),
    'column':   (False, ),
}

use_pretrained_weights = True
save_trained_weights = True

step_size = 0.0005
train_fraction = .8
DIR_PATH = "/Data/Ptab/n=7_2row"
# DIR_PATH = '/root/Hwang/mathematics_conjectures'
# GRAPH_DIR = os.path.join(DIR_PATH, f'Hwang/Data/N_{N}')
MODEL_DIR = './trained_models'
# PARAM_DIR = os.path.join(DIR_PATH, 'Parameters')
# JSON_DIR = os.path.join(DIR_PATH, 'json')
# NUM_GRAPHS = len([f for f in os.listdir(GRAPH_DIR) if f.startswith("graph_")])
MODEL_FILE = os.path.join(MODEL_DIR, f'parameters_{graph_deg}_{num_layers}_{num_features}')
for key in feature_list.keys():
    if feature_list[key][0] == True:
        MODEL_FILE += f'_{key}'
MODEL_FILE += '.pickle'

