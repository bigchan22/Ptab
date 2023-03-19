import os
# from feature_functions import *

N = 5
# partition_parts = [i for i in range(1,N+1)]
num_layers = 3
num_features = 64
num_epochs = 300
batch_size = 64


use_pretrained_weights = True
save_trained_weights = True


step_size = 0.001
train_fraction = .8
DIR_PATH = "/Data/Ptab/n=5"
# DIR_PATH = '/root/Hwang/mathematics_conjectures'
# GRAPH_DIR = os.path.join(DIR_PATH, f'Hwang/Data/N_{N}')
PARAM_DIR = os.path.join(DIR_PATH, 'Parameters')
JSON_DIR = os.path.join(DIR_PATH, 'json')
# NUM_GRAPHS = len([f for f in os.listdir(GRAPH_DIR) if f.startswith("graph_")])
PARAM_FILE = os.path.join(PARAM_DIR, f'parameters_{N}_{num_layers}_{num_features}')
# for key in feature_list.keys():
#     PARAM_FILE += f'_{key}'
PARAM_FILE += '.pickle'

