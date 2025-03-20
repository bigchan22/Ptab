import os

from src.data.feature_functions import *

graph_deg = 7
num_layers = 7
num_features = 64
feature_list = {
    'constant': (True, constant_feature),
    'column': (False,),
}

MODEL_DIR = 'models/trained_models'
# MODEL_FILE = os.path.join(MODEL_DIR, f'parameters_{graph_deg}_{num_layers}_{num_features}')
MODEL_FILE = os.path.join(MODEL_DIR,
                          'parameters_8_1_1024_all_with_inductive_connectedness_criterion_connected_UPTO_constant_222_conv.pickle')

# for key in feature_list.keys():
#    if feature_list[key][0] == True:
#        MODEL_FILE += f'_{key}'
# MODEL_FILE += '.pickle'
