import os
import shutil
import sys
import time

import yaml

# Compute the project root relative to this script (adjust ".." as needed)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)

# from src.data.generate_data import  generate_data_PTabs_ppath, generate_data_PTabs, generate_data_PTabs_position_of_one,generate_data_PTabs_decomposed_via_first_entry
from src.data.generate_data import generate_data_PTabs
from src.data.shapes import is_2row_less, is_3row_less, is_hook, is_3col_less, any_shape
from src.data.criterion import check_all_row_connected
from src.data.criterion import check_inductive_disconnectedness_criterion,check_inductive_disconnectedness_criterion_forward, trivial_criterion

print('start_time', time.strftime('%c'))
script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, "config_data_generation.yaml")
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

N = config["N"]
UPTO = config["UPTO"]
column_info = config["column_info"]
use_ppath = config["use_ppath"]
use_position_of_one = config["use_position_of_one"]
mode = config["generate_mode"]
data_dir = config["data_dir"]
shapes = config['shapes']
filters = config['filters']
connectedness = config['connectedness']
donotsave = config['donotsave']

SHAPES_MAP = {
    "2row_less": is_2row_less,
    "3row_less": is_3row_less,
    "all": any_shape,
    "hook": is_hook,
    "3col_less": is_3col_less,
}
SHAPES_FT = [SHAPES_MAP[shape] for shape in shapes]

FILTERS_MAP = {
    "with_all_row_connectedness_criterion": check_all_row_connected,
    'with_inductive_connectedness_criterion': check_inductive_disconnectedness_criterion,
    'with_inductive_connectedness_criterion_forward': check_inductive_disconnectedness_criterion_forward,
    'trivial_criterion':trivial_criterion
}
CONNECTED_MAP = {
    "connected": True,
    'disconnected': False
}
mkdir_order = "!mkdir /Data/Ptab/"
# Loop over the configuration values to generate data.
# for shape in config['shapes']:
for connect in config['connectedness']:
    filter = config['filters']
    # Build the directory path using the parameters.
    DIR_PATH = os.path.join(
        data_dir,
        f"n={N}_{shapes[0]}_{filter}_{connect}"
    )
    if UPTO:
        DIR_PATH += "_UPTO"
    if column_info != 'original':
        DIR_PATH += f"_{column_info}"
#             os.makedirs(DIR_PATH, exist_ok=True)
    DIR_PATH += f"{mode}"
    print("Generating in:", DIR_PATH)
    os.makedirs(DIR_PATH, exist_ok=True)
    generate_data_PTabs(
        DIR_PATH,
        N,
        SHAPES_FT,
        FILTERS_MAP[filter],
        primitive=True,
        connected=CONNECTED_MAP[connect],
        column_info=column_info,
        UPTO_N=UPTO,
        mode = mode,
        donotsave = donotsave
    )    
#         if mode == "test":
#             DIR_PATH += f"_test"
#             print("Generating in:", DIR_PATH)
#             os.makedirs(DIR_PATH, exist_ok=True)
#             generate_data_PTabs_v2(
#                 DIR_PATH,
#                 N,
#                 [SHAPES_MAP[shape]],
#                 FILTERS_MAP[filter],
#                 primitive=True,
#                 connected=CONNECTED_MAP[connect],
#                 column_info=column_info,
#                 UPTO_N=UPTO,
#                 mode = mode
#             )        
#         elif mode == "ppath":
#             DIR_PATH = DIR_PATH + "_ppath"
#             print("Generating in:", DIR_PATH_ppath)
#             os.makedirs(DIR_PATH, exist_ok=True)
#             generate_data_PTabs_ppath(
#                 DIR_PATH_ppath,
#                 N,
#                 [SHAPES_MAP[shape]],
#                 [FILTERS_MAP[filter]],
#                 primitive=True,
#                 connected=CONNECTED_MAP[connect],
#                 UPTO_N=UPTO
#             )
#         elif mode == "position_one":
#             DIR_PATH += f"_positionone"
#             print("Generating in:", DIR_PATH)
#             os.makedirs(DIR_PATH, exist_ok=True)
#             generate_data_PTabs_position_of_one(DIR_PATH,
#                N,
#                [SHAPES_MAP[shape]],

#                primitive=True,
#                connected=CONNECTED_MAP[connect],
#                column_info=column_info,
#                UPTO_N=UPTO
#             )
#         elif mode == "vanilla":
#             print("Generating in:", DIR_PATH)
#             os.makedirs(DIR_PATH, exist_ok=True)
#             generate_data_PTabs(
#                 DIR_PATH,
#                 N,
#                 [SHAPES_MAP[shape]],
#                 [FILTERS_MAP[filter]],
#                 primitive=True,
#                 connected=CONNECTED_MAP[connect],
#                 column_info=column_info,
#                 UPTO_N=UPTO
#             )
#         elif mode == "decomp":
#             DIR_PATH += f"_decomp"
#             print("Generating in:", DIR_PATH)
#             os.makedirs(DIR_PATH, exist_ok=True)
#             generate_data_PTabs_decomposed_via_first_entry(
#                 DIR_PATH,
#                 N,
#                 [SHAPES_MAP[shape]],
#                 FILTERS_MAP[filter],
#                 primitive=True,
#                 connected=CONNECTED_MAP[connect],
#                 column_info=column_info,
#                 UPTO_N=UPTO
#             )
#         else:
#             raise ValueError()

print('end_time', time.strftime('%c'))
shutil.copy(config_path, DIR_PATH)