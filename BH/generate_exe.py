from data_loader import *
from generate_data import *
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
use_ppath = False
N = 8

#directions = {"FORWARD": (Direction.FORWARD, 'F'),
#             "BACKWARD": (Direction.BACKWARD, 'B'),
#             "BOTH": (Direction.BOTH, '2'),
#             }

shapes = {
#         "2row_less": (is_2row_less, "2row_less"),
#          "3row_less": (is_3row_less, "3row_less"),
          "all": (any_shape, "all"),
#           "hook": (is_hook, "hook"),
#           "3col_less": (is_3col_less, "3col_less"),
         }
filters ={
#    "with_all_row_connectedness_criterion":(check_all_row_connected,"with_all_row_connectedness_criterion"), 
    'with_inductive_connectedness_criterion':(check_inductive_disconnectedness_criterion,'with_inductive_connectedness_criterion'),
}
connectedness = {
#                "connected": (True, "connected"),
                 "disconnected": (False, "disconnected")
                }


mkdir_order = "!mkdir /Data/Ptab/"
for shape in shapes.keys():
    for connect in connectedness.keys():
        for filter in filters.keys():
            DIR_PATH = f"/Data/Ptab/n={N}_{shapes[shape][1]}_{filters[filter][1]}_{connectedness[connect][1]}"
            os.makedirs(DIR_PATH, exist_ok=True)
            if use_ppath:
                DIR_PATH += f"_ppath"
                print(DIR_PATH)
                os.makedirs(DIR_PATH, exist_ok=True)
                generate_data_PTabs_v8(DIR_PATH,
                                       N,
                                       [shapes[shape][0]],
                                       filters[filter][0],
                                       primitive=True,
                                       connected=connectedness[connect][0]
                                      )
            else:
                print(DIR_PATH)
                os.makedirs(DIR_PATH, exist_ok=True)
                generate_data_PTabs_v7(DIR_PATH,
                                       N,
                                       [shapes[shape][0]],
                                       filters[filter][0],
                                       primitive=True,
                                       connected=connectedness[connect][0]
                                      )
