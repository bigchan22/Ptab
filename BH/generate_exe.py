from data_loader import *
from generate_data import *
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

N = 6

directions = {"FORWARD": (Direction.FORWARD, 'F'),
             "BACKWARD": (Direction.BACKWARD, 'B'),
             "BOTH": (Direction.BOTH, '2'),
             }

shapes = {
          "2row_less": (is_2row_less, "2row_less"),
#           "3row_less": (is_3row_less, "3row_less"),
#           "all": (any_shape, "all"),
#           "hook": (is_hook, "hook"),
#           "3col_less": (is_3col_less, "3col_less"),
         }

connectedness = {
#                 "connected": (True, "connected"),
                 "disconnected": (False, "disconnected")
                }

mkdir_order = "!mkdir /Data/Ptab/"

for first_direction, fc in directions.items():
    for second_direction, sc in directions.items():
        for third_direction, tc in directions.items():
            for fourth_direction, tc in directions.items():
                for shape in shapes.keys():
                    for connect in connectedness.keys():
                        DIR_PATH = f"/Data/Ptab/n={N}_{shapes[shape][1]}_{fc[1]+sc[1]+tc[1]+fc[1]}_{connectedness[connect][1]}"
                        print(DIR_PATH)
                        generate_data_PTabs_v2(DIR_PATH,
                                               N,
                                               [shapes[shape][0]],
                                               primitive=True,
                                               connected=connectedness[connect][0],
                                               direction=(fc[0],
                                                          sc[0],
                                                          tc[0]))

