from predictor import *

MODEL_LIST = load_models(show=False,keywords=['7_7', 'disconnected', '64'], cutoff=0.991)
MODELS = [MODEL for MODEL in MODEL_LIST if (not 'hook' in MODEL) and (not '3col' in MODEL) and (not 'rev' in MODEL)]

print("Model list")
for MODEL in MODELS:
    print(MODEL.split('parameters_')[-1].split('.')[0])
print("========================================================================")
for MODEL in MODELS:
    result = check_inclusion_criterion(MODEL, shape_checker=is_2row_less)
    print(result, MODEL)