def is_1row(shape):
    if shape == None: return False
    if len(shape) == 1: return True
    return False


def is_2row(shape):
    if shape == None: return False
    if len(shape) == 2: return True
    return False


def is_2row_less(shape):
    if shape == None: return False
    if len(shape) <= 2: return True
    return False


def is_3row(shape):
    if shape == None: return False
    if len(shape) == 3: return True
    return False


def is_3row_less(shape):
    if shape == None: return False
    if len(shape) <= 3: return True
    return False


def is_hook(shape):
    if shape == None: return False
    if len(shape) == 1 or shape[1] == 1: return True
    return False


def is_2col(shape):
    if shape == None: return False
    if shape[0] == 2: return True
    return False


def is_2col_less(shape):
    if shape == None: return False
    if shape[0] <= 2: return True
    return False


def is_3col(shape):
    if shape == None: return False
    if shape[0] == 3: return True
    return False


def is_3col_less(shape):
    if shape == None: return False
    if shape[0] <= 3: return True
    return False


def is_4col(shape):
    if shape == None: return False
    if shape[0] == 4: return True
    return False


def is_4col_less(shape):
    if shape == None: return False
    if shape[0] <= 4: return True
    return False


def is_43(shape):
    if shape == None: return False
    if shape == [4, 3]: return True
    return False


def is_52(shape):
    if shape == None: return False
    if shape == [5, 2]: return True
    return False


def is_61(shape):
    if shape == None: return False
    if shape == [6, 1]: return True
    return False


def is_511(shape):
    if shape == None: return False
    if shape == [5, 1, 1]: return True
    return False


def is_4111(shape):
    if shape == None: return False
    if shape == [4, 1, 1, 1]: return True
    return False


def is_31111(shape):
    if shape == None: return False
    if shape == [3, 1, 1, 1, 1]: return True
    return False


def is_211111(shape):
    if shape == None: return False
    if shape == [2, 1, 1, 1, 1, 1]: return True
    return False


def any_shape(shape):
    if shape == None: return False
    return True
