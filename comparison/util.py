import numpy as np


def cal_per_frame_error(y_true, y_pred):
    if y_true.shape != y_pred.shape:
        raise ValueError('The shape of the predict frame shape is not matched with the grouth truth frame shape!')
    return np.mean(np.power(y_true - y_pred, 2), axis=(2, 3, 4))