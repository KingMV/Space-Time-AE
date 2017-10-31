from src.data_iterator import DataIterator
import ConfigParser
import logging
import time, datetime
import os
from comparison.model import STAE
from comparison.util import cal_per_frame_error
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from src.utils import compute_eer
from src.plots import plot_loss, plot_auc, plot_regularity
from src.train import train

if __name__ == "__main__":
    Config = ConfigParser.ConfigParser()
    Config.read('config/config.ini')
    num_iteration = int(Config.get("Default", "NUM_ITER"))
    batch_size = int(Config.get("Default", "BATCH_SIZE"))
    train_path = Config.get("Default", "P_TRAIN")
    test_path = Config.get("Default", "P_TEST")
    label_path = Config.get("Default", "P_LABELS")
    t_volume = int(Config.get("Default", "TVOL"))
    learning_rate = float(Config.get("Default", "ALPHA"))

    logging.basicConfig(filename=os.path.join("results", "STAE.log"), level=logging.INFO)

    # manually modify the batch_size
    batch_size=40
    data = DataIterator(train_path, test_path, label_path, batch_size=batch_size)
    data_shape = (t_volume, 224, 224, 1)
    logging.info('The learning rate is {} and batch_size is {}'.format(learning_rate, batch_size))
    stae = STAE(data_shape=data_shape, learning_rate=learning_rate, optimizer='adam')

    train(data, stae, num_iteration=num_iteration, result_path="results/")