from __future__ import print_function, division
from spatial_temporal_autoencoder import SpatialTemporalAutoencoder
from data_iterator import DataIterator
import ConfigParser
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import logging
from plots import plot_loss, plot_auc, plot_regularity
from utils import compute_eer
import os
import time
import datetime
from src.train import train


if __name__ == "__main__":
    Config = ConfigParser.ConfigParser()
    config_path = os.path.join("config", "config.ini")
    Config.read(config_path)
    NUM_ITER = int(Config.get("Default", "NUM_ITER"))
    ALPHA = float(Config.get("Default", "ALPHA"))
    LAMBDA = float(Config.get("Default", "LAMBDA"))
    BATCH_SIZE = int(Config.get("Default", "BATCH_SIZE"))
    P_TRAIN = Config.get("Default", "P_TRAIN")
    P_TEST = Config.get("Default", "P_TEST")
    P_LABELS = Config.get("Default", "P_LABELS")

    ts = time.time()
    dt = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    result_path = os.path.join("results", "archive", dt)
    os.makedirs(result_path)
    logging.basicConfig(filename=os.path.join(result_path, "STAE.log"), level=logging.INFO)

    d = DataIterator(P_TRAIN, P_TEST, P_LABELS, batch_size=BATCH_SIZE)
    stae = SpatialTemporalAutoencoder(alpha=ALPHA, batch_size=BATCH_SIZE, lambd=LAMBDA)

    area_under_roc, equal_error_rate = train(data=d, model=stae, num_iteration=NUM_ITER, result_path=result_path)
    logging.info("Best area under the roc curve: {0:g}".format(area_under_roc))
    logging.info("Equal error rate corresponding to best auc: {0:g}".format(equal_error_rate))
