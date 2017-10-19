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


def train(data, net):
    best_auc = -float('inf')
    best_reg_scores = None
    aucs, eers, losses, valid_losses = [], [], [], []
    print_every, auc_every = 50, 50
    # around 180 iterations exhausts whole training data once
    for i in xrange(NUM_ITER + 1):
        tr_batch = data.get_train_batch()
        net.step(tr_batch, is_training=True)
        losses.append(net.get_loss(tr_batch, is_training=False))
        if i % print_every == 0:
            logging.info("average training reconstruction loss over {0:d} iterations: {1:g}"
                         .format(print_every, np.mean(losses[-print_every:])))
        if i % auc_every == 0:
            reg, auc, eer, valid_loss = test(data, net)
            logging.info("area under the roc curve at iteration {0:d}: {1:g}".format(i, auc))
            logging.info("validation loss at iteration {0:d}: {1:g}".format(i, valid_loss))
            aucs.append(auc)
            eers.append(eer)
            if auc > best_auc:
                best_reg_scores = reg
                best_auc = auc
                best_eer = eer
                net.save_model()
    plot_loss(losses=losses, valid_losses=valid_losses)
    plot_auc(aucs=aucs)
    plot_regularity(regularity_scores=best_reg_scores, labels=data.get_test_labels())
    np.save('../results/aucs.npy', aucs)
    np.save('../results/losses.npy', losses)
    np.save('../results/regularity_scores.npy', reg)
    return best_auc, best_eer


def test(data, net):
    data.reset_index()
    per_frame_error = [[] for _ in range(data.get_test_size())]
    while not data.check_data_exhausted():
        test_batch, frame_indices = data.get_test_batch()
        frame_error = net.get_recon_errors(test_batch, is_training=False)
        for i in xrange(frame_indices.shape[0]):
            for j in xrange(frame_indices.shape[1]):
                if frame_indices[i, j] != -1:
                    per_frame_error[frame_indices[i, j]].append(frame_error[i, j])

    per_frame_average_error = np.asarray(map(lambda x: np.mean(x), per_frame_error))
    # min-max normalize to linearly scale into [0, 1]
    abnorm_scores = (per_frame_average_error - per_frame_average_error.min()) / \
        (per_frame_average_error.max() - per_frame_average_error.min())
    reg_scores = 1 - abnorm_scores
    auc = roc_auc_score(y_true=data.get_test_labels(), y_score=abnorm_scores)
    valid_loss = np.mean(per_frame_average_error[data.get_test_labels() == 0])
    fpr, tpr, thresholds = roc_curve(y_true=data.get_test_labels(), y_score=abnorm_scores, pos_label=1)
    eer = compute_eer(far=fpr, frr=1 - tpr)
    return reg_scores, auc, eer, valid_loss


if __name__ == "__main__":
    Config = ConfigParser.ConfigParser()
    Config.read('../config/config.ini')
    NUM_ITER = int(Config.get("Default", "NUM_ITER"))
    ALPHA = float(Config.get("Default", "ALPHA"))
    LAMBDA = float(Config.get("Default", "LAMBDA"))
    BATCH_SIZE = int(Config.get("Default", "BATCH_SIZE"))
    P_TRAIN = Config.get("Default", "P_TRAIN")
    P_TEST = Config.get("Default", "P_TEST")
    P_LABELS = Config.get("Default", "P_LABELS")

    ts = time.time()
    dt = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    if not os.path.exists("logs"):
        os.makedirs("logs")
    logging.basicConfig(filename="logs/"+dt+".log", level=logging.INFO)

    d = DataIterator(P_TRAIN, P_TEST, P_LABELS, batch_size=BATCH_SIZE)
    stae = SpatialTemporalAutoencoder(alpha=ALPHA, batch_size=BATCH_SIZE, lambd=LAMBDA)

    area_under_roc, equal_error_rate = train(data=d, net=stae)
    logging.info("Best area under the roc curve: {0:g}".format(area_under_roc))
    logging.info("Equal error rate corresponding to best auc: {0:g}".format(equal_error_rate))
