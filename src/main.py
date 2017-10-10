from __future__ import print_function, division
from spatial_temporal_autoencoder import SpatialTemporalAutoencoder
from data_iterator import DataIterator
import ConfigParser
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import logging
from plots import plot_loss, plot_auc, plot_regularity


def compute_eer(far, frr):
    cords = zip(far, frr)
    min_dist = 999999
    for item in cords:
        item_far, item_frr = item
        dist = abs(item_far - item_frr)
        if dist < min_dist:
            min_dist = dist
            eer = (item_far + item_frr) / 2
    return eer


def train(data, net):
    best_auc = -float('inf')
    best_reg_scores = None
    aucs = []
    eers = []
    losses = []
    print_every = 20
    auc_every = 50
    # around 180 iterations exhausts whole training data once
    for i in xrange(1, NUM_ITER + 1):
        tr_batch = data.get_train_batch()
        net.step(tr_batch, is_training=True)
        losses.append(net.get_loss(tr_batch, is_training=False))
        if i % print_every == 0:
            logging.info("average training reconstruction loss over {0:d} iterations: {1:g}"
                         .format(print_every, np.mean(losses[-print_every:])))
        if i % auc_every == 0:
            reg, auc, eer = test(data, net)
            logging.info("area under the roc curve at iteration {0:d}: {1:g}"
                         .format(i, auc))
            aucs.append(auc)
            eers.append(eer)
            if auc > best_auc:
                best_reg_scores = reg
                best_auc = auc
                best_eer = eer
                net.save_model()
    plot_loss(iters=NUM_ITER, losses=losses)
    plot_auc(aucs=aucs)
    return best_reg_scores, best_auc, best_eer


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
    auc = roc_auc_score(data.get_test_labels(), abnorm_scores)
    fpr, tpr, thresholds = roc_curve(data.get_test_labels(), abnorm_scores, pos_label=1)
    eer = compute_eer(fpr, 1 - tpr)
    return reg_scores, auc, eer


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
    logging.basicConfig(filename="STAE.log", level=logging.INFO)

    d = DataIterator(P_TRAIN, P_TEST, P_LABELS, batch_size=BATCH_SIZE)
    stae = SpatialTemporalAutoencoder(alpha=ALPHA, batch_size=BATCH_SIZE, lambd=LAMBDA)

    regularity_scores, area_under_roc = train(d, stae)
    logging.info("Best area under the roc curve: {0:g}".format(area_under_roc))
    plot_regularity(regularity_scores, d.get_test_labels())

    np.save('../results/regularity_scores.npy', regularity_scores)
