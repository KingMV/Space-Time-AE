from __future__ import print_function, division
from spatial_temporal_autoencoder import SpatialTemporalAutoencoder
from data_iterator import DataIterator
import ConfigParser
import numpy as np
from sklearn.metrics import roc_auc_score
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def train(data, net):
    best_auc = -float('inf')
    best_reg_scores = None
    aucs = []
    losses = []
    print_every = 10
    auc_every = 1
    for i in xrange(1, NUM_ITER + 1):
        tr_batch = data.get_train_batch()
        net.step(tr_batch)
        losses.append(net.get_loss(tr_batch))
        if i % print_every == 0:
            logging.info("average training reconstruction loss over {0:d} iterations: {1:g}"
                         .format(print_every, np.mean(losses[-print_every:])))
        if i % auc_every == 0:
            reg, a_roc = test(data, net)
            aucs.append(a_roc)
            if a_roc > best_auc:
                best_reg_scores = reg
                best_auc = a_roc
                net.save_model()
    # plot loss vs. iteration number
    plt.figure()
    plt.plot(range(1, NUM_ITER + 1), losses)
    plt.xlabel("Iteration")
    plt.ylabel("Reconstruction loss")
    plt.savefig("../results/Loss.png")
    # plot auc vs. iteration number
    plt.figure()
    plt.plot(range(1, len(aucs) + 1), aucs)
    plt.xlabel("Iteration")
    plt.ylabel("Area under the roc curve")
    plt.savefig("../results/AUC.png")
    return best_reg_scores, best_auc


def test(data, net):
    per_frame_error = [[] for _ in range(data.get_test_size())]
    while not data.check_data_exhausted():
        test_batch, frame_indices = data.get_test_batch()
        frame_error = net.get_recon_errors(test_batch)
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
    return reg_scores, auc


if __name__ == "__main__":
    Config = ConfigParser.ConfigParser()
    Config.read('../config/config.ini')
    NUM_ITER = int(Config.get("Default", "NUM_ITER"))
    ALPHA = float(Config.get("Default", "ALPHA"))
    BATCH_SIZE = int(Config.get("Default", "BATCH_SIZE"))
    P_TRAIN = Config.get("Default", "P_TRAIN")
    P_TEST = Config.get("Default", "P_TEST")
    P_LABELS = Config.get("Default", "P_LABELS")
    logging.basicConfig(filename="STAE.log", level=logging.INFO)

    d = DataIterator(P_TRAIN, P_TEST, P_LABELS, batch_size=BATCH_SIZE)
    stae = SpatialTemporalAutoencoder(alpha=ALPHA, batch_size=BATCH_SIZE)

    regularity_scores, area_under_roc = train(d, stae)

    logging.info("Best area under the roc curve: {0:g}".format(area_under_roc))

    plt.figure()
    plt.plot(range(1, regularity_scores.shape[0] + 1), regularity_scores)
    plt.xlabel("Frame number")
    plt.ylabel("Regularity score")
    plt.savefig("../results/Regularity.png")

    np.save('../results/regularity_scores.npy', regularity_scores)
