from spatial_temporal_autoencoder import SpatialTemporalAutoencoder
from data_iterator import DataIterator
import ConfigParser
import numpy as np
from sklearn.metrics import roc_auc_score


def train(data, net):
    for i in xrange(NUM_ITER):
        tr_batch = data.get_train_batch()
        net.step(tr_batch)
        if i % 10 == 0:
            print("training batch reconstruction loss:", net.get_loss(tr_batch))
    return


def test(data, net):
    per_frame_error = [None] * data.get_test_size()
    while not data.check_data_exhausted():
        test_batch, frame_indices = data.get_test_batch()
        frame_error = net.get_recon_errors(test_batch)
        for i in xrange(frame_indices.shape[0]):
            for j in xrange(frame_indices.shape[1]):
                if frame_indices[i][j] != -1:
                    per_frame_error[frame_indices[i][j]].append(frame_error[i][j])

    per_frame_average_error = np.asarray(map(lambda x: np.mean(x), per_frame_error))
    abnorm_scores = per_frame_average_error - per_frame_average_error.min() / \
        (per_frame_average_error.max() - per_frame_average_error.min())
    reg_scores = 1 - abnorm_scores

    return abnorm_scores, reg_scores


if __name__ == "__main__":
    Config = ConfigParser.ConfigParser()
    Config.read('../config/config.ini')
    NUM_ITER = int(Config.get("Default", "NUM_ITER"))
    ALPHA = float(Config.get("Default", "ALPHA"))
    BATCH_SIZE = int(Config.get("Default", "BATCH_SIZE"))
    P_TRAIN = Config.get("Default", "P_TRAIN")
    P_TEST = Config.get("Default", "P_TEST")
    P_LABELS = Config.get("Default", "P_LABELS")

    d = DataIterator(P_TRAIN, P_TEST, P_LABELS, batch_size=BATCH_SIZE)
    stae = SpatialTemporalAutoencoder(alpha=ALPHA, batch_size=BATCH_SIZE)

    train(d, stae)
    abnormality_scores, regularity_scores = test(d, stae)

    auc = roc_auc_score(d.get_test_labels(), abnormality_scores)
    print("area under the roc curve:", auc)
