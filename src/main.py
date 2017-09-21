from spatial_temporal_autoencoder import SpatialTemporalAutoencoder
from data_iterator import DataIterator
import ConfigParser
import numpy as np
from sklearn.metrics import roc_auc_score

def train(d, stae):
    for i in xrange(NUM_ITER):
        tr_batch = d.get_train_batch()
        stae.step(tr_batch)
        if i % 10 == 0:
            print("training batch reconstruction loss: ", stae.get_loss(tr_batch))
    return


def test(d, stae):
    per_frame_error = [] * d.get_test_size()
    while not d.check_data_exhausted():
        test_batch, frame_indices = d.get_test_batch()
        frame_error = stae.get_recon_errors(test_batch)
        for i in xrange(frame_indices.shape[0]):
            for j in xrange(frame_indices.shape[1]):
                if frame_indices[i][j] != -1:
                    per_frame_error[frame_indices[i][j]].append(frame_error[i][j])

    per_frame_average_error = np.asarray(map(lambda x: np.mean(x), per_frame_error))
    abnormality_scores = per_frame_average_error - per_frame_average_error.min() / \
                              (per_frame_average_error.max() - per_frame_average_error.min())
    regularity_scores = 1 - abnormality_scores

    return regularity_scores

def get_threshold(d):
    pass

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
    regularity_scores = test(d, stae)
