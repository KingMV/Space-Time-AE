from spatial_temporal_autoencoder import SpatialTemporalAutoencoder
from data_iterator import DataIterator

NUM_ITER = 200
ALPHA = 1e-4
BATCH_SIZE = 64
P_TRAIN = "../data/train.npy"
P_TEST = "../data/test.npy"
P_LABELS = "../data/labels.npy"

if __name__ == "__main__":
    # load NUM_ITER, ALPHA, BATCH_SIZE, P_TRAIN, P_TEST, P_LABELS from config file
    D = DataIterator(P_TRAIN, P_TEST, P_LABELS)
    stae = SpatialTemporalAutoencoder(alpha=ALPHA, batch_size=BATCH_SIZE)
    for i in xrange(NUM_ITER):
        tr_batch = D.get_train_batch(BATCH_SIZE)
        stae.step(tr_batch)
        if i % 10 == 0:
            print(stae.get_loss(tr_batch))
