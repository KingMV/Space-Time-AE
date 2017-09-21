from spatial_temporal_autoencoder import SpatialTemporalAutoencoder
from data_iterator import DataIterator
import ConfigParser


def run():
    d = DataIterator(P_TRAIN, P_TEST, P_LABELS)
    stae = SpatialTemporalAutoencoder(alpha=ALPHA, batch_size=BATCH_SIZE)
    for i in xrange(NUM_ITER):
        tr_batch = d.get_train_batch(BATCH_SIZE)
        stae.step(tr_batch)
        if i % 10 == 0:
            print("training batch reconstruction loss: ", stae.get_loss(tr_batch))

    #test_iters = d.get_test_size() / (TVOL * BATCH_SIZE):
    #for i in xrange(d.get_test_size() / BATCH_SIZE):
    #    tr_batch = d.get_train_batch(BATCH_SIZE)
    #    stae.step(tr_batch)
    #    if i % 10 == 0:
    #        print("training batch reconstruction loss: ", stae.get_loss(tr_batch))
    #test, labels = d.get_test_batch()


if __name__ == "__main__":
    Config = ConfigParser.ConfigParser()
    Config.read('../config/config.ini')
    NUM_ITER = int(Config.get("Default", "NUM_ITER"))
    ALPHA = float(Config.get("Default", "ALPHA"))
    BATCH_SIZE = int(Config.get("Default", "BATCH_SIZE"))
    P_TRAIN = Config.get("Default", "P_TRAIN")
    P_TEST = Config.get("Default", "P_TEST")
    P_LABELS = Config.get("Default", "P_LABELS")
    run()
